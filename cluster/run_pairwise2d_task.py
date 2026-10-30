"""
E7 of the thesis revision plan: is there information beyond the terminal resistance in a
2D network's pairwise resistances?

Part A (same terminal R, different (N, a)): the 22 validation candidates of
validation_plan.json. Their networks are REGENERATED from the seeds of
run_validation_task.py with exactly its draw sequence (stable_seed(N, round(a, 3),
'inverse_validation', 'clip'), generate_percolating_network(N, a, k_uniform=1000,
max_attempts=200, boundary='clip') until 30 networks or a give-up). Every regenerated
terminal R_eff must reproduce validation_results/candidate_NN.json to 1e-9 (relative), and
the regenerated count must match the stored count, or the task stops with an error.

Part B (forward grid): the 101 (N, a) cells of the pooled 2D sweeps behind the fitted law
(the cells pool_and_fit() in run_validation_task.py pools), with FRESH draws seeded by
stable_seed(N, round(a, 3), 'pairwise2d'), 30 percolating networks per cell, same generator
call and give-up rule as Part A.

Per network, ONE pseudoinverse of the weighted pre-merge Laplacian (R_edge = 1000 x length,
laplacian_utils._weighted_laplacian_distance_dependent, the Laplacian of
effective_resistance_from_reference) gives every R_ij = L+_ii + L+_jj - 2 L+_ij, and from it:
  S     terminal R_eff (distance_dependent_impedance, the convention of the thesis)
  anchor curves: the onion curve (mean R_eff from the anchor to the nodes of each shell
        0.1, ..., 0.9, half-width 0.05, multimeter.onion_layers) from up to 16 anchors:
          s0          = pre_merge_source_nodes[0] (the reference node of the thesis)
          7 more      source-strip nodes (x < 0.1), drawn without replacement
          8           interior nodes (0.1 <= x <= 0.9), drawn without replacement
        C1  = curve of s0 alone
        C4  = shell-wise mean over s0, the first extra source node, the first 2 interior nodes
        C16 = shell-wise mean over all anchors
        (nested sets; an anchor with an empty shell is skipped in that shell's mean)
  P     all-pairs profile: mean R_ij over the node pairs i < j whose distance lies within
        0.05 of each shell distance 0.1, ..., 0.9 (same rule as onion_layers)
Anchors are drawn with their own Generator, seeded stable_seed(N, round(a, 3), tag, k,
'anchors'), so they never touch the global numpy stream the networks are drawn from.
Raw values (ohm) are stored; logs are taken in pairwise2d_analysis.py.

Inputs : pairwise2d_plan.json, built once with --prepare on a machine that has
         validation_plan.json, validation_results/ and ../experiments/ (the stored Part A
         values and the Part B cell list are copied into the plan, so the cluster needs
         nothing else).
Output : <out>/partial.jsonl, one JSON object per candidate/cell, appended as each finishes;
         a resubmitted job skips the units already there.
Usage:
  python run_pairwise2d_task.py --prepare                      # once, writes the plan
  python run_pairwise2d_task.py --parallel 48                  # whole plan in one job
  python run_pairwise2d_task.py --units A07,B_N50_a0.5 --trials 2 --out /tmp/pw   # quick check
"""
import sys, os, json, time, glob
if "--parallel" in sys.argv:
    # One BLAS thread per pool worker. Must run before numpy loads BLAS.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils")); sys.path.insert(0, HERE)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402
from laplacian_utils import _weighted_laplacian_distance_dependent, _bfs_reachable  # noqa: E402
from multimeter import onion_layers  # noqa: E402

R0 = 1000.0
N_TRIALS = 30
MAX_ATTEMPTS = 200          # per network, = MAX_ATTEMPTS_VAL of run_validation_task.py
BOUNDARY = "clip"
LAYERS = [round(0.1 * n, 2) for n in range(1, 10)]
TOL = min(b - a for a, b in zip(LAYERS, LAYERS[1:])) / 2      # onion_layers' default half-width
N_SOURCE_ANCHORS, N_INTERIOR_ANCHORS = 8, 8
REPRO_TOL = 1e-9
PLAN_PATH = os.path.join(HERE, "pairwise2d_plan.json")


def _opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


# ----------------------------------------------------------------------------- plan
def prepare():
    vp = json.load(open(os.path.join(HERE, "validation_plan.json")))
    units = []
    for i, c in enumerate(vp["plan"]):
        st = json.load(open(os.path.join(HERE, "validation_results", f"candidate_{i:02d}.json")))
        if st["task_id"] != i or st["N"] != c["N"] or abs(st["a"] - c["a"]) > 1e-12:
            raise SystemExit(f"validation_results/candidate_{i:02d}.json does not match plan entry {i}")
        units.append({"part": "A", "id": f"A{i:02d}", "candidate_index": i, "N": c["N"], "a": c["a"],
                      "R_target": c["R_target"], "tag": c["tag"],
                      "stored_R_eff_values": st["R_eff_values"]})
    # Part B: the (N, a) cells pool_and_fit() pools (post-fix, boundary 'clip', duplicates skipped)
    import run_validation_task as rv
    paths = [p for p in sorted(glob.glob(os.path.join(rv.EXPERIMENTS_DIR, "exp_*",
             "effective_resistance_sweep_summary*.json"))) if rv._is_post_fix(p)]
    pooled, seen = {}, set()
    for p in paths:
        for rec in json.load(open(p)).get("sweep", []):
            vals = rec.get("R_eff_values") or []
            if not vals:
                continue
            key = (int(rec["N"]), round(float(rec["a"]), 3))
            if (key, tuple(vals)) in seen:
                continue
            seen.add((key, tuple(vals))); pooled[key] = pooled.get(key, 0) + len(vals)
    if len(pooled) != vp["fit"]["n_points"]:
        raise SystemExit(f"{len(pooled)} pooled cells, but the fit in validation_plan.json used {vp['fit']['n_points']}")
    for (N, a) in sorted(pooled):
        units.append({"part": "B", "id": f"B_N{N}_a{a:g}", "N": N, "a": a,
                      "pooled_n": pooled[(N, a)], "nodes_est": 0.5 * N * N * a * a})
    fit = {k: v for k, v in vp["fit"].items() if k != "R_mean_pts"}
    json.dump({"experiment": "E7 pairwise 2D", "sweep_files": [os.path.relpath(p, HERE) for p in paths],
               "fit": fit, "units": units}, open(PLAN_PATH, "w"), indent=1)
    nA = sum(u["part"] == "A" for u in units)
    print(f"wrote {PLAN_PATH}: {nA} Part A candidates, {len(units) - nA} Part B cells")


# ----------------------------------------------------------------------------- one network
def network_features(res, rng):
    inter_list = res["pre_merge_intersections"]
    inter = np.asarray(inter_list, dtype=float)
    stn = res["pre_merge_segment_to_nodes"]
    src = [int(v) for v in res["pre_merge_source_nodes"]]
    n = len(inter)
    L = _weighted_laplacian_distance_dependent(inter_list, stn, R0=R0)
    Lp = np.linalg.pinv(L)
    dg = np.diag(Lp).copy()
    s0 = src[0]
    reach = _bfs_reachable(n, stn, s0)

    src_pool = np.array([v for v in src[1:] if reach[v]], dtype=int)
    interior = np.where((inter[:, 0] >= 0.1) & (inter[:, 0] <= 0.9) & reach)[0]
    s_more = [int(v) for v in rng.permutation(src_pool)[:N_SOURCE_ANCHORS - 1]]
    i_more = [int(v) for v in rng.permutation(interior)[:N_INTERIOR_ANCHORS]]
    anchors = [s0] + s_more + i_more
    roles = ["s0"] + ["source"] * len(s_more) + ["interior"] * len(i_more)
    sets = {"C1": [s0], "C4": [s0] + s_more[:1] + i_more[:2], "C16": anchors}

    curves = {}
    for v in anchors:
        r = dg[v] + dg - 2.0 * Lp[v, :]
        r[v] = 0.0
        r[~reach] = np.inf
        layers = onion_layers(inter_list, v, layer_distances=LAYERS)
        cv = []
        for dl in LAYERS:
            vals = [r[j] for j in layers[dl] if np.isfinite(r[j])]
            cv.append(float(np.mean(vals)) if vals else None)
        curves[v] = cv

    def shellwise(vs):
        out = []
        for i in range(len(LAYERS)):
            vals = [curves[v][i] for v in vs if curves[v][i] is not None]
            out.append(float(np.mean(vals)) if vals else None)
        return out

    idx = np.where(reach)[0]
    iu, ju = np.triu_indices(len(idx), 1)
    I, J = idx[iu], idx[ju]
    del iu, ju
    dist = np.linalg.norm(inter[I] - inter[J], axis=1)
    rij = dg[I] + dg[J] - 2.0 * Lp[I, J]
    P, Pn = [], []
    for dl in LAYERS:
        m = np.abs(dist - dl) <= TOL
        k = int(m.sum()); Pn.append(k)
        P.append(float(rij[m].mean()) if k else None)

    return {"n_nodes": n, "n_reachable": int(reach.sum()), "n_source": len(src),
            "n_interior": int(len(interior)), "anchors": anchors, "anchor_roles": roles,
            "anchor_curves": [curves[v] for v in anchors],
            "C1": shellwise(sets["C1"]), "C4": shellwise(sets["C4"]), "C16": shellwise(sets["C16"]),
            "P": P, "P_pairs": Pn}


# ----------------------------------------------------------------------------- one unit
def run_unit(u, n_trials=None):
    t0 = time.time()
    N, a = int(u["N"]), float(u["a"])
    n_trials = n_trials or N_TRIALS
    if u["part"] == "A":
        np.random.seed(stable_seed(N, round(a, 3), "inverse_validation", BOUNDARY))
        tag, stored = "inverse_validation", u["stored_R_eff_values"]
    else:
        np.random.seed(stable_seed(N, round(a, 3), "pairwise2d"))
        tag, stored = "pairwise2d", None
    R, feats, attempts_total = [], [], 0
    while len(R) < n_trials:
        res, att = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS, boundary=BOUNDARY)
        attempts_total += att
        if res is None:
            break
        Rk = float(distance_dependent_impedance(res, R0)["impedance"]); k = len(R)
        if stored is not None:
            if k >= len(stored):
                raise RuntimeError(f"{u['id']} (N={N}, a={a}): network {k} exists, but only {len(stored)} are stored")
            if abs(Rk - stored[k]) > REPRO_TOL * abs(stored[k]):
                raise RuntimeError(f"{u['id']} (N={N}, a={a}) network {k}: R_eff {Rk!r} != stored {stored[k]!r}")
        rng = np.random.default_rng(stable_seed(N, round(a, 3), tag, k, "anchors"))
        R.append(Rk); feats.append(network_features(res, rng))
    if stored is not None and n_trials >= N_TRIALS and len(R) != len(stored):
        raise RuntimeError(f"{u['id']}: regenerated {len(R)} networks, {len(stored)} stored")
    row = {k: u[k] for k in u if k != "stored_R_eff_values"}
    row.update({"seed_tag": tag, "n": len(R), "R_eff_values": R,
                "reproduced_stored": (len(R) if stored is not None else None),
                "attempts_total": attempts_total, "perc_rate": len(R) / attempts_total if attempts_total else 0.0,
                "layer_distances": LAYERS, "tolerance": TOL, "seconds": time.time() - t0})
    for key in ("n_nodes", "n_reachable", "n_source", "n_interior", "anchors", "anchor_roles",
                "anchor_curves", "C1", "C4", "C16", "P", "P_pairs"):
        row[key] = [f[key] for f in feats]
    return row


def _job(args):
    u, nt = args
    return u["id"], run_unit(u, nt)


def _resume(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); rows[r["id"]] = r
    return rows


def main():
    args = sys.argv[1:]
    if "--prepare" in args:
        prepare(); return
    plan = json.load(open(PLAN_PATH))["units"]
    trials = int(_opt(args, "--trials")) if "--trials" in args else None
    out = _opt(args, "--out") or os.environ.get("PAIRWISE2D_OUT", os.path.join(HERE, "pairwise2d_results"))
    if "--units" in args:
        want = _opt(args, "--units").split(",")
        units = [u for u in plan if u["id"] in want]
        missing = set(want) - {u["id"] for u in units}
        if missing:
            raise SystemExit(f"unknown unit id(s): {sorted(missing)}")
    else:
        units = plan
    os.makedirs(out, exist_ok=True)
    part = os.path.join(out, "partial.jsonl")
    done = _resume(part)
    # most expensive first (node count ~ 0.5 N^2 a^2), so the pool does not end on one long unit
    todo = sorted([u for u in units if u["id"] not in done], key=lambda u: -(u["N"] * u["a"]) ** 2)
    print(f"{len(done)} units already in {part}; {len(todo)} to run", flush=True)
    nproc = int(_opt(args, "--parallel", 1))
    with open(part, "a") as fh:
        if nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as pool:
                it = pool.imap_unordered(_job, [(u, trials) for u in todo])
                for i, (uid, row) in enumerate(it, 1):
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{i}/{len(todo)}] {uid} N={row['N']} a={row['a']} n={row['n']} "
                          f"nodes<={max(row['n_nodes'], default=0)} {row['seconds']:.0f}s", flush=True)
        else:
            for i, u in enumerate(todo, 1):
                row = run_unit(u, trials)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{i}/{len(todo)}] {u['id']} N={row['N']} a={row['a']} n={row['n']} "
                      f"nodes<={max(row['n_nodes'], default=0)} {row['seconds']:.0f}s", flush=True)
    print(f"done -> {part}")


if __name__ == "__main__":
    main()
