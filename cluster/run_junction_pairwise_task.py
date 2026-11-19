"""
Junction resistance and the pairwise resistances (paper item A).

The paper's inverse problem has two unknowns, (N, a), and the thesis's junction
model (utils/junction_model.py, Appendix F) adds a third, the junction-to-stick
ratio rho = R_j / (R0 a). With junctions the terminal resistance alone is one
equation for three unknowns, and the question this run asks is whether the
pairwise resistances separate a from rho (and N) at a fixed terminal resistance.

For every (N, a) cell the networks are drawn once, and every network is
measured at every rho of RHOS, so the comparison across rho is paired:

  S         terminal resistance of the junction graph, strips shorted to the
            electrodes (the convention of Appendix F, solve_two_terminal)
  C1,C4,C16 radial profiles from 1, 4 and 16 anchors (the anchor sets and shells of
            run_pairwise2d_task.network_features, applied to the copies of the
            junction graph)
  P         all-pairs profile, mean R_uv over the pairs of copies whose distance
            lies within 0.05 of each shell distance 0.1, ..., 0.9

Modelling choice (stated in the paper): the pairwise values come from the full
junction graph, every junction at R_j, the strip junctions included, because a
probe on a strip crossing measures through that crossing. S uses the shorted
strips of Appendix F, as the thesis does. rho = 0 is solved by contracting the
junctions (exact), and then reproduces run_pairwise2d_task.network_features up
to the doubling of the nodes (checked by --validate).

Output: <out>/partial.jsonl, one row per cell, appended as each finishes; a
resubmitted job skips the cells already there.

Usage:
  python run_junction_pairwise_task.py --validate [--units A07] [--trials 2]
  python run_junction_pairwise_task.py --grid pilot --parallel 16
  python run_junction_pairwise_task.py --grid full --parallel 16
  python run_junction_pairwise_task.py --grid pilot --units J_N100_a0.4 --trials 1 --out /tmp/jp
"""
import sys, os, json, time
if "--parallel" in sys.argv:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils")); sys.path.insert(0, HERE)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402
from junction_model import build_junction_network, solve_two_terminal  # noqa: E402

R0 = 1000.0
MAX_ATTEMPTS = 200
BOUNDARY = "clip"
TAG = "junction_pairwise"
TAG_CONFIRM = "junction_pairwise_confirm"      # fresh seeds for the pre-registered run
TAG_DENSE = "junction_pairwise_dense"          # fresh seeds for the dense grid
TAG_TRANSFER = "junction_pairwise_transfer"    # fresh seeds for the transfer run (PREREGISTRATION_JUNCTION_TRANSFER.md)
LAYERS = [round(0.1 * n, 2) for n in range(1, 10)]
TOL = 0.05
N_SOURCE_ANCHORS, N_INTERIOR_ANCHORS = 8, 8
MAX_NODES = 6500            # copies; a dense inverse of this size is ~340 MB per matrix
RHOS = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0]
SIGMA_LN, KAPPA, P_METAL = 1.5, 100.0, 1.0 / 3.0
RHOS_VAR = [0.3, 1.0, 3.0, 10.0, 30.0]


def uniform_conds(rhos):
    return [{"variant": "uniform", "rho": r} for r in rhos]


def variant_conds():
    """Lognormal junctions (arithmetic mean rho R0 a, log-sd SIGMA_LN) and the bimodal
    metallic/semiconducting rule (same kind: rho R0 a; mixed: KAPPA times that)."""
    return ([{"variant": f"lognormal_s{SIGMA_LN:g}", "rho": r} for r in RHOS_VAR] +
            [{"variant": f"bimodal_k{KAPPA:g}", "rho": r} for r in RHOS_VAR])


def transfer_units(n_geom=150, trials=40, seed=0):
    """Geometries off any grid: a log-uniform in [0.2, 1], N a^2 uniform in [5, 16], N = round(N a^2 / a^2).
    The junction ratios are drawn per network (see run_unit), so conds is the marker "draw"."""
    rng = np.random.default_rng(seed)
    units, seen = [], set()
    while len(units) < n_geom:
        a = round(float(10 ** rng.uniform(np.log10(0.2), 0.0)), 3)
        x = float(rng.uniform(5.0, 16.0))
        N = int(round(x / a ** 2))
        uid = f"T_N{N}_a{a:g}"
        if uid in seen:
            continue
        seen.add(uid)
        units.append({"id": uid, "N": N, "a": a, "x": N * a * a, "trials": trials, "conds": "draw", "tag": TAG_TRANSFER})
    return units


N_RHO_DRAWS = 4
RHO_LOG_RANGE = (float(np.log10(0.03)), 2.0)           # log10 rho in [log10 0.03, 2]
TRANSFER_RULES = ("uniform", f"lognormal_s{SIGMA_LN:g}", f"bimodal_k{KAPPA:g}")


def draw_conds(N, a, tag, k):
    """Four junction ratios per network, log-uniform and continuous, the same four for each rule."""
    rng = np.random.default_rng(stable_seed(N, round(a, 3), tag, k, "rho"))
    rhos = 10 ** rng.uniform(RHO_LOG_RANGE[0], RHO_LOG_RANGE[1], size=N_RHO_DRAWS)
    return [{"variant": v, "rho": float(r)} for v in TRANSFER_RULES for r in rhos]


GRIDS = {
    # a values, Na^2 values, networks per cell, conditions (variant, rho)
    "pilot": dict(a=[0.2, 0.4, 0.8], x=[5.0, 8.0, 12.0], trials=8, conds=uniform_conds([0.0, 1.0, 10.0]), tag=TAG),
    "full": dict(a=[0.2, 0.3, 0.4, 0.6, 0.8, 1.0], x=[5.0, 6.5, 8.0, 10.0, 12.0, 16.0], trials=30,
                 conds=uniform_conds(RHOS), tag=TAG),
    # the pre-registered confirmation: same cells as "full", fresh seeds, 60 networks,
    # the uniform rule plus the two variable-junction rules (PREREGISTRATION_JUNCTION.md)
    # the dense grid (PREREGISTRATION_JUNCTION_DENSE.md): uniform junctions only, 10 stick lengths,
    # 10 densities, 9 junction ratios (the lowest ones below the range tested so far), 100 networks per cell
    "dense": dict(a=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                  x=[5.0, 5.75, 6.5, 7.25, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0], trials=100,
                  conds=uniform_conds([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]), tag=TAG_DENSE),
    "confirm": dict(a=[0.2, 0.3, 0.4, 0.6, 0.8, 1.0], x=[5.0, 6.5, 8.0, 10.0, 12.0, 16.0], trials=60,
                    conds=uniform_conds(RHOS) + variant_conds(), tag=TAG_CONFIRM),
}


def _opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def plan(grid):
    if grid == "transfer":
        return transfer_units()
    g = GRIDS[grid]
    units, seen = [], set()
    for a in g["a"]:
        for x in g["x"]:
            N = int(round(x / a ** 2))
            uid = f"J_N{N}_a{a:g}"
            if uid in seen:                 # two densities that round to the same N are one cell
                continue
            seen.add(uid)
            units.append({"id": uid, "N": N, "a": a, "x": N * a * a,
                          "trials": g["trials"], "conds": g["conds"], "tag": g["tag"]})
    return units


# ----------------------------------------------------------------------------- one network
def junction_values(net, cond, a, rng):
    """Resistance of every junction (order of net['jun_u']), or None for the ideal contact."""
    rho, var = cond["rho"], cond["variant"]
    if var == "uniform":
        return None if rho == 0.0 else np.full(net["n_cross"], rho * R0 * a)
    if var.startswith("lognormal"):
        z = rng.standard_normal(net["n_cross"])
        return rho * R0 * a * np.exp(SIGMA_LN * z - 0.5 * SIGMA_LN ** 2)     # mean rho R0 a
    if var.startswith("bimodal"):
        sticks = sorted({s for n in net["crossings"] for s in net["sticks_of"][n]})
        metal = dict(zip(sticks, rng.random(len(sticks)) < P_METAL))
        mixed = np.array([metal[net["sticks_of"][n][0]] != metal[net["sticks_of"][n][1]] for n in net["crossings"]])
        return rho * R0 * a * np.where(mixed, KAPPA, 1.0)
    raise ValueError(var)


def main_component(net):
    """Copies of the largest connected component of the junction graph. Without the
    terminals merged, the graph can fall into several components (strip nodes joined
    to the rest only through the electrode); the Laplacian is then singular, so pairwise
    values are computed inside one component, as the E7 does by keeping the nodes
    reachable from the anchor. Junction edges join the copies of a crossing, so the
    components do not depend on rho or on the junction rule."""
    n = len(net["coords"])
    eu = np.concatenate([net["seg_u"], net["jun_u"]]); ev = np.concatenate([net["seg_v"], net["jun_v"]])
    _, lab = connected_components(sp.coo_matrix((np.ones(len(eu)), (eu, ev)), shape=(n, n)), directed=False)
    return np.where(lab == np.bincount(lab).argmax())[0]


def _pairwise_inverse(net, Rj, keep):
    """Pseudoinverse of the weighted Laplacian of the junction graph restricted to the
    copies `keep` (one connected component), and the map from those copies to the nodes
    it is built on (identity, or the contraction at Rj = None)."""
    n = len(net["coords"])
    if Rj is None:
        label = np.arange(n)
        label[net["jun_v"]] = net["jun_u"]
        uniq = {}
        m = np.array([uniq.setdefault(int(l), len(uniq)) for l in label])
        eu, ev, eR = net["seg_u"], net["seg_v"], R0 * net["seg_L"]
    else:
        m = np.arange(n)
        eu = np.concatenate([net["seg_u"], net["jun_u"]])
        ev = np.concatenate([net["seg_v"], net["jun_v"]])
        eR = np.concatenate([R0 * net["seg_L"], Rj])
    k = int(m.max()) + 1
    mu, mv = m[eu], m[ev]
    g = 1.0 / eR
    L = np.zeros((k, k))
    np.add.at(L, (mu, mu), g); np.add.at(L, (mv, mv), g)
    np.add.at(L, (mu, mv), -g); np.add.at(L, (mv, mu), -g)
    # the Laplacian is block diagonal over components; invert the block of `keep`
    sub = np.unique(m[keep])
    pos = np.full(k, -1); pos[sub] = np.arange(len(sub))
    Ls = L[np.ix_(sub, sub)]
    J = 1.0 / len(sub)
    Lp = np.linalg.inv(Ls + J) - J       # (L + 11'/k)^-1 = L+ + 11'/k for a connected graph
    return Lp, pos[m[keep]]


def pick_anchors(coords, rng):
    n = len(coords)
    src = np.where(coords[:, 0] < 0.1)[0]
    interior = np.where((coords[:, 0] >= 0.1) & (coords[:, 0] <= 0.9))[0]
    s0 = int(src[0])
    s_more = [int(v) for v in rng.permutation(src[1:])[:N_SOURCE_ANCHORS - 1]]
    i_more = [int(v) for v in rng.permutation(interior)[:N_INTERIOR_ANCHORS]]
    return [s0] + s_more + i_more, {"C1": [s0], "C4": [s0] + s_more[:1] + i_more[:2]}


def features_at(net, Rj, anchors, sets, D, keep):
    Lp, m = _pairwise_inverse(net, Rj, keep)
    dg = np.diag(Lp).copy()
    dgc, Lc = dg[m], None
    curves = {}
    for v in anchors:
        r = dgc[v] + dgc - 2.0 * Lp[m[v], m]
        r[v] = 0.0
        dv = D[v]
        cv = []
        for dl in LAYERS:
            sel = np.abs(dv - dl) <= TOL
            sel[v] = False
            cv.append(float(r[sel].mean()) if sel.any() else None)
        curves[v] = cv

    def shellwise(vs):
        out = []
        for i in range(len(LAYERS)):
            vals = [curves[v][i] for v in vs if curves[v][i] is not None]
            out.append(float(np.mean(vals)) if vals else None)
        return out

    n = len(m)
    Rm = dgc[:, None] + dgc[None, :] - 2.0 * Lp[np.ix_(m, m)]
    iu = np.triu(np.ones((n, n), dtype=bool), 1)
    P, Pn = [], []
    for dl in LAYERS:
        sel = (np.abs(D - dl) <= TOL) & iu
        k = int(sel.sum()); Pn.append(k)
        P.append(float(Rm[sel].mean()) if k else None)
    del Rm, iu
    return {"C1": shellwise(sets["C1"]), "C4": shellwise(sets["C4"]), "C16": shellwise(anchors),
            "P": P, "P_pairs": Pn}


def measure(res, a, conds, rng, seed_key=None):
    net = build_junction_network(res)
    n = len(net["coords"])
    if n > MAX_NODES:
        return None
    keep = main_component(net)
    coords = net["coords"][keep]
    anchors, sets = pick_anchors(coords, rng)      # indices into `keep`
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    out = {"n_nodes": int(n), "n_main": int(len(keep)), "n_cross": int(net["n_cross"]), "anchors": anchors,
           "by_cond": []}
    for ci, cond in enumerate(conds):
        # junction draws use their own generator, per network and condition, so they never
        # touch the stream of the geometry or of the anchors
        rj_rng = np.random.default_rng(stable_seed(*(seed_key or (0,)), "rj", ci))
        Rj = junction_values(net, cond, a, rj_rng)
        sol = solve_two_terminal(net, R0, 0.0 if Rj is None else Rj)
        f = features_at(net, Rj, anchors, sets, D, keep)
        f.update({"rho": cond["rho"], "variant": cond["variant"], "S": float(sol["R_eff"])})
        out["by_cond"].append(f)
    return out


# ----------------------------------------------------------------------------- one cell
def run_unit(u, n_trials=None):
    t0 = time.time()
    N, a = int(u["N"]), float(u["a"])
    n_trials = n_trials or u["trials"]
    tag = u["tag"]
    np.random.seed(stable_seed(N, round(a, 3), tag))
    nets, S_wire, attempts_total, skipped = [], [], 0, 0
    while len(nets) < n_trials:
        res, att = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS, boundary=BOUNDARY)
        attempts_total += att
        if res is None:
            break
        k = len(nets) + skipped
        rng = np.random.default_rng(stable_seed(N, round(a, 3), tag, k, "anchors"))
        conds = draw_conds(N, a, tag, k) if u["conds"] == "draw" else u["conds"]
        m = measure(res, a, conds, rng, seed_key=(N, round(a, 3), tag, k))
        if m is None:
            skipped += 1
            if skipped > 3 * n_trials:
                break
            continue
        m["wire_check"] = float(distance_dependent_impedance(res, R0)["impedance"])
        nets.append(m)
    return {"id": u["id"], "N": N, "a": a, "x": u["x"], "conds": u["conds"], "tag": tag, "n": len(nets),
            "skipped_too_large": skipped, "attempts_total": attempts_total,
            "layer_distances": LAYERS, "tolerance": TOL, "seconds": time.time() - t0, "networks": nets}


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


# ----------------------------------------------------------------------------- validation
def validate(args):
    """rho = 0 against the thesis: regenerate E7 Part A networks from the seeds of the
    validation run, check the terminal resistance against the stored value, and the
    all-pairs profile against the stored one (anchor-independent)."""
    import run_pairwise2d_task as rp
    pl = json.load(open(rp.PLAN_PATH))["units"]
    want = (_opt(args, "--units") or "A07").split(",")
    trials = int(_opt(args, "--trials", 2))
    stored_path = _opt(args, "--stored") or os.path.join(HERE, "pairwise2d_results", "partial.jsonl")
    stored = _resume(stored_path)
    worst_S = worst_P = 0.0
    for u in [x for x in pl if x["id"] in want]:
        N, a = int(u["N"]), float(u["a"])
        np.random.seed(stable_seed(N, round(a, 3), "inverse_validation", BOUNDARY))
        for k in range(trials):
            res, _ = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS, boundary=BOUNDARY)
            rng = np.random.default_rng(0)
            m = measure(res, a, uniform_conds([0.0]), rng)
            f = m["by_cond"][0]
            ref_S = u["stored_R_eff_values"][k]
            dS = abs(f["S"] - ref_S) / ref_S
            st = stored[u["id"]]
            dP = max(abs(p - q) / q for p, q in zip(f["P"], st["P"][k]) if p is not None and q)
            print(f"{u['id']} net {k}: nodes {m['n_nodes']} (E7: {st['n_nodes'][k]})  "
                  f"S rel.diff {dS:.2e}  P max rel.diff {dP:.2e}", flush=True)
            worst_S, worst_P = max(worst_S, dS), max(worst_P, dP)
    print(f"worst: S {worst_S:.2e}, P {worst_P:.2e}")
    if worst_S > 1e-9 or worst_P > 1e-6:
        raise SystemExit("validation FAILED")
    print("validate: rho = 0 reproduces the thesis")


def main():
    args = sys.argv[1:]
    if "--validate" in args:
        validate(args); return
    grid = _opt(args, "--grid", "pilot")
    units = plan(grid)
    trials = int(_opt(args, "--trials")) if "--trials" in args else None
    out = _opt(args, "--out") or os.path.join(HERE, f"junction_pairwise_{grid}_results")
    if "--units" in args:
        want = _opt(args, "--units").split(",")
        units = [u for u in units if u["id"] in want]
    os.makedirs(out, exist_ok=True)
    part = os.path.join(out, "partial.jsonl")
    done = _resume(part)
    todo = sorted([u for u in units if u["id"] not in done], key=lambda u: -(u["N"] * u["a"]) ** 2)
    print(f"{len(done)} cells already in {part}; {len(todo)} to run", flush=True)
    nproc = int(_opt(args, "--parallel", 1))
    with open(part, "a") as fh:
        if nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as pool:
                for i, (uid, row) in enumerate(pool.imap_unordered(_job, [(u, trials) for u in todo]), 1):
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{i}/{len(todo)}] {uid} n={row['n']} skipped={row['skipped_too_large']} {row['seconds']:.0f}s", flush=True)
        else:
            for i, u in enumerate(todo, 1):
                row = run_unit(u, trials)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{i}/{len(todo)}] {u['id']} n={row['n']} skipped={row['skipped_too_large']} {row['seconds']:.0f}s", flush=True)
    print(f"done -> {part}")


if __name__ == "__main__":
    main()
