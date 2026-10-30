"""
E4a of the thesis revision plan: the onion curves of the 3D grid with shells measured IN THE
PLANE of the film, as a probe on its surface would measure them.

run_onion_grid_task.py assigns nodes to shells by the 3D distance between node positions
(multimeter.onion_layers), so a node's depth enters its shell. This task redraws exactly the
same networks (same build_combos, same stable_seed(N, a, d, 'ext'), same draw order), refuses
to continue unless every regenerated R_eff reproduces onion_grid_results to 1e-9 and the 3D
curve reproduces the stored per-realization curve, and then recomputes each curve with shells
defined by the (x, y) distance. The pairwise resistances themselves are unchanged (same
Laplacian); only the shell membership changes.

Output: onion_inplane_results/chunk_NN.json, in the format of onion_grid_results (so
onion_tolerance_sweep.py --onion and curve_inversion.py --results read it unchanged), with
the 3D curves kept under per_realization_curves_3d.
Usage:
  python run_onion_inplane_task.py <task_id>          # one of N_TASKS chunks (array mode)
  python run_onion_inplane_task.py --parallel 48      # whole grid in one job
  python run_onion_inplane_task.py <task_id> --limit 1 --trials 2   # quick local check
"""
import sys, os, json, time, glob
if "--parallel" in sys.argv:
    # One BLAS thread per pool worker. This must run before numpy loads BLAS: set later,
    # it has no effect, and each of the 48 workers starts 48 BLAS threads of its own.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils")); sys.path.insert(0, HERE)
import run_onion_grid_task as og  # noqa: E402
from pipeline_utils_3d import generate_percolating_network_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402
from multimeter import onion_layers  # noqa: E402
from laplacian_utils import effective_resistance_from_reference  # noqa: E402

RESULTS_DIR = os.environ.get("INPLANE_OUT", os.path.join(HERE, "onion_inplane_results"))
REPRO_TOL = 1e-9


def stored_rows():
    rows = {}
    for p in sorted(glob.glob(os.path.join(HERE, "onion_grid_results", "chunk_*.json"))):
        for r in json.load(open(p)):
            rows[(r["N"], round(r["a"], 3), round(r["d"], 3))] = r
    return rows


def curves_for_network(res):
    inter = res["pre_merge_intersections"]; stn = res["pre_merge_segment_to_nodes"]
    ref = res["pre_merge_source_nodes"][0]
    r = effective_resistance_from_reference(inter, stn, ref, R0=og.R0)
    flat = [(p[0], p[1], 0.0) for p in inter]
    out = {}
    for name, pts in (("3d", inter), ("inplane", flat)):
        layers = onion_layers(pts, ref, layer_distances=og.LAYER_DISTANCES)
        lm = {}
        for dl, cand in layers.items():
            vals = [r[j] for j in cand if np.isfinite(r[j])]
            if vals:
                lm[dl] = float(np.mean(vals))
        out[name] = lm
    return out


def run_one_combo(N, a, d, n_trials=None, stored=None):
    n_trials = n_trials or og.N_TRIALS
    ref = (stored or {}).get((N, round(a, 3), round(d, 3)))
    np.random.seed(stable_seed(N, round(a, 3), round(d, 3), 'ext'))
    R_effs, c3, cp, attempts_total = [], [], [], 0
    while len(R_effs) < n_trials:
        res, att = generate_percolating_network_3d(N=N, a=a, contact_distance=og.CONTACT_DISTANCE,
                                                   box_dims=(1.0, 1.0, d), k_uniform=og.R0, max_attempts=og.MAX_ATTEMPTS)
        attempts_total += att
        if res is None:
            break
        R = float(distance_dependent_impedance(res, og.R0)["impedance"]); k = len(R_effs)
        cv = curves_for_network(res)
        if ref is not None and k < len(ref["R_eff_values"]):
            if abs(R - ref["R_eff_values"][k]) > REPRO_TOL * abs(ref["R_eff_values"][k]):
                raise RuntimeError(f"({N},{a},{d}) realization {k}: R_eff {R} != stored {ref['R_eff_values'][k]}")
            sc = ref["per_realization_curves"][k]
            for dl in og.LAYER_DISTANCES:
                s, m = sc.get(str(dl)), cv["3d"].get(dl)
                if (s is None) != (m is None) or (s is not None and abs(s - m) > 1e-7 * abs(s)):
                    raise RuntimeError(f"({N},{a},{d}) realization {k}: 3D shell {dl} {m} != stored {s}")
        R_effs.append(R); c3.append(cv["3d"]); cp.append(cv["inplane"])
    def pack(lms):
        return [{str(dl): lm.get(dl) for dl in og.LAYER_DISTANCES} for lm in lms]
    olc = {}
    for dl in og.LAYER_DISTANCES:
        v = [lm[dl] for lm in cp if dl in lm]
        if v:
            olc[str(dl)] = {"per_realization_means": v, "mean": float(np.mean(v)), "std": float(np.std(v)), "n_realizations": len(v)}
    return {"N": N, "a": a, "d": d, "n": len(R_effs), "per_realization_curves": pack(cp),
            "per_realization_curves_3d": pack(c3), "shell_distance": "inplane",
            "mean_ohm": float(np.mean(R_effs)) if R_effs else None, "median_ohm": float(np.median(R_effs)) if R_effs else None,
            "std_ohm": float(np.std(R_effs)) if R_effs else None, "perc_rate": len(R_effs) / attempts_total if attempts_total else 0.0,
            "R_eff_values": R_effs, "reference_node_convention": "pre_merge_source_nodes[0]",
            "layer_distances": og.LAYER_DISTANCES, "onion_layer_curves": olc,
            "reproduced_from": "onion_grid_results" if ref is not None else None}


def _job(args):
    N, a, d, nt = args
    t0 = time.time(); row = run_one_combo(N, a, d, nt, STORED); return (N, a, d), row, time.time() - t0


STORED = None



def _key(N, a, d):
    return (int(N), round(float(a), 6), round(float(d), 6))


def _resume(path):
    """rows already finished by an earlier run of this job (one JSON object per line)"""
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); rows[_key(r["N"], r["a"], r["d"])] = r
    return rows


def main():
    global STORED
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    trials = int(args[args.index("--trials") + 1]) if "--trials" in args else None
    combos = og.build_combos(); STORED = stored_rows()
    chunk = -(-len(combos) // og.N_TASKS)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args and args[0] == "--parallel":
        from multiprocessing import Pool
        part = os.path.join(RESULTS_DIR, "partial.jsonl")   # written as each combination finishes
        rows = _resume(part)
        todo = [c for c in (combos[:limit] if limit else combos) if _key(*c) not in rows]
        print(f"{len(rows)} combinations already in {part}; {len(todo)} to run", flush=True)
        with Pool(int(args[1])) as pool, open(part, "a") as fh:
            for i, (c, row, dt) in enumerate(pool.imap_unordered(_job, [(*c, trials) for c in todo]), 1):
                rows[_key(*c)] = row; fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{i}/{len(todo)}] N={c[0]} a={c[1]} d={c[2]} n={row['n']} {dt:.0f}s", flush=True)
        for t in range(og.N_TASKS):
            mine = [rows[_key(*c)] for c in combos[t * chunk:(t + 1) * chunk] if _key(*c) in rows]
            json.dump(mine, open(os.path.join(RESULTS_DIR, f"chunk_{t:02d}.json"), "w"), indent=1)
        return
    task_id = int(args[0]); mine = combos[task_id * chunk:(task_id + 1) * chunk]
    if limit: mine = mine[:limit]
    rows = []
    for (N, a, d) in mine:
        t0 = time.time(); row = run_one_combo(N, a, d, trials, STORED); rows.append(row)
        print(f"N={N} a={a} d={d} n={row['n']} rate={row['perc_rate']:.1%} {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json"), "w"), indent=1)
    print(f"[task {task_id}] {len(rows)} combos -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
