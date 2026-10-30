"""
E4b of the thesis revision plan: a quasi-three-dimensional control for the film-thickness
result. In run_onion_grid_task.py the sticks are oriented uniformly on the sphere and
clipped at all six faces of the slab, so a slab thinner than a stick is made of fragments
whose length d controls. Here every stick lies FLAT (parallel to the substrate), with its
centre at a uniform height z in [0, d], and is clipped only at the four side faces of the
film; d then changes the stacking of whole sticks and nothing else, and d -> 0 recovers the
two-dimensional film with contact distance delta_c.

Everything else is the onion grid: same build_combos (N, a), the same d values plus d = 0.02
(below delta_c = 0.03, the thin limit), the same contact criterion, electrodes, pruning,
distance-dependent solve, reference node and 24 realizations per combination. Each row
carries the scalar R_eff and the onion curve with shells measured IN THE PLANE (the format
of onion_grid_results, so onion_tolerance_sweep.py and curve_inversion.py read it), and the
3D-distance curve under per_realization_curves_3d.

Seeds: stable_seed(N, a, d, 'flat'). Output: onion_flat_results/chunk_NN.json.
Usage:
  python run_flat_sticks_task.py <task_id>             # array mode, N_TASKS chunks
  python run_flat_sticks_task.py --parallel 48         # whole grid in one job
  python run_flat_sticks_task.py <task_id> --limit 1 --trials 2
"""
import sys, os, json, time, io, contextlib
if "--parallel" in sys.argv:
    # One BLAS thread per pool worker. This must run before numpy loads BLAS: set later,
    # it has no effect, and each of the 48 workers starts 48 BLAS threads of its own.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils")); sys.path.insert(0, HERE)
import run_onion_grid_task as og  # noqa: E402
from run_onion_inplane_task import curves_for_network  # noqa: E402
from pipeline_utils_3d import run_full_pipeline_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402
from point_generator import clip_segment_to_box  # noqa: E402

RESULTS_DIR = os.environ.get("FLAT_OUT", os.path.join(HERE, "onion_flat_results"))
D_VALUES = [0.02] + list(og.D_VALUES)


def build_combos():
    base = sorted({(N, a) for (N, a, _) in og.build_combos()}, key=lambda t: (t[1], t[0]))
    return [(N, a, d) for (N, a) in base for d in D_VALUES]


def flat_points(N, a, d):
    """Flat sticks: centres uniform in [-a/2, 1+a/2]^2 (round(N (1+a)^2) of them, as the 2D clip
    rule), height z ~ U(0, d), in-plane orientation; clipped to the film's side faces."""
    p1, p2 = [], []
    for _ in range(int(round(N * (1.0 + a) ** 2))):
        cx, cy = np.random.uniform(-a / 2, 1 + a / 2, 2); z = np.random.uniform(0.0, d)
        al = np.random.uniform(0, 2 * np.pi); hx, hy = 0.5 * a * np.cos(al), 0.5 * a * np.sin(al)
        c = clip_segment_to_box((cx - hx, cy - hy, z), (cx + hx, cy + hy, z), (0.0, 0.0, 0.0), (1.0, 1.0, d))
        if c is not None:
            p1.append(c[0]); p2.append(c[1])
    return p1, p2


def percolating_flat(N, a, d, max_attempts):
    for att in range(1, max_attempts + 1):
        p1, p2 = flat_points(N, a, d)
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_full_pipeline_3d(p1, p2, N, a, og.CONTACT_DISTANCE, k=og.R0, verbose=False)
        if res.get("has_percolation") and "merged_segment_to_nodes" in res:
            return res, att
    return None, max_attempts


def run_one_combo(N, a, d, n_trials=None):
    n_trials = n_trials or og.N_TRIALS
    np.random.seed(stable_seed(N, round(a, 3), round(d, 3), 'flat'))
    R_effs, cp, c3, total, nodes = [], [], [], 0, []
    while len(R_effs) < n_trials:
        res, att = percolating_flat(N, a, d, og.MAX_ATTEMPTS); total += att
        if res is None:
            break
        R_effs.append(float(distance_dependent_impedance(res, og.R0)["impedance"]))
        cv = curves_for_network(res); cp.append(cv["inplane"]); c3.append(cv["3d"])
        nodes.append(len(res["pre_merge_intersections"]))
    pack = lambda lms: [{str(dl): lm.get(dl) for dl in og.LAYER_DISTANCES} for lm in lms]
    olc = {}
    for dl in og.LAYER_DISTANCES:
        v = [lm[dl] for lm in cp if dl in lm]
        if v:
            olc[str(dl)] = {"per_realization_means": v, "mean": float(np.mean(v)), "std": float(np.std(v)), "n_realizations": len(v)}
    return {"N": N, "a": a, "d": d, "n": len(R_effs), "model": "flat sticks at uniform height", "shell_distance": "inplane",
            "per_realization_curves": pack(cp), "per_realization_curves_3d": pack(c3),
            "mean_ohm": float(np.mean(R_effs)) if R_effs else None, "median_ohm": float(np.median(R_effs)) if R_effs else None,
            "std_ohm": float(np.std(R_effs)) if R_effs else None, "perc_rate": len(R_effs) / total if total else 0.0,
            "R_eff_values": R_effs, "mean_nodes": float(np.mean(nodes)) if nodes else None,
            "reference_node_convention": "pre_merge_source_nodes[0]", "layer_distances": og.LAYER_DISTANCES,
            "onion_layer_curves": olc}


def _job(args):
    N, a, d, nt = args; t0 = time.time(); return (N, a, d), run_one_combo(N, a, d, nt), time.time() - t0



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
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    trials = int(args[args.index("--trials") + 1]) if "--trials" in args else None
    combos = build_combos(); chunk = -(-len(combos) // og.N_TASKS)
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
                print(f"[{i}/{len(todo)}] N={c[0]} a={c[1]} d={c[2]} n={row['n']} rate={row['perc_rate']:.1%} {dt:.0f}s", flush=True)
        for t in range(og.N_TASKS):
            mine = [rows[_key(*c)] for c in combos[t * chunk:(t + 1) * chunk] if _key(*c) in rows]
            json.dump(mine, open(os.path.join(RESULTS_DIR, f"chunk_{t:02d}.json"), "w"), indent=1)
        return
    task_id = int(args[0]); mine = combos[task_id * chunk:(task_id + 1) * chunk]
    if limit: mine = mine[:limit]
    rows = []
    for (N, a, d) in mine:
        t0 = time.time(); row = run_one_combo(N, a, d, trials); rows.append(row)
        print(f"N={N} a={a} d={d} n={row['n']} rate={row['perc_rate']:.1%} nodes={row['mean_nodes']} {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
