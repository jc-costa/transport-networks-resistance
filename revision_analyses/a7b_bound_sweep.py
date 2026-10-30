"""A7b: the Dirichlet (affine-potential) bound R_eff * P_aff >= 1 on fresh clipping-rule networks across the
density range of the 2D sweeps (revision plan, M11/M18). Uses the measurement code of the E2 task
(run_redbond_scaling_task.affine_bound_ratio) and generate_percolating_network, seeded with the tag 'bound'.
Usage: python a7b_bound_sweep.py <a> <N> [<N> ...] [--n 10]   -> appends to a7b_result.jsonl"""
import sys, os, json, time, warnings, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "cluster")
sys.path.insert(0, os.path.join(CL, "..", "utils")); sys.path.insert(0, CL)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa
from run_redbond_scaling_task import affine_bound_ratio, R0  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("a", type=float); ap.add_argument("N", type=int, nargs="+"); ap.add_argument("--n", type=int, default=10)
args = ap.parse_args()
for N in args.N:
    a = args.a; np.random.seed(stable_seed(N, round(a, 3), "bound")); ratios, Rs, nodes, draws, t0 = [], [], [], 0, time.time()
    while len(ratios) < args.n and draws < 3000:
        res, att = generate_percolating_network(N, a, k_uniform=R0, max_attempts=500); draws += att
        if res is None: continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning); rl = distance_dependent_impedance(res, R0)
        R = float(rl["impedance"]); ratios.append(affine_bound_ratio(res, R)); Rs.append(R); nodes.append(len(res["pre_merge_intersections"]))
    row = {"N": N, "a": a, "x": N * a * a, "n": len(ratios), "draws": draws, "ratios": ratios, "R": Rs, "nodes": nodes}
    open(os.path.join(HERE, "a7b_result.jsonl"), "a").write(json.dumps(row) + "\n")
    r = np.array(ratios)
    print(f"a={a} N={N} Na2={N*a*a:6.2f} n={len(r)} min={r.min() if len(r) else float('nan'):.3f} median={np.median(r) if len(r) else float('nan'):.3f} max nodes={max(nodes) if nodes else 0} {time.time()-t0:.0f}s", flush=True)
