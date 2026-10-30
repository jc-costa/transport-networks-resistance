"""
Near-threshold sweep of the 2D stick network at three box sizes, to pin the
conductivity exponent t of the critical form

    R_eff = A * a^q * (N a^2 - c)^(-t)

that inverse_N_a_from_R_eff.ipynb section 3.2 fits on the main sweeps. Those
sweeps span N a^2 from 3 to 85 in a box only 1-5 stick lengths across, so t
there is an effective exponent over the whole range and c is a finite-size
threshold. This sweep samples N a^2 densely from 4 to 20 (12 at a = 0.1, where
the networks exceed ~7000 nodes) in boxes 5, 6.7 and 10 stick lengths across,
so that t can be fitted close to the threshold and its drift with box size
seen. One array task per (a, N a^2) point; each writes
critical_sweep_results/point_NN.json.

Usage:
  python run_critical_sweep_task.py --plan       # print the 36 points and exit
  python run_critical_sweep_task.py <task_id>    # run one point

Seeds are stable_seed(N, a, 'critical'), so any point can be regenerated.
"""
import sys
import os
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "critical_sweep_results")

sys.path.insert(0, UTILS_PATH)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402

R0 = 1000.0
N_TRIALS = 30          # percolating realizations wanted per point
MAX_ATTEMPTS = 500     # per generate_percolating_network call
DRAW_BUDGET = 6000     # total draws per point; near the threshold most fail

# (a, list of N a^2 targets). Steps of 1 up to 10, of 2 beyond, as in the
# thesis; a = 0.1 stops at 12 to keep the dense solve under ~7000 nodes.
X_COARSE = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
GRID = [
    (0.20, X_COARSE),
    (0.15, X_COARSE),
    (0.10, [4, 5, 6, 7, 8, 9, 10, 11, 12]),
]


def build_plan():
    plan = []
    for a, xs in GRID:
        for x in xs:
            N = int(round(x / a ** 2))
            plan.append({"a": a, "x_target": x, "N": N, "x": N * a ** 2})
    return plan


def run_point(task_id):
    plan = build_plan()
    if task_id >= len(plan):
        print(f"task_id {task_id} out of range (plan has {len(plan)} points)")
        return
    pt = plan[task_id]
    N, a = pt["N"], pt["a"]
    print(f"[task {task_id}] a={a}  N={N}  N a^2={pt['x']:.2f}  box={1/a:.1f} stick lengths across", flush=True)

    np.random.seed(stable_seed(N, round(a, 3), "critical"))
    R_effs, n_nodes, n_edges = [], [], []
    total_attempts = 0
    t_start = time.time()
    while len(R_effs) < N_TRIALS and total_attempts < DRAW_BUDGET:
        res, attempts = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS)
        total_attempts += attempts
        if res is None:
            # a 500-draw block without percolation: keep going until the
            # global budget is spent, this is exactly the regime we want
            continue
        rl = distance_dependent_impedance(res, R0)
        R_effs.append(float(rl["impedance"]))
        n_nodes.append(len(res["merged_intersections"]))
        n_edges.append(len(res["merged_segment_to_nodes"]))
        print(f"  {len(R_effs):2d}/{N_TRIALS}  R_eff={R_effs[-1]:9.2f}  nodes={n_nodes[-1]:5d}  "
              f"draws so far={total_attempts:5d}  {time.time()-t_start:6.0f}s", flush=True)

    out = {
        "task_id": task_id, "a": a, "N": N, "x": pt["x"], "x_target": pt["x_target"],
        "R0": R0, "n": len(R_effs), "n_target": N_TRIALS,
        "total_draws": total_attempts, "perc_rate": len(R_effs) / total_attempts if total_attempts else 0.0,
        "R_eff_values": R_effs, "n_nodes": n_nodes, "n_edges": n_edges,
        "seconds": time.time() - t_start,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"point_{task_id:02d}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[task {task_id}] done: {len(R_effs)} realizations from {total_attempts} draws "
          f"(rate {out['perc_rate']:.3%}), mean R_eff={np.mean(R_effs) if R_effs else float('nan'):.1f}, "
          f"{out['seconds']:.0f}s -> {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == "--plan":
        for i, pt in enumerate(build_plan()):
            print(f"{i:2d}  a={pt['a']:.2f}  N={pt['N']:5d}  N a^2={pt['x']:6.2f}")
        sys.exit(0)
    run_point(int(sys.argv[1]))
