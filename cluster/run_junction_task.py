"""
Junction resistance on the near-threshold sweep: the crossover from the
wire-limited to the junction-limited regime, and the red junctions.

Redraws the networks of run_critical_sweep_task.py from their seeds (checked
against critical_sweep_results/ to a relative 1e-9), builds the two-node-per-
crossing junction graph of utils/junction_model.py for each, and solves it at
every junction-to-stick ratio

    rho = R_j / (R0 a)   in   RHOS

(rho = 0 is the wire-limited model, solved by contracting the junctions, and
must reproduce the stored R_eff). Per network and rho it records R_eff, the
share of R_eff in the red bonds split into red segments and red junctions,
and the number of segments/junctions carrying the whole current. The red-bond
sets are topological (independent of rho > 0) and are recorded once.

Variant 1 of the junction study (uniform R_j). Distributed and bimodal
junction resistances are handled by the notebook on top of the same graphs.

Usage:
  python run_junction_task.py --plan
  python run_junction_task.py <task_id> [--max-trials 3]
"""
import sys
import os
import json
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "junction_results")
CRITICAL_DIR = os.path.join(HERE, "critical_sweep_results")

sys.path.insert(0, UTILS_PATH)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402
from junction_model import build_junction_network, solve_two_terminal, red_bonds  # noqa: E402
from run_critical_sweep_task import build_plan, R0, N_TRIALS, MAX_ATTEMPTS, DRAW_BUDGET  # noqa: E402

RHOS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
REPRO_TOL = 1e-9        # agreement demanded between rho = 0 and the stored R_eff


def measure(res, ref_R, a):
    net = build_junction_network(res)
    rows = []
    red_seg = red_jun = None
    for rho in RHOS:
        R_j = rho * R0 * a
        sol = solve_two_terminal(net, R0, R_j)
        if rho == 0.0 and abs(sol["R_eff"] - ref_R) > REPRO_TOL * ref_R:
            raise RuntimeError(f"rho = 0 does not reproduce the stored network: "
                               f"{sol['R_eff']!r} vs {ref_R!r}")
        rs, rj = red_bonds(sol)
        if rj is not None and red_seg is None:
            red_seg, red_jun = rs, rj
        rows.append(dict(
            rho=rho, R_eff=sol["R_eff"],
            red_seg_R=float(np.sum(sol["seg_R"][rs])),
            red_jun_R=0.0 if rj is None else float(np.sum(sol["jun_R"][rj])),
            jun_R_total_dissipated=0.0 if rj is None else float(np.sum(sol["jun_R"] * sol["I_jun"] ** 2) / sol["I_total"] ** 2),
        ))
    return dict(
        n_cross=int(net["n_cross"]), n_nodes=int(len(net["coords"])), n_seg=int(len(net["seg_u"])),
        n_red_seg=int(red_seg.sum()), n_red_jun=int(red_jun.sum()),
        red_jun_x=net["coords"][net["jun_u"][red_jun], 0].tolist(),
        by_rho=rows,
    )


def run_point(task_id, max_trials=None):
    plan = build_plan()
    if task_id >= len(plan):
        print(f"task_id {task_id} out of range (plan has {len(plan)} points)")
        return
    pt = plan[task_id]
    N, a = pt["N"], pt["a"]
    n_trials = N_TRIALS if max_trials is None else min(max_trials, N_TRIALS)
    with open(os.path.join(CRITICAL_DIR, f"point_{task_id:02d}.json")) as f:
        ref = json.load(f)
    if (ref["N"], ref["a"]) != (N, a):
        raise RuntimeError(f"stored point is (N={ref['N']}, a={ref['a']}), plan says ({N}, {a})")
    ref_R = ref["R_eff_values"]
    print(f"[task {task_id}] a={a}  N={N}  N a^2={pt['x']:.2f}; {n_trials} realizations, "
          f"{len(RHOS)} values of rho", flush=True)

    np.random.seed(stable_seed(N, round(a, 3), "critical"))
    nets, total_attempts, t_start = [], 0, time.time()
    while len(nets) < n_trials and total_attempts < DRAW_BUDGET:
        res, attempts = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS)
        total_attempts += attempts
        if res is None:
            continue
        k = len(nets)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            R_wire = float(distance_dependent_impedance(res, R0)["impedance"])
        if k < len(ref_R) and abs(R_wire - ref_R[k]) > REPRO_TOL * abs(ref_R[k]):
            raise RuntimeError(f"realization {k} does not reproduce the stored network: "
                               f"{R_wire!r} vs {ref_R[k]!r}")
        t0 = time.time()
        m = measure(res, R_wire, a)
        m["realization"] = k
        m["seconds_measure"] = time.time() - t0
        nets.append(m)
        R = {r["rho"]: r["R_eff"] for r in m["by_rho"]}
        print(f"  {k+1:2d}/{n_trials}  nodes={m['n_nodes']:5d}  R(0)={R[0.0]:9.2f}  R(1)={R[1.0]:9.2f}  "
              f"R(100)={R[100.0]:10.1f}  red seg/jun={m['n_red_seg']:3d}/{m['n_red_jun']:3d}  "
              f"{m['seconds_measure']:5.1f}s  {time.time()-t_start:6.0f}s", flush=True)

    out = dict(task_id=task_id, a=a, N=N, x=pt["x"], x_target=pt["x_target"], R0=R0, rhos=RHOS,
               n=len(nets), n_target=n_trials, total_draws=total_attempts,
               networks=nets, seconds=time.time() - t_start)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"point_{task_id:02d}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[task {task_id}] done: {len(nets)} realizations, mean red junctions "
          f"{np.mean([m['n_red_jun'] for m in nets]):.2f} vs red segments "
          f"{np.mean([m['n_red_seg'] for m in nets]):.2f}, {out['seconds']:.0f}s -> {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)
    if args[0] == "--plan":
        for i, pt in enumerate(build_plan()):
            print(f"{i:2d}  a={pt['a']:.2f}  N={pt['N']:5d}  N a^2={pt['x']:6.2f}")
        sys.exit(0)
    max_trials = int(args[args.index("--max-trials") + 1]) if "--max-trials" in args else None
    run_point(int(args[0]), max_trials=max_trials)
