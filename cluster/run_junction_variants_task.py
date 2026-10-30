"""
Distributed and bimodal junction resistances on the near-threshold sweep
(variants 2 and 3 of the junction study; variant 1, uniform R_j, is
run_junction_task.py).

Same 990 networks, regenerated from their seeds and checked against
critical_sweep_results/. On each:

  * variant 2, distributed: R_j lognormal with arithmetic mean rho_bar R0 a,
        R_j = rho_bar R0 a * exp(sigma z - sigma^2 / 2),  z ~ N(0, 1),
    for rho_bar in RHO_BAR and sigma in SIGMAS, N_DRAWS independent draws
    each; records R_eff(distributed) / R_eff(uniform at the same mean);
  * variant 3, bimodal (metallic / semiconducting sticks): each stick is
    metallic with probability P_METAL; a junction between sticks of the same
    kind costs rho_c R0 a, a mixed junction kappa times that, for kappa in
    KAPPAS, N_ASSIGN independent assignments each; records
    R_eff(kappa) / R_eff(kappa = 1) and the share of the power dissipated in
    the mixed junctions.

Junction draws and stick assignments use a generator seeded per
(point, realization), so they are reproducible and independent of the
network-generation stream.

Usage:
  python run_junction_variants_task.py --plan
  python run_junction_variants_task.py <task_id> [--max-trials 3]
"""
import sys
import os
import json
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "junction_variants_results")
CRITICAL_DIR = os.path.join(HERE, "critical_sweep_results")

sys.path.insert(0, UTILS_PATH)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402
from junction_model import build_junction_network, solve_two_terminal  # noqa: E402
from run_critical_sweep_task import build_plan, R0, N_TRIALS, MAX_ATTEMPTS, DRAW_BUDGET  # noqa: E402

REPRO_TOL = 1e-9
# variant 2
RHO_BAR = [0.1, 1.0, 10.0]
SIGMAS = [0.5, 1.0, 1.5]
N_DRAWS = 5
# variant 3
P_METAL = 1.0 / 3.0
RHO_C = 1.0
KAPPAS = [10.0, 100.0, 1000.0]
N_ASSIGN = 5


def measure(net, a, rng):
    R0a = R0 * a
    out = {"n_cross": int(net["n_cross"]), "distributed": [], "bimodal": []}

    for rb in RHO_BAR:
        Rbar = rb * R0a
        R_uni = solve_two_terminal(net, R0, Rbar)["R_eff"]
        for sg in SIGMAS:
            ratios = []
            for _ in range(N_DRAWS):
                Rj = Rbar * np.exp(sg * rng.standard_normal(net["n_cross"]) - sg ** 2 / 2)
                ratios.append(solve_two_terminal(net, R0, Rj)["R_eff"] / R_uni)
            out["distributed"].append(dict(rho_bar=rb, sigma=sg, ratio_mean=float(np.mean(ratios)),
                                           ratio_min=float(np.min(ratios)), ratio_max=float(np.max(ratios))))

    Rj = RHO_C * R0a
    base = solve_two_terminal(net, R0, Rj)["R_eff"]
    n_sticks = max(max(s) for s in net["sticks_of"].values()) + 1
    pairs = [net["sticks_of"][n] for n in net["crossings"]]
    for _ in range(N_ASSIGN):
        metallic = rng.random(n_sticks) < P_METAL
        mixed = np.array([metallic[i] != metallic[j] for (i, j) in pairs])
        for kappa in KAPPAS:
            Rj_arr = np.where(mixed, kappa * Rj, Rj)
            sol = solve_two_terminal(net, R0, Rj_arr)
            P_mixed = float(np.sum(sol["jun_R"][mixed] * sol["I_jun"][mixed] ** 2))
            out["bimodal"].append(dict(kappa=kappa, ratio=sol["R_eff"] / base,
                                       mixed_fraction=float(mixed.mean()),
                                       mixed_dissipation=P_mixed / (sol["R_eff"] * sol["I_total"] ** 2)))
    return out


def run_point(task_id, max_trials=None):
    plan = build_plan()
    if task_id >= len(plan):
        print(f"task_id {task_id} out of range"); return
    pt = plan[task_id]; N, a = pt["N"], pt["a"]
    n_trials = N_TRIALS if max_trials is None else min(max_trials, N_TRIALS)
    with open(os.path.join(CRITICAL_DIR, f"point_{task_id:02d}.json")) as f:
        ref = json.load(f)
    ref_R = ref["R_eff_values"]
    print(f"[task {task_id}] a={a}  N={N}  N a^2={pt['x']:.2f}; {n_trials} realizations; "
          f"{len(RHO_BAR)*len(SIGMAS)*N_DRAWS + len(KAPPAS)*N_ASSIGN + len(RHO_BAR) + 1} solves per network", flush=True)

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
            raise RuntimeError(f"realization {k} does not reproduce the stored network")
        t0 = time.time()
        rng = np.random.default_rng(stable_seed(N, round(a, 3), "junction-variants", k))
        m = measure(build_junction_network(res), a, rng)
        m["realization"] = k; m["R_wire"] = R_wire; m["seconds_measure"] = time.time() - t0
        nets.append(m)
        d = {(r["rho_bar"], r["sigma"]): r["ratio_mean"] for r in m["distributed"]}
        b = {}
        for r in m["bimodal"]:
            b.setdefault(r["kappa"], []).append(r["ratio"])
        print(f"  {k+1:2d}/{n_trials}  crossings={m['n_cross']:5d}  dist(rho=1,s=1)={d[(1.0,1.0)]:.3f}  "
              f"dist(rho=10,s=1.5)={d[(10.0,1.5)]:.3f}  bimodal kappa=100: {np.mean(b[100.0]):7.2f}  "
              f"{m['seconds_measure']:5.1f}s  {time.time()-t_start:6.0f}s", flush=True)

    out = dict(task_id=task_id, a=a, N=N, x=pt["x"], x_target=pt["x_target"], R0=R0,
               rho_bar=RHO_BAR, sigmas=SIGMAS, n_draws=N_DRAWS, p_metal=P_METAL, rho_c=RHO_C, kappas=KAPPAS, n_assign=N_ASSIGN,
               n=len(nets), n_target=n_trials, total_draws=total_attempts, networks=nets, seconds=time.time() - t_start)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"point_{task_id:02d}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[task {task_id}] done: {len(nets)} realizations, {out['seconds']:.0f}s -> {path}")


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
