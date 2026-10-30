"""
Large-film sweep (paper item d): terminal resistance and red bonds of 2D stick
films 5 to 80 stick lengths across, from N a^2 = 4 to 20, with the sparse
builder of sparse_network.py. Extends the near-threshold sweep of the thesis
(films 5, 6.7 and 10 stick lengths across, dense solver) to test

  * whether R / (R0 a) collapses onto one curve above N a^2 ~ 8 in large films;
  * the finite-size scaling R / (R0 a) ~ Lambda^(t/nu) near the threshold;
  * the half-percolation density and the width of the transition against Lambda;
  * how the number of red bonds and their share of R_eff scale with Lambda.

One array task per (a, N a^2) point; each writes large_film_results/point_NN.json.
Before measuring anything, every task checks that the sparse builder reproduces
stored networks of the thesis's near-threshold sweep (sparse_network.validate).

Usage:
  python run_large_film_task.py --parallel 16 # all points in one job, 16 at a time;
                                              # points already in large_film_results/ are skipped
  python run_large_film_task.py --plan        # print the points and exit
  python run_large_film_task.py --validate    # full validation, then exit
  python run_large_film_task.py <task_id>     # run one point
  python run_large_film_task.py <task_id> --max-trials 3   # quick local check

Seeds: stable_seed(N, a, 'large_film'), so any network can be regenerated.
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sparse_network as sn  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "large_film_results")
R0 = 1000.0
N_TRIALS = 50            # percolating realizations per point
DRAW_BUDGET = 3000       # total draws per point
MAX_FAILS_IN_A_ROW = 600 # stop a point far below its threshold early

# Films Lambda = 1/a stick lengths across. The first three repeat the thesis's
# sizes 5 and 10 for overlap (new seeds), the rest extend them.
SIZES = [0.2, 0.1, 0.05, 0.025, 0.0125]
X = [4, 5, 5.5, 6, 6.5, 7, 8, 9, 10, 12, 14, 16, 18, 20]


def build_plan():
    plan = []
    for a in SIZES:
        for x in X:
            N = int(round(x / a ** 2))
            plan.append({"a": a, "x_target": x, "N": N, "x": N * a ** 2, "Lambda": 1.0 / a})
    return plan


def run_point(task_id, max_trials=None, validate=True):
    plan = build_plan()
    if task_id >= len(plan):
        print(f"task_id {task_id} out of range (plan has {len(plan)} points)")
        return
    if validate:
        sn.validate(points=(0, 24), n_check=2, verbose=False)   # guards the environment

    pt = plan[task_id]; N, a = pt["N"], pt["a"]
    n_target = N_TRIALS if max_trials is None else min(max_trials, N_TRIALS)
    print(f"[task {task_id}] a={a} N={N} N a^2={pt['x']:.3f} Lambda={pt['Lambda']:.0f}", flush=True)
    np.random.seed(sn.stable_seed(N, round(a, 4), "large_film"))
    nets, draws, fails, t0 = [], 0, 0, time.time()
    while len(nets) < n_target and draws < DRAW_BUDGET and fails < MAX_FAILS_IN_A_ROW:
        P1, P2 = sn.draw_sticks(N, a); draws += 1
        net = sn.build_network(P1, P2, a, R0)
        if net is None:
            fails += 1
            continue
        fails = 0
        s = sn.solve(net)
        s.update(n_sticks=int(len(P1)), n_crossings=int(net["n_crossings"]), draw=draws)
        nets.append(s)
        if len(nets) % 10 == 0:
            print(f"  {len(nets)}/{n_target} after {draws} draws, {time.time() - t0:.0f}s", flush=True)
    out = dict(task_id=task_id, **pt, R0=R0, n=len(nets), n_target=n_target, draws=draws,
               perc_rate=len(nets) / draws if draws else 0.0,
               stopped=("done" if len(nets) >= n_target else "budget" if draws >= DRAW_BUDGET else "fails_in_a_row"),
               R_eff_values=[m["R_eff"] for m in nets], networks=nets, seconds=time.time() - t0)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"point_{task_id:02d}.json")
    json.dump(out, open(path, "w"), indent=1)
    ys = np.array(out["R_eff_values"]) / (R0 * a)
    print(f"[task {task_id}] {len(nets)} networks from {draws} draws ({out['stopped']}); "
          f"mean R/(R0 a) = {ys.mean() if len(ys) else float('nan'):.3f}; "
          f"mean red bonds {np.mean([m['n_red'] for m in nets]) if nets else float('nan'):.1f}; "
          f"{out['seconds']:.0f}s -> {path}", flush=True)


def _job(task_id):
    run_point(task_id, validate=False)
    return task_id


def run_all(nproc):
    """Every point not yet in RESULTS_DIR, nproc at a time, the most expensive first."""
    sn.validate(points=(0, 24), n_check=2, verbose=False)
    plan = build_plan()
    done = {i for i in range(len(plan)) if os.path.exists(os.path.join(RESULTS_DIR, f"point_{i:02d}.json"))}
    todo = sorted((i for i in range(len(plan)) if i not in done), key=lambda i: -plan[i]["N"])
    print(f"{len(done)} points already done; {len(todo)} to run on {nproc} workers", flush=True)
    from multiprocessing import Pool
    with Pool(nproc, maxtasksperchild=1) as pool:
        for k, tid in enumerate(pool.imap_unordered(_job, todo), 1):
            print(f"== finished point {tid} ({k}/{len(todo)})", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--parallel" in args:
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[_v] = "1"
        run_all(int(args[args.index("--parallel") + 1])); sys.exit(0)
    if not args:
        print(__doc__); sys.exit(2)
    if args[0] == "--plan":
        for i, pt in enumerate(build_plan()):
            print(f"{i:2d}  a={pt['a']:<7} Lambda={pt['Lambda']:>4.0f}  N={pt['N']:7d}  N a^2={pt['x']:6.3f}")
        sys.exit(0)
    if args[0] == "--validate":
        sn.validate(points=(0, 5, 12, 16, 24, 30), n_check=10); sys.exit(0)
    mt = int(args[args.index("--max-trials") + 1]) if "--max-trials" in args else None
    run_point(int(args[0]), mt)
