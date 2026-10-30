"""
Run a whole grid in ONE SLURM job with a process pool, instead of a job array.

Apuana's `simple` QoS caps a user at 4 concurrent jobs and 16 CPUs, and the
task scripts are single-threaded Python, so a 20-task array uses about four
cores at a time. The `complex` QoS allows one job of up to 48 CPUs: this
driver takes that one job, maps the grid's combinations over a
multiprocessing pool of `--workers` processes, and writes the SAME chunk files
the array would have written (same N_TASKS split, same row contents), so the
analysis notebooks need no change. Every combination seeds itself from its
parameters inside run_one_combo, so the results do not depend on which
process ran it or in what order.

Usage:
  python run_grid_parallel.py --task scalar --workers 48
  python run_grid_parallel.py --task onion  --workers 48
  python run_grid_parallel.py --task critical --workers 33
  python run_grid_parallel.py --task preview --workers 48
"""
import os, sys, json, time, argparse, importlib
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
# one thread per worker: the pool provides the parallelism; BLAS threads would fight
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

TASKS = {
    "scalar":   dict(module="run_effective_resistance_d_task", combos="build_combos", worker="run_one_combo", n_tasks="N_TASKS", results="RESULTS_DIR"),
    "onion":    dict(module="run_onion_grid_task",             combos="build_combos", worker="run_one_combo", n_tasks="N_TASKS", results="RESULTS_DIR"),
}


def _combo_job(args):
    task, combo = args
    mod = importlib.import_module(TASKS[task]["module"])
    # smoke-test hook: GRID_PARALLEL_TRIALS=2 caps the realizations per combination
    if os.environ.get("GRID_PARALLEL_TRIALS"):
        mod.N_TRIALS = int(os.environ["GRID_PARALLEL_TRIALS"])
    t0 = time.time()
    row = getattr(mod, TASKS[task]["worker"])(*combo)
    return combo, row, time.time() - t0


def run_grid(task, workers):
    mod = importlib.import_module(TASKS[task]["module"])
    combos = getattr(mod, TASKS[task]["combos"])()
    n_tasks = getattr(mod, TASKS[task]["n_tasks"]); results_dir = getattr(mod, TASKS[task]["results"])
    # smoke-test hooks: GRID_PARALLEL_LIMIT=8 keeps the first 8 combinations,
    # GRID_PARALLEL_OUT redirects the chunk files
    if os.environ.get("GRID_PARALLEL_LIMIT"):
        combos = combos[: int(os.environ["GRID_PARALLEL_LIMIT"])]
    if os.environ.get("GRID_PARALLEL_OUT"):
        results_dir = os.environ["GRID_PARALLEL_OUT"]
    print(f"[{task}] {len(combos)} combinations, {workers} workers, writing {n_tasks} chunk files to {results_dir}", flush=True)
    rows_by_combo = {}
    t_start = time.time(); done = 0
    with Pool(workers) as pool:
        for combo, row, dt in pool.imap_unordered(_combo_job, [(task, c) for c in combos]):
            rows_by_combo[tuple(combo)] = row; done += 1
            mean = row.get("mean_ohm"); mean_str = f"{mean:.1f}" if mean is not None else "n/a"
            print(f"  [{done:3d}/{len(combos)}] N={combo[0]:3d} a={combo[1]:.2f} d={combo[2]:.2f}  n={row['n']:2d}  "
                  f"rate={row['perc_rate']:.1%}  mean={mean_str}  {dt:6.0f}s  (elapsed {time.time()-t_start:6.0f}s)", flush=True)
    os.makedirs(results_dir, exist_ok=True)
    chunk_size = -(-len(combos) // n_tasks)
    for task_id in range(n_tasks):
        my = combos[task_id * chunk_size: min((task_id + 1) * chunk_size, len(combos))]
        rows = [rows_by_combo[tuple(c)] for c in my]
        with open(os.path.join(results_dir, f"chunk_{task_id:02d}.json"), "w") as f:
            json.dump(rows, f, indent=2)
    print(f"[{task}] done in {time.time()-t_start:.0f}s; {n_tasks} chunk files written", flush=True)


def run_points(task, workers):
    """Point-per-task scripts (critical sweep, preview curves): each unit writes
    its own file, so the pool just calls the task's own entry point."""
    if task == "critical":
        import run_critical_sweep_task as m
        ids = list(range(len(m.build_plan()))); fn = m.run_point
    elif task == "preview":
        import run_preview_curves_task as m
        ids = list(range(m.N_TASKS)); fn = m.cmd_task
    else:
        raise SystemExit(f"unknown task {task}")
    print(f"[{task}] {len(ids)} units, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(fn, ids), 1):
            print(f"  [{i}/{len(ids)}] unit done (elapsed {time.time()-t0:.0f}s)", flush=True)
    print(f"[{task}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True, choices=["scalar", "onion", "critical", "preview"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    (run_grid if a.task in TASKS else run_points)(a.task, a.workers)
