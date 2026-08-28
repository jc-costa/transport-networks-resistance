"""
Standalone script backing the SLURM array version of test_150_random_networks.ipynb's
Section 4 (the 1500-case batch run). A notebook can't be sliced across array
tasks, so this script duplicates just the *driving* logic (which cases go to
which task) -- the actual per-case pipeline (run_full_network_case) is
imported unchanged from utils/random_network_batch.py, not duplicated.

No separate "prepare" step needed (unlike inverse_N_a_from_R_eff.ipynb's array
version): generate_diversified_configs is a pure, deterministic function of
fixed parameters (param_seed=2025, case_seed_base=10_000, same ranges as the
notebook) -- every task independently recomputes the identical 1500-config
list and just slices into it, so there's no risk of drift between tasks.

Usage: python run_random_networks_task.py <task_id>
  (or reads $SLURM_ARRAY_TASK_ID if no argument given)

Results land in cluster/random_networks_results/chunk_<NN>.json --
test_150_random_networks.ipynb's Section 4 cell picks these up automatically
instead of recomputing, if a complete set of N_TASKS chunk files is present
(see that cell's source). Only the flat per-case summary rows are cached, not
the raw `example` data Section 8's analyze_case() demo uses for plotting --
that section falls back to "no data" for cache-loaded results, which doesn't
affect the Section 5/6 statistical claims (those only need the summary rows).
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "random_networks_results")

sys.path.insert(0, UTILS_PATH)
from random_network_batch import generate_diversified_configs, run_full_network_case  # noqa: E402

# Must match test_150_random_networks.ipynb's Section 2 cell exactly, or this
# script's (N, a) assignment per case_id would silently diverge from what the
# notebook itself would generate.
N_CASES = 1500
N_MIN, N_MAX = 8, 90
A_MIN, A_MAX = 0.15, 0.80
K_OHM = 1000.0
N_TASKS = 20


def main():
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])

    configs = generate_diversified_configs(
        N_CASES, N_MIN, N_MAX, A_MIN, A_MAX, param_seed=2025, case_seed_base=10_000
    )

    chunk_size = -(-N_CASES // N_TASKS)  # ceil division
    start = task_id * chunk_size
    end = min(start + chunk_size, N_CASES)
    my_configs = configs[start:end]

    if not my_configs:
        print(f"[task {task_id}] no configs assigned (start={start} >= N_CASES={N_CASES})")
        return

    print(f"[task {task_id}] running cases {start}..{end - 1} ({len(my_configs)} cases)")

    rows = []
    for cfg in my_configs:
        row, _ = run_full_network_case(cfg, k=K_OHM, plot=False, verbose=False)
        rows.append(row)
        singular_label = "N/A" if row["is_singular"] is None else ("YES" if row["is_singular"] else "no")
        print(f"  case {row['case_id']:4d}: N={row['N']:3d} a={row['a']:.3f}  "
              f"percolates={str(row['has_percolation']):5s}  singular={singular_label:3s}  [{row['status']}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[task {task_id}] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
