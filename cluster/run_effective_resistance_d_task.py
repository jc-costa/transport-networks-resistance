"""
Standalone script backing the SLURM array (parallel) version of
effective_resistance_vs_d.ipynb's Part C extended (N, a, d) grid. A notebook
can't be sliced across array tasks, so this script duplicates just the
*driving* logic (the grid construction and which combos go to which task) --
the actual per-combo pipeline (generate_percolating_network_3d,
distance_dependent_impedance) is imported unchanged from utils/, not
duplicated. Mirrors run_random_networks_task.py's structure exactly.

The grid construction (n_max_for_a_d, a_values_ext, d_values_ext, n_per_a,
N_floor_ext, node_cap, n_trials_ext) below MUST match
effective_resistance_vs_d.ipynb's Part C "full grid" branch
(LOCAL_PREVIEW=False) exactly, or this script's combo list -- and therefore
its task_id -> combo-slice assignment -- would silently diverge from what
the notebook itself would generate. Both are deterministic functions of
these fixed constants, so every task independently recomputes the identical
420-combo list and just slices into it; no separate "prepare" step needed.

Usage: python run_effective_resistance_d_task.py <task_id>
  (or reads $SLURM_ARRAY_TASK_ID if no argument given)

Results land in cluster/effective_resistance_d_results/chunk_<NN>.json --
effective_resistance_vs_d.ipynb's Part C can pick these up automatically
instead of recomputing, once a complete set of N_TASKS chunk files exists
(same pattern as test_150_random_networks.ipynb's Section 4 cache check).
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "effective_resistance_d_results")

sys.path.insert(0, UTILS_PATH)
import numpy as np  # noqa: E402
from pipeline_utils_3d import generate_percolating_network_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402

# Must match effective_resistance_vs_d.ipynb's Part C (LOCAL_PREVIEW=False
# branch) exactly -- see module docstring.
A_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
D_VALUES = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
N_PER_A = 6
N_FLOOR = 20
NODE_CAP = 1500
CONTACT_DISTANCE = 0.03
N_TRIALS = 6
MAX_ATTEMPTS = 500
R0 = 1000.0
N_TASKS = 20


def n_max_for_a_d(a, d, node_cap=NODE_CAP, N_ceiling=150, N_floor=N_FLOOR, round_to=5):
    C, pN, pa, pd = 0.0016, 2.66, 2.05, -0.60
    raw = (node_cap / (C * a ** pa * d ** pd)) ** (1 / pN)
    capped = min(N_ceiling, raw)
    return int(max(N_floor, round(capped / round_to) * round_to))


def build_combos():
    grid_ext = {}
    for a in A_VALUES:
        n_max = n_max_for_a_d(a, min(D_VALUES))
        Ns = sorted(set(int(round(x / 5) * 5) for x in np.linspace(N_FLOOR, n_max, N_PER_A)))
        grid_ext[a] = Ns
    return [(N, a, d) for a in A_VALUES for N in grid_ext[a] for d in D_VALUES]


def run_one_combo(N, a, d):
    # stable_seed(), not hash() -- hash() of a string-tagged tuple is
    # randomized per-process (PYTHONHASHSEED), which silently made every
    # run of this script (including each of the 20 parallel SLURM array
    # tasks) draw networks from an unreproducible, effectively random
    # seed_base, despite the code looking deterministic. See
    # utils/pipeline_utils.py's stable_seed() docstring.
    seed_base = stable_seed(N, round(a, 3), round(d, 3), 'ext')
    np.random.seed(seed_base)

    R_effs = []
    total_attempts = 0
    while len(R_effs) < N_TRIALS:
        res, attempts = generate_percolating_network_3d(
            N=N, a=a, contact_distance=CONTACT_DISTANCE,
            box_dims=(1.0, 1.0, d), k_uniform=R0, max_attempts=MAX_ATTEMPTS
        )
        total_attempts += attempts
        if res is None:
            break
        rl = distance_dependent_impedance(res, R0)
        R_effs.append(rl['impedance'])

    rate = len(R_effs) / total_attempts if total_attempts else 0.0
    return {
        "N": N, "a": a, "d": d, "n": len(R_effs),
        "mean_ohm": float(np.mean(R_effs)) if R_effs else None,
        "median_ohm": float(np.median(R_effs)) if R_effs else None,
        "std_ohm": float(np.std(R_effs)) if R_effs else None,
        "perc_rate": rate,
        "R_eff_values": R_effs,
    }


def main():
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])

    combos = build_combos()
    chunk_size = -(-len(combos) // N_TASKS)  # ceil division
    start = task_id * chunk_size
    end = min(start + chunk_size, len(combos))
    my_combos = combos[start:end]

    if not my_combos:
        print(f"[task {task_id}] no combos assigned (start={start} >= total={len(combos)})")
        return

    print(f"[task {task_id}] running combos {start}..{end - 1} ({len(my_combos)} combos)")

    rows = []
    for (N, a, d) in my_combos:
        row = run_one_combo(N, a, d)
        rows.append(row)
        mean_str = f"{row['mean_ohm']:.1f}" if row['mean_ohm'] is not None else "n/a"
        print(f"  N={N:3d} a={a:.1f} d={d:.2f}  {row['n']}/{N_TRIALS} percolating  "
              f"mean={mean_str}  perc_rate={row['perc_rate']:.1%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[task {task_id}] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
