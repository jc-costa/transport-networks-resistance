"""
Standalone script backing the SLURM array (parallel) run of the R_eff,ij
onion-layer curve grid over (N, a, d): the MSc-thesis inversion study asking
whether the full R_eff,ij(r) curve shape -- not just the scalar source-drain
R_eff -- helps disambiguate (N, a, d), since the scalar R_eff is known to
collapse onto ~N*a^2 (see notebooks/inverse_N_a_from_R_eff.ipynb) and never
even used d.

The grid construction (A_VALUES, D_VALUES, N_PER_A, N_FLOOR, NODE_CAP,
CONTACT_DISTANCE, N_TRIALS, MAX_ATTEMPTS, R0, N_TASKS, n_max_for_a_d,
build_combos) is copied VERBATIM from run_effective_resistance_d_task.py, and
each combo's percolating-network draw loop uses the identical seed_base
formula and calls generate_percolating_network_3d in the same order -- so
this script draws the *same* accepted networks as the existing scalar-R_eff
grid (effective_resistance_vs_d.ipynb Part C / effective_resistance_d_results/),
making the onion curves here a paired add-on to that dataset rather than an
independent sample. Do not let the two constant blocks drift apart.

Per accepted network, this script records BOTH the scalar whole-face R_eff
(recomputed here rather than re-loaded, so this script doesn't depend on the
other grid's cache existing) AND the mean R_eff,ij(r) onion curve from a
fixed source-boundary reference node, via multimeter.run_onion_protocol_pinv
(Laplacian-pseudoinverse fast path -- see utils/laplacian_utils.py's
effective_resistance_from_reference -- validated against the solve-based
run_onion_protocol in notebooks/multimeter_r_eff.ipynb Section 6b).

Usage: python run_onion_grid_task.py <task_id>
  (or reads $SLURM_ARRAY_TASK_ID if no argument given)

Results land in cluster/onion_grid_results/chunk_<NN>.json --
notebooks/r_eff_ij_vs_N_a_d.ipynb picks these up automatically instead of
recomputing, once a complete set of N_TASKS chunk files exists (same pattern
as effective_resistance_vs_d.ipynb's Part C cache check).
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "onion_grid_results")

sys.path.insert(0, UTILS_PATH)
import numpy as np  # noqa: E402
from pipeline_utils_3d import generate_percolating_network_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402
from multimeter import run_onion_protocol_pinv  # noqa: E402

# Must match effective_resistance_vs_d.ipynb's Part C (LOCAL_PREVIEW=False
# branch) / run_effective_resistance_d_task.py exactly -- see module
# docstring: this is what makes the two grids' networks identical.
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

# Onion-protocol layer distances: default of onion_layers() (0.1..0.9 step
# 0.1), spelled out here so it's part of this script's fixed contract rather
# than an implicit default that could silently change upstream.
LAYER_DISTANCES = [round(0.1 * n, 2) for n in range(1, 10)]


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
    # stable_seed(), not hash() -- see utils/pipeline_utils.py's docstring:
    # hash() of a string-tagged tuple is randomized per-process, which
    # silently broke reproducibility for this exact 'ext'-tagged formula
    # everywhere else it was used (run_effective_resistance_d_task.py,
    # run_validation_task.py, effective_resistance_vs_d.ipynb).
    seed_base = stable_seed(N, round(a, 3), round(d, 3), 'ext')
    np.random.seed(seed_base)

    R_effs = []
    # Per-network per-layer mean R_eff,ij (list of dicts: {layer_distance: mean_r_eff}),
    # one entry per accepted percolating network -- the "several mean curves" building
    # block that gets aggregated into one mean-of-means curve for this (N, a, d) below.
    per_network_layer_means = []
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

        reference_node = res['pre_merge_source_nodes'][0]
        records = run_onion_protocol_pinv(
            res['pre_merge_intersections'], res['pre_merge_segment_to_nodes'],
            reference_node, layer_distances=LAYER_DISTANCES, R0=R0
        )
        layer_means = {}
        for layer_d in LAYER_DISTANCES:
            finite_vals = [r['r_eff'] for r in records
                          if r['layer_distance'] == layer_d and np.isfinite(r['r_eff'])]
            if finite_vals:
                layer_means[layer_d] = float(np.mean(finite_vals))
        per_network_layer_means.append(layer_means)

    rate = len(R_effs) / total_attempts if total_attempts else 0.0

    onion_layer_curves = {}
    for layer_d in LAYER_DISTANCES:
        per_realization = [lm[layer_d] for lm in per_network_layer_means if layer_d in lm]
        if per_realization:
            onion_layer_curves[str(layer_d)] = {
                "per_realization_means": per_realization,
                "mean": float(np.mean(per_realization)),
                "std": float(np.std(per_realization)),
                "n_realizations": len(per_realization),
            }

    return {
        "N": N, "a": a, "d": d, "n": len(R_effs),
        "mean_ohm": float(np.mean(R_effs)) if R_effs else None,
        "median_ohm": float(np.median(R_effs)) if R_effs else None,
        "std_ohm": float(np.std(R_effs)) if R_effs else None,
        "perc_rate": rate,
        "R_eff_values": R_effs,
        "reference_node_convention": "pre_merge_source_nodes[0]",
        "layer_distances": LAYER_DISTANCES,
        "onion_layer_curves": onion_layer_curves,
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
        n_layers = len(row['onion_layer_curves'])
        print(f"  N={N:3d} a={a:.1f} d={d:.2f}  {row['n']}/{N_TRIALS} percolating  "
              f"mean={mean_str}  perc_rate={row['perc_rate']:.1%}  "
              f"onion layers covered={n_layers}/{len(LAYER_DISTANCES)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[task {task_id}] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
