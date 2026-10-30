"""
Standalone script for boosting statistical power on the onion-curve degeneracy finding.

cluster/run_onion_grid_task.py's full-grid run (N_TRIALS=6) found real but partial evidence
(notebooks/r_eff_ij_vs_N_a_d.ipynb Part C: one-way ANOVA per (group, layer) on matched-scalar-
R_eff (N, a, d) triples) that onion-curve SHAPE differs between triples the scalar R_eff can't
tell apart -- 18% of 214 tests significant at p<0.05 (vs ~5% expected by chance), 2 surviving
Bonferroni correction. But most matched groups (REL_TOL=0.15 on scalar R_eff) don't reach
significance yet, most likely because N_TRIALS=6 leaves too much per-triple noise to resolve a
real-but-modest shape effect in the weaker groups. Rather than rerunning the entire 420-combo
grid at higher N_TRIALS (~355 combos land in SOME matched group -- almost the whole grid, since
R_eff degeneracy is pervasive), this reruns only the triples in the TOP_N_GROUPS strongest-
signal groups (lowest ANOVA min-p from the existing run) at a much higher N_TRIALS_BOOST, to turn
the clearest evidence into something thesis-defensible instead of a noisy fraction of tests.

Same seed formula (stable_seed) and same per-combo params (R0, CONTACT_DISTANCE, MAX_ATTEMPTS,
LAYER_DISTANCES) as run_onion_grid_task.py, so the FIRST N_TRIALS_ORIGINAL realizations drawn
here are byte-identical to what's already in cluster/onion_grid_results/ -- np.random.seed(seed_base)
followed by the same generate_percolating_network_3d calls in the same order reproduces the same
draw sequence regardless of how many more draws follow. This is a strict extension of the
existing data, not an independent resample.

Two modes:
  --prepare   Recompute the matched-R_eff groups + per-group ANOVA (same logic as
              r_eff_ij_vs_N_a_d.ipynb Part C) from the CURRENT cluster/onion_grid_results/ +
              cluster/effective_resistance_d_results/ data, pick the TOP_N_GROUPS groups by
              ANOVA min-p, and save the combo list to cluster/onion_grid_boost_plan.json. Run
              this ONCE, before submitting the array job -- every task reads the same frozen
              plan, so results stay aligned by index regardless of execution order.
  <task_id>   Run this task's slice of onion_grid_boost_plan.json["combos"] at
              N_TRIALS_BOOST, save to cluster/onion_grid_boost_results/chunk_<NN>.json.
"""
import sys
import os
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
ONION_DIR = os.path.join(HERE, "onion_grid_results")
SCALAR_DIR = os.path.join(HERE, "effective_resistance_d_results")
PLAN_PATH = os.path.join(HERE, "onion_grid_boost_plan.json")
RESULTS_DIR = os.path.join(HERE, "onion_grid_boost_results")

sys.path.insert(0, UTILS_PATH)
import numpy as np  # noqa: E402
from pipeline_utils_3d import generate_percolating_network_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402
from multimeter import run_onion_protocol_pinv  # noqa: E402

# Must match run_onion_grid_task.py exactly (same per-combo model/protocol).
R0 = 1000.0
CONTACT_DISTANCE = 0.03
MAX_ATTEMPTS = 500
N_TASKS = 20
LAYER_DISTANCES = [round(0.1 * n, 2) for n in range(1, 10)]

N_TRIALS_ORIGINAL = 24  # what cluster/onion_grid_results/ is computed at (6 until 2026-09-15)
N_TRIALS_BOOST = 50     # target for this boosted rerun
REL_TOL = 0.15          # same matched-scalar-R_eff grouping tolerance as the notebook
TOP_N_GROUPS = 5        # how many strongest-signal groups to boost


def load_chunks(results_dir, n_tasks=N_TASKS):
    chunk_files = [os.path.join(results_dir, f"chunk_{i:02d}.json") for i in range(n_tasks)]
    rows = []
    for p in chunk_files:
        if os.path.exists(p):
            with open(p) as f:
                rows.extend(json.load(f))
    return rows


def build_matched_groups(onion_by_key, scalar_by_key):
    """Same grouping logic as r_eff_ij_vs_N_a_d.ipynb Part C (partCgroups cell) -- must stay
    identical or the selected groups here would diverge from what the notebook itself finds."""
    paired_keys = [k for k in onion_by_key if k in scalar_by_key and onion_by_key[k]['onion_layer_curves']
                  and scalar_by_key[k]['mean_ohm'] is not None]
    paired_keys.sort(key=lambda k: scalar_by_key[k]['mean_ohm'])

    groups = []
    used = set()
    for i, k1 in enumerate(paired_keys):
        if k1 in used:
            continue
        r1 = scalar_by_key[k1]['mean_ohm']
        group = [k1]
        for k2 in paired_keys[i + 1:]:
            if k2 in used:
                continue
            r2 = scalar_by_key[k2]['mean_ohm']
            if abs(r2 - r1) / r1 <= REL_TOL and k2[:2] != k1[:2]:
                group.append(k2)
        if len(group) >= 2:
            groups.append(group)
            used.update(group)
    return groups


def per_realization_normalized(onion_by_key, key, shared_layers):
    """Same as r_eff_ij_vs_N_a_d.ipynb Part C.2's helper: individual (not averaged) realization
    curves for `key`, restricted to shared_layers, each normalized by its own value at the
    largest shared layer."""
    curve = onion_by_key[key]['onion_layer_curves']
    n_real = min(curve[str(d)]['n_realizations'] for d in shared_layers if str(d) in curve)
    out = []
    for i in range(n_real):
        vals = [curve[str(d)]['per_realization_means'][i] for d in shared_layers]
        out.append(vals)
    out = np.array(out)
    return out / out[:, -1:]


def rank_groups_by_anova(groups, onion_by_key):
    """Same ANOVA logic as the notebook's corrected Part C.2, returns groups sorted by min p.
    scipy is only needed here (--prepare, run once, typically locally) -- imported locally so
    the array task processes (cmd_run_task) don't need scipy in their environment at all."""
    from scipy import stats
    results = []
    for gi, group in enumerate(groups):
        curve_ds = {key: sorted(float(d) for d in onion_by_key[key]['onion_layer_curves']) for key in group}
        shared = sorted(set.intersection(*[set(ds) for ds in curve_ds.values()]))
        if len(shared) < 2:
            continue

        per_key_curves = {key: per_realization_normalized(onion_by_key, key, shared) for key in group}
        per_key_curves = {k: v for k, v in per_key_curves.items() if v.shape[0] >= 3}
        if len(per_key_curves) < 2:
            continue

        layer_pvals = []
        for li, d in enumerate(shared):
            if li == len(shared) - 1:
                continue
            samples = [curves[:, li] for curves in per_key_curves.values()]
            _, p_val = stats.f_oneway(*samples)
            layer_pvals.append(p_val)

        if layer_pvals:
            results.append({'group_index': gi, 'triples': group, 'min_p': float(min(layer_pvals)),
                            'frac_sig_p05': float(np.mean([p < 0.05 for p in layer_pvals]))})

    results.sort(key=lambda g: g['min_p'])
    return results


def cmd_prepare():
    onion_rows = load_chunks(ONION_DIR)
    scalar_rows = load_chunks(SCALAR_DIR)
    onion_by_key = {(r['N'], r['a'], r['d']): r for r in onion_rows}
    scalar_by_key = {(r['N'], r['a'], r['d']): r for r in scalar_rows}
    print(f"Loaded {len(onion_by_key)} onion combos, {len(scalar_by_key)} scalar combos")

    groups = build_matched_groups(onion_by_key, scalar_by_key)
    ranked = rank_groups_by_anova(groups, onion_by_key)
    print(f"{len(groups)} matched-R_eff groups, {len(ranked)} with a valid ANOVA test")

    top = ranked[:TOP_N_GROUPS]
    combos = sorted(set(tuple(t) for g in top for t in g['triples']))
    print(f"\nTop {TOP_N_GROUPS} groups selected for boosting:")
    for g in top:
        print(f"  group {g['group_index']:2d}: {len(g['triples'])} triples, min_p={g['min_p']:.2e}, "
              f"frac layers p<0.05 = {g['frac_sig_p05']:.0%}")
    print(f"\n{len(combos)} unique (N, a, d) combos across the top {TOP_N_GROUPS} groups")

    plan = {
        "config": {"REL_TOL": REL_TOL, "TOP_N_GROUPS": TOP_N_GROUPS,
                  "N_TRIALS_ORIGINAL": N_TRIALS_ORIGINAL, "N_TRIALS_BOOST": N_TRIALS_BOOST},
        "selected_groups": [{"group_index": g["group_index"], "min_p": g["min_p"],
                             "frac_sig_p05": g["frac_sig_p05"],
                             "triples": [list(t) for t in g["triples"]]} for g in top],
        "combos": [list(c) for c in combos],
    }
    with open(PLAN_PATH, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nWrote {len(combos)}-combo boost plan to {PLAN_PATH}")


def run_one_combo(N, a, d, n_trials):
    """Same logic as run_onion_grid_task.py's run_one_combo, just parameterized by n_trials
    instead of the hardcoded N_TRIALS=6 -- same seed_base means the first N_TRIALS_ORIGINAL
    draws here are identical to what's already in cluster/onion_grid_results/."""
    seed_base = stable_seed(N, round(a, 3), round(d, 3), 'ext')
    np.random.seed(seed_base)

    R_effs = []
    per_network_layer_means = []
    total_attempts = 0

    while len(R_effs) < n_trials:
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

    # Aligned record, one entry per accepted realization in the same order as
    # R_eff_values, with None where a shell was empty. The per-layer lists
    # below skip empty shells, so across layers they are NOT aligned by index
    # whenever any realization lacks a shell; any per-realization analysis
    # (shape normalization, ANOVA, regression) must read this field instead.
    per_realization_curves = [
        {str(layer_d): lm.get(layer_d) for layer_d in LAYER_DISTANCES}
        for lm in per_network_layer_means
    ]

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
        "per_realization_curves": per_realization_curves, "n_trials_requested": n_trials,
        "mean_ohm": float(np.mean(R_effs)) if R_effs else None,
        "median_ohm": float(np.median(R_effs)) if R_effs else None,
        "std_ohm": float(np.std(R_effs)) if R_effs else None,
        "perc_rate": rate,
        "R_eff_values": R_effs,
        "reference_node_convention": "pre_merge_source_nodes[0]",
        "layer_distances": LAYER_DISTANCES,
        "onion_layer_curves": onion_layer_curves,
    }


def cmd_run_task(task_id):
    with open(PLAN_PATH) as f:
        plan = json.load(f)
    combos = [tuple(c) for c in plan["combos"]]
    n_trials = plan["config"]["N_TRIALS_BOOST"]

    chunk_size = -(-len(combos) // N_TASKS)
    start = task_id * chunk_size
    end = min(start + chunk_size, len(combos))
    my_combos = combos[start:end]

    if not my_combos:
        print(f"[task {task_id}] no combos assigned (start={start} >= total={len(combos)})")
        return

    print(f"[task {task_id}] running combos {start}..{end - 1} ({len(my_combos)} combos) at N_TRIALS={n_trials}")

    rows = []
    for (N, a, d) in my_combos:
        row = run_one_combo(N, a, d, n_trials)
        rows.append(row)
        mean_str = f"{row['mean_ohm']:.1f}" if row['mean_ohm'] is not None else "n/a"
        n_layers = len(row['onion_layer_curves'])
        print(f"  N={N:3d} a={a:.1f} d={d:.2f}  {row['n']}/{n_trials} percolating  "
              f"mean={mean_str}  perc_rate={row['perc_rate']:.1%}  "
              f"onion layers covered={n_layers}/{len(LAYER_DISTANCES)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[task {task_id}] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--prepare":
        cmd_prepare()
    else:
        tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
        cmd_run_task(tid)
