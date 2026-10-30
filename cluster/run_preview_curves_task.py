"""
Standalone script backing the SLURM array run of the Part A preview onion
curves (notebooks/r_eff_ij_vs_N_a_d.ipynb Part A, Fig. "onion-preview" in
chapters/resultados/resultados.tex).

Why this exists as a cluster job at all: the preview figure originally shipped
at N_TRIALS_PREVIEW=4, and because the draw loop there stops at the first
500-attempt block that fails to percolate, two of its three combos landed a
single realization each. A three-curve figure with no error bars, two of whose
curves are one network, cannot support the parameter comparisons the thesis
draws from it. This script recomputes all three combos at N_TRIALS=50.

The bottleneck is (45, 0.2, 0.10), which percolates on roughly 1 draw in 3000
(measured locally: 13 successes in 40339 draws). Fifty realizations of it is
about an hour of single-core work, which is why the realizations are spread
across an array here.

SEEDING DIFFERS from run_onion_grid_task.py, deliberately. That script seeds
once per combo and draws N_TRIALS networks from one stream, which cannot be
split across tasks. Here each (combo, realization) pair is its own work unit
with its own seed, stable_seed(N, a, d, 'ext', r), so any task can compute any
subset and the pooled result is identical regardless of how the array is
partitioned. The consequence is that these realizations are NOT the same
networks the paired scalar/onion grids drew, so this output feeds the preview
figure only and must not be pooled into the grid datasets of
cluster/onion_grid_results/ or cluster/effective_resistance_d_results/.

Usage:
  python run_preview_curves_task.py <task_id>   run this task's units
  python run_preview_curves_task.py --verify    check a synced-back set of chunks
                                                is complete and every unit percolated
  python run_preview_curves_task.py --aggregate rebuild figure + summary JSON
                                                from a complete set of chunks

Results land in cluster/preview_curves_results/chunk_<NN>.json. Existing chunk
files are never overwritten without --force: this directory is synced back from
Apuana by hand and a stray local run would otherwise destroy real results.
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "preview_curves_results")

sys.path.insert(0, UTILS_PATH)
import numpy as np  # noqa: E402
from pipeline_utils_3d import generate_percolating_network_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402
from multimeter import run_onion_protocol_pinv  # noqa: E402

# Match run_onion_grid_task.py's physical constants exactly. Only N_TRIALS and
# the per-draw attempt budget differ, for the reasons in the docstring.
CONTACT_DISTANCE = 0.03
R0 = 1000.0
LAYER_DISTANCES = [round(0.1 * n, 2) for n in range(1, 10)]

# (70, 0.2, 0.1), (70, 0.2, 0.15), (45, 0.2, 0.1) until 2026-09-15: under the clipping
# boundary rule those no longer percolate, so the same two contrasts (d at fixed N,
# N at fixed d) are taken at higher N, where the measured clipped-rule rates are 22.0%, 8.7% and 4.8%.
PREVIEW_COMBOS = [(150, 0.2, 0.1), (150, 0.2, 0.15), (125, 0.2, 0.1)]
N_TRIALS = 50
N_TASKS = 25

# One call, not a retry loop: at the worst combo's rate (~3.2e-4) a 100000-draw
# budget fails with probability e^-32, so a unit that comes back empty is a bug
# or a genuinely non-percolating combo, not bad luck worth retrying.
MAX_ATTEMPTS = 100000


def build_units():
    """(combo_index, realization_index) work units, in a fixed deterministic order."""
    return [(ci, r) for ci in range(len(PREVIEW_COMBOS)) for r in range(N_TRIALS)]


def run_one_unit(ci, r):
    N, a, d = PREVIEW_COMBOS[ci]
    np.random.seed(stable_seed(N, round(a, 3), round(d, 3), 'ext', r))

    res, attempts = generate_percolating_network_3d(
        N=N, a=a, contact_distance=CONTACT_DISTANCE,
        box_dims=(1.0, 1.0, d), k_uniform=R0, max_attempts=MAX_ATTEMPTS
    )
    if res is None:
        return {"combo_index": ci, "realization": r, "N": N, "a": a, "d": d,
                "percolated": False, "attempts": attempts,
                "R_eff": None, "layer_means": {}}

    R_eff = float(distance_dependent_impedance(res, R0)['impedance'])
    reference_node = res['pre_merge_source_nodes'][0]
    records = run_onion_protocol_pinv(
        res['pre_merge_intersections'], res['pre_merge_segment_to_nodes'],
        reference_node, layer_distances=LAYER_DISTANCES, R0=R0
    )
    layer_means = {}
    for layer_d in LAYER_DISTANCES:
        finite_vals = [x['r_eff'] for x in records
                       if x['layer_distance'] == layer_d and np.isfinite(x['r_eff'])]
        if finite_vals:
            layer_means[str(layer_d)] = float(np.mean(finite_vals))

    return {"combo_index": ci, "realization": r, "N": N, "a": a, "d": d,
            "percolated": True, "attempts": attempts,
            "R_eff": R_eff, "layer_means": layer_means}


def cmd_task(task_id, force=False):
    units = build_units()
    # Round-robin, not contiguous slices: the (45, 0.2, 0.10) units cost ~100x
    # the others, so a contiguous split would leave one task running for an hour
    # while the rest sat idle.
    mine = [u for i, u in enumerate(units) if i % N_TASKS == task_id]
    if not mine:
        print(f"[task {task_id}] no units assigned")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json")
    if os.path.exists(out_path) and not force:
        print(f"[task {task_id}] {out_path} already exists, refusing to overwrite "
              f"(pass --force if you really mean to)")
        sys.exit(1)

    print(f"[task {task_id}] {len(mine)} units at N_TRIALS={N_TRIALS}", flush=True)
    rows = []
    for (ci, r) in mine:
        row = run_one_unit(ci, r)
        rows.append(row)
        N, a, d = PREVIEW_COMBOS[ci]
        status = f"{row['R_eff']:.1f} Ohm" if row['percolated'] else "NO PERCOLATION"
        print(f"  N={N:3d} a={a} d={d:.2f} r={r:2d}  {row['attempts']:6d} draws  {status}",
              flush=True)

    with open(out_path, "w") as f:
        json.dump({"task_id": task_id, "n_tasks": N_TASKS,
                   "combos": [list(c) for c in PREVIEW_COMBOS],
                   "n_trials": N_TRIALS, "layer_distances": LAYER_DISTANCES,
                   "results": rows}, f, indent=1)
    print(f"[task {task_id}] wrote {out_path} ({len(rows)} units)")


def cmd_verify():
    """Answer 'did the array run correctly' from the chunk files alone.

    Checks completeness (every planned unit present exactly once), success (every
    unit percolated), and consistency (no chunk claims a different plan), then
    prints the per-combination realization counts the figure will be built from.
    Exits non-zero if anything is wrong, so it can gate the aggregate step.
    """
    import glob
    from collections import Counter

    problems = []
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "chunk_*.json")))
    print(f"Chunk files: {len(paths)}/{N_TASKS}")
    if not paths:
        print("FAIL: no chunk files found. Did the rsync back from Apuana run?")
        sys.exit(1)

    found_tasks = set()
    rows = []
    for p in paths:
        blob = json.load(open(p))
        found_tasks.add(blob["task_id"])
        if blob["n_tasks"] != N_TASKS or blob["n_trials"] != N_TRIALS:
            problems.append(f"{os.path.basename(p)}: written by a different plan "
                            f"(n_tasks={blob['n_tasks']}, n_trials={blob['n_trials']})")
        if [tuple(c) for c in blob["combos"]] != PREVIEW_COMBOS:
            problems.append(f"{os.path.basename(p)}: different combo list")
        rows.extend(blob["results"])

    missing_tasks = sorted(set(range(N_TASKS)) - found_tasks)
    if missing_tasks:
        problems.append(f"missing chunks for task ids {missing_tasks} "
                        f"(check logs/preview_curves_*_<id>.err on the cluster)")

    planned = set(build_units())
    got = Counter((r["combo_index"], r["realization"]) for r in rows)
    missing_units = sorted(planned - set(got))
    dupes = sorted(u for u, c in got.items() if c > 1)
    if missing_units:
        problems.append(f"{len(missing_units)} of {len(planned)} work units missing, "
                        f"first few: {missing_units[:5]}")
    if dupes:
        problems.append(f"{len(dupes)} work units computed more than once: {dupes[:5]}")

    failed = [r for r in rows if not r["percolated"]]
    if failed:
        problems.append(f"{len(failed)} units never percolated within "
                        f"{MAX_ATTEMPTS} draws: "
                        f"{[(r['N'], r['d'], r['realization']) for r in failed[:5]]}")

    print(f"Work units:  {len(rows)}/{len(planned)}")
    print()
    for ci, (N, a, d) in enumerate(PREVIEW_COMBOS):
        mine = [r for r in rows if r["combo_index"] == ci and r["percolated"]]
        att = [r["attempts"] for r in mine]
        rate = len(att) / sum(att) if sum(att) else 0.0
        print(f"  N={N:3d} a={a} d={d:.2f}  {len(mine):2d}/{N_TRIALS} realizations  "
              f"{sum(att):7d} draws  percolation rate {rate:.5f}"
              + (f"  (median {int(np.median(att))}, max {max(att)} draws/realization)"
                 if att else ""))
    print()
    if problems:
        print("PROBLEMS FOUND:")
        for x in problems:
            print(f"  - {x}")
        sys.exit(1)
    print("OK: all units present exactly once, all percolated. Safe to --aggregate.")


def cmd_aggregate():
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "chunk_*.json")))
    if len(paths) != N_TASKS:
        print(f"WARNING: found {len(paths)} chunk files, expected {N_TASKS}. "
              f"Aggregating what is present.")
    rows = []
    for p in paths:
        rows.extend(json.load(open(p))["results"])
    print(f"Loaded {len(rows)} units from {len(paths)} chunks")

    summary = []
    for ci, (N, a, d) in enumerate(PREVIEW_COMBOS):
        mine = [r for r in rows if r["combo_index"] == ci and r["percolated"]]
        R_effs = [r["R_eff"] for r in mine]
        n = len(R_effs)
        curve = {}
        for layer_d in LAYER_DISTANCES:
            vals = [r["layer_means"][str(layer_d)] for r in mine
                    if str(layer_d) in r["layer_means"]]
            if vals:
                nv = len(vals)
                sd = float(np.std(vals, ddof=1)) if nv > 1 else 0.0
                curve[str(layer_d)] = {"mean": float(np.mean(vals)), "std": sd,
                                       "sem": sd / np.sqrt(nv), "n": nv}
        sd_R = float(np.std(R_effs, ddof=1)) if n > 1 else 0.0
        summary.append({"N": N, "a": a, "d": d, "n": n,
                        "mean_ohm": float(np.mean(R_effs)) if R_effs else None,
                        "sem_ohm": sd_R / np.sqrt(n) if n else None,
                        "total_attempts": sum(r["attempts"] for r in mine),
                        "onion_curve": curve})
        print(f"  N={N:3d} a={a} d={d:.2f}  n={n}/{N_TRIALS}  "
              f"R_eff={summary[-1]['mean_ohm']:.1f} +/- {summary[-1]['sem_ohm']:.1f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for row in summary:
        ds = sorted(float(k) for k in row["onion_curve"])
        means = [row["onion_curve"][str(x)]["mean"] for x in ds]
        sems = [row["onion_curve"][str(x)]["sem"] for x in ds]
        ax.errorbar(ds, means, yerr=sems, fmt='o-', capsize=3, markersize=4, linewidth=1.4,
                    label=f"$N$={row['N']}, $a$={row['a']}, $d$={row['d']} ($n$={row['n']})")
    ax.set_yscale('log')
    ax.set_xlabel(r'Distance from reference node, $r$')
    ax.set_ylabel(r'Mean $R_{\mathrm{eff},ij}$ per layer ($\Omega$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    fig_path = os.path.join(HERE, "..", "figures", "partA_preview_onion_curves_errorbars.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    json_path = os.path.join(RESULTS_DIR, "preview_curves_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {os.path.normpath(fig_path)}\nWrote {json_path}")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x != "--force"]
    force = "--force" in sys.argv
    if args and args[0] == "--aggregate":
        cmd_aggregate()
    elif args and args[0] == "--verify":
        cmd_verify()
    else:
        task_id = int(args[0]) if args else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        cmd_task(task_id, force=force)
