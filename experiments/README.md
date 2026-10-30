# `experiments/`: the full run archive

Every run this project ever produced, kept rather than pruned, so any figure or number in
the thesis can be traced back to the data it came from.

Folders are numbered automatically by `utils/experiment_manager.py`. The number records
the order runs happened, nothing else. Most of these are exploratory single-network runs
from the development of the pipeline; a handful carry the results the thesis is built on.

---

## Start here

If you are looking for the runs behind the reported results, it is these, all made under
the clipping boundary rule of 2026-09-15 (see the main README). The earlier canonical runs
under the rejection rule, `exp_843`–`exp_869`, and everything before the merge-geometry
correction of 2026-08-31 (`exp_813`, `exp_814`, `exp_829`, `exp_840`, ...) are kept as
history and are not cited:

| Folder | Contents |
|---|---|
| `exp_883/` | 2D baseline 4×3 `(N, a)` sweep: box plots, heatmaps, one representative network per cell (`exp_870` is the same seeded run before the figure titles were removed; the trials agree to 1e-13) |
| `exp_871/` | 2D extended adaptive `(N, a)` grid, copied from the cluster run (`cluster/experiments_clip/`), including the `R_eff` against `N·a²` collapse test |
| `exp_872/` | Fixed `N = 70, a = 0.6` thickness sweep and the three robustness `(N, a)` combinations |
| `exp_873/` | Analysis of the 420-combination scalar `(N, a, d)` grid, 7800 individual resistance values, with the three-parameter power-law fit (`fit_3d`) |
| `exp_876/` | `multimeter_r_eff.ipynb`: reference network (`N = 40, a = 0.6`, seed 11), manual probes, onion protocol validation against the direct solve, five-seed monotonicity check (`exp_875` is an aborted execution of the same notebook) |
| `exp_877/` | The onion-curve study: matched-group comparisons for all 32 groups, shape spread and the ANOVA; `partC_group_7_*` and `partC_group_31_*` are the thesis panels (`exp_874` is the same execution before those two groups were chosen) |
| `exp_884/` | Inverse notebook with the clip-rule pool (2919 trials): power-law and critical-form fits, solution curves and the 22-candidate validation loaded from `cluster/validation_results/` (`exp_878`–`exp_884` are earlier executions of the same notebook, superseded when the figure titles were removed) |

They are discussed in [`../docs/RESULTS.md`](../docs/RESULTS.md).

---

## What a run folder holds

```
exp_N/
├── experiment_summary.json     Complete record of the run
├── network_data.json           Network geometry, reloadable
├── nodes_detailed.json
├── edges_detailed.json
├── percolating_paths.json
├── all_intersections.json
├── lines_generated.json
├── *.png                       Figures
└── experiment_N_report.pdf     Generated report
```

Sweep runs replace the per-network files with a single summary such as
`effective_resistance_sweep_summary.json` or `r_eff_ij_vs_N_a_d_summary.json`, holding
every individual trial rather than only the aggregates.

`all_experiments_report.pdf` is the cumulative report across every run.

---

## Working with the archive

```python
from experiment_manager import setup_experiment, load_network

exp = setup_experiment()      # creates the next free exp_N/ folder
net = load_network(813)       # reloads a previous run's geometry
```

To regenerate figures for existing runs from their saved JSON, without recomputing
anything:

```bash
python utils/rebuild_experiments.py
```

---

## Two irregularities, recorded rather than tidied away

**Renumbered cluster runs.** `exp_813` and `exp_814` were executed on Apuana, where they
were written as `exp_1` and `exp_2` because that machine's `experiments/` directory was
empty at the time. They were renumbered when synced back, to avoid colliding with
unrelated local runs already holding those numbers. The `Results directory: .../exp_1`
lines in the saved notebook outputs are the accurate record of what ran on the cluster and
do not match where the files now live.

**Repeated runs of the same notebook.** `r_eff_ij_vs_N_a_d.ipynb` was re-executed several
times on 2026-08-24 while a statistical error in its analysis was being corrected, leaving
`exp_835` through `exp_840`. `exp_840` was the canonical run of that analysis until the
merge-geometry correction of 2026-08-31, after which the notebook was re-executed on the
regenerated grids (`exp_858`), and again after the boundary-rule change of 2026-09-15;
`exp_877` is now canonical and `exp_840` and `exp_858` are history.

**`exp_NNN_alt` folders.** Twelve folders numbered 508 through 519 exist in two versions.
Both are real runs with different contents; a sync conflict gave them the same number.
The second copy was originally named `exp_NNN (1)`, renamed here to `exp_NNN_alt` to keep
spaces and parentheses out of the repository paths. No data was discarded.
