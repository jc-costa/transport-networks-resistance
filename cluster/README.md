# `cluster/`: large sweeps on SLURM

The large parameter grids in this project are too expensive to run in a notebook. The
impedance solve is a dense `O(n_nodes³)` operation and the node count grows sharply with
`N`, with `a`, and (in 3D) with *shrinking* film thickness `d`. These jobs ran on the
**Apuana** cluster at CIn/UFPE.

**Their output is committed.** The `*_results/` directories hold the raw JSON chunks each
job produced, so the analysis notebooks reload them instead of recomputing. Nothing here
has to be re-run to reproduce a figure.

---

## How the jobs are structured

Every sweep follows the same pattern:

1. A `run_*_task.py` script takes an array-task index, computes only its own slice of the
   grid, and writes `<results_dir>/chunk_NN.json`.
2. A `run_*_array.sbatch` file submits the whole set as a SLURM array job.
3. The corresponding notebook checks for a complete set of chunk files. If they are all
   present it loads them; otherwise it computes serially in-notebook, checkpointing after
   every combination so an interrupt does not lose the sweep.

Because the grid slicing and the seed derivation are identical across scripts, results are
independent of how the work was split across tasks.

```bash
cd cluster
mkdir -p logs
sbatch run_onion_grid_array.sbatch
```

**On the QOS setting.** The jobs specify `--qos=simple` with `--array=0-19%4`, running
four tasks concurrently. The `complex` QOS was tried and is worse here: it capped
concurrency at one.

---

## The jobs

| Job | Task script | Output | Feeds |
|---|---|---|---|
| `run_effective_resistance_d_array.sbatch` | `run_effective_resistance_d_task.py` | `effective_resistance_d_results/` (20 chunks) | `effective_resistance_vs_d.ipynb` Part C |
| `run_onion_grid_array.sbatch` | `run_onion_grid_task.py` | `onion_grid_results/` (20 chunks) | `r_eff_ij_vs_N_a_d.ipynb` Parts B and C |
| `run_onion_grid_boost_array.sbatch` | `run_onion_grid_boost_task.py` | `onion_grid_boost_results/` (18 chunks) | `r_eff_ij_vs_N_a_d.ipynb` Part D |
| `run_validation_array.sbatch` | `run_validation_task.py` | `validation_results/` (28 candidate files) + `validation_plan.json` | `inverse_N_a_from_R_eff.ipynb` section 6 |
| `run_preview_curves_array.sbatch` | `run_preview_curves_task.py` | `preview_curves_results/` | `r_eff_ij_vs_N_a_d.ipynb` Part A, thesis probe-distance section |
| `run_critical_sweep_array.sbatch` | `run_critical_sweep_task.py` | `critical_sweep_results/` (33 points) | `critical_sweep_fit.py`, thesis critical-form subsection |
| `run_random_networks_array.sbatch` | `run_random_networks_task.py` | (batch statistics) | `test_150_random_networks.ipynb` section 4 |
| `run_backbone_array.sbatch` | `run_backbone_task.py` | `backbone_results/` (33 points) | `backbone_analysis.py`, thesis Sec. "The electrical backbone near the threshold" |
| `run_junction_array.sbatch` | `run_junction_task.py` | `junction_results/` (33 points) | `junction_analysis.py`, thesis Sec. "Junction resistance" |
| `run_junction_variants_array.sbatch` | `run_junction_variants_task.py` | `junction_variants_results/` (33 points) | `junction_variants_analysis.py`, thesis Secs. "Distributed junction resistances", "Metallic and semiconducting sticks" |

Since the 2026-08-31 rework the array jobs request 8 CPUs and 8 GB per task on the
`short-simple` partition (the earlier 2 CPU / 4 GB sizing is gone), with the BLAS thread
pool pinned to the allocation. Walltimes range from 20 minutes for the random-network
batch to 4 hours for the scalar `(N, a, d)` grid. The array throttle is set above what the
QoS grants because the QoS enforces the real cap; surplus tasks wait with
`REASON=QOSMaxJobsPerUserLimit`.

### Whole-notebook jobs

Two scripts re-execute an entire notebook end to end through `nbclient` as a single job,
rather than slicing a grid. Simpler to launch, with no chunk files to assemble, but
serial.

| Script | Runs |
|---|---|
| `run_effective_resistance_sweep.sbatch` | `effective_resistance_vs_N_a.ipynb` |
| `run_effective_resistance_d_sweep.sbatch` | `effective_resistance_vs_d.ipynb`, Parts A, B, and C |
| `run_inverse_validation.sbatch` | `inverse_N_a_from_R_eff.ipynb` |

### Local analysis scripts

Several scripts here are not SLURM jobs. They run on a laptop against committed results
and back specific thesis sections.

| Script | Reads | Writes | Feeds |
|---|---|---|---|
| `probe_distance_figure.py` | `preview_curves_results/`, the thesis' `docs/index.html` (the 20 supplementary 2D networks) | the four `probe_distance_*.png` panels in the thesis' `images/resultados/`; prints every number quoted in the subsection | thesis Fig. "probe-distance" |
| `regenerate_preview_networks.py` | `preview_curves_results/` (for verification only) | `preview_curves_decomposition.json`; `--tables` prints the two LaTeX tables of Appendix D and the resistor-chain control | thesis Appendix D |
| `thesis_figures_2d.py` | `experiments/exp_871/effective_resistance_sweep_summary_ext.json` | the extended-sweep box plot (`--out`), without re-running the sweep | thesis Fig. "extended-boxplots" |
| `probe_figures.py` | nothing (draws the reference network `N = 40, a = 0.6`, seed 11) | `probe_sourcedrain.png`, `probe_farpair.png`, `probe_mid.png`, `probe_near.png` in the thesis; prints the probe numbers, self-checks and the all-pairs range | thesis Sec. "Single-pair probing", Fig. "probe" |
| `onion_tolerance_sweep.py` | `onion_grid_results/`, `effective_resistance_d_results/` | `onion_grid_results/tolerance_sweep_summary.json` (with `--json`); prints the CV table and the grouping+ANOVA at nine tolerances | thesis Tables "onion-cv", "onion-tolerance-sweep" |
| `curve_inversion.py` | `onion_grid_results/` | `curve_inversion_summary.json`, `curve_inversion_k.png` (with `--json`, `--figure`) | thesis Sec. "Inverting (N, a, d) from the onion curve" |
| `density_control.py` | nothing (draws `N/d = 1500` at `a = 0.2`, 250 realizations each, process pool) | `figures/partA_constant_density_control.json` | thesis Table "density-control" |
| `critical_sweep_fit.py` | `critical_sweep_results/` | fits and `critical_sweep.png` (with `--figure`) | thesis Sec. "The near-threshold sweep" |
| `junction_analysis.py`, `junction_variants_analysis.py` | `junction_results/`, `junction_variants_results/` | `junction_summary.json`, `junction_variants_summary.json` (with `--json`), the six `junction_*.png` figures (with `--figures DIR`); print every number of the junction section | thesis Table "junction", Figs. "junction-crossover", "junction-red-share", "junction-dispersion", "junction-bimodal" |
| `backbone_analysis.py` | `backbone_results/` | `backbone_summary.json` (with `--json`), the five `backbone_*.png` figures (with `--figures DIR`); prints the table and every number of the section | thesis Table "backbone", Figs. "backbone-red", "backbone-dead-bypass", "backbone-positions" |
| `export_supplement_networks.py` | nothing (draws the 20 supplement networks) | `supplement_networks_clip.json` / `--html` | thesis `docs/index.html` |

`regenerate_preview_networks.py` redraws the 150 preview networks from their seeds, which
is possible because `run_preview_curves_task.py` seeds each `(combo, realization)` unit on
its own. It refuses to continue unless every regenerated network reproduces its stored
`R_eff` and shell means (they do, to a relative ~1e-11 or better). A few minutes with
`--workers 7` under the clipping rule.

### Plan files

`run_onion_grid_boost_task.py` and `run_validation_task.py` need their work enumerated
before submission, since the set of combinations depends on the analysis of a previous
run. `onion_grid_boost_plan.json` and `validation_plan.json` are those enumerations, and
are committed so the exact submitted workload is on record.

```bash
python run_onion_grid_boost_task.py --prepare   # regenerates the plan, run before sbatch
```

---

## Backbone measurement on the critical sweep

`run_backbone_task.py` re-draws the 33 x 30 networks of `run_critical_sweep_task.py` from
their seeds (same plan, same `stable_seed(N, a, 'critical')`, same draw order) and refuses
to continue unless each regenerated `R_eff` matches `critical_sweep_results/` to a
relative 1e-9. On each network it inverts the grounded Laplacian once (Cholesky) and reads
off red bonds (`R_eff,uv = r_uv` plus the series condition through the terminals), cut
vertices (tight triangle `R_sb + R_bd = R_sd`), the bypass ratio of every edge, and the
current-free fraction of the pruned graph, each cross-checked against the current field
and against Tarjan bridges / articulation points. The topological routines are hand-written
(checked against networkx on 300 random graphs) so the job needs only numpy and scipy.
Measurement cost is ~0.3 s per 2700-node network; regeneration dominates, so the sizing is
that of the critical sweep. Result (2026-09-17, Apuana): 990 networks, zero disagreements
between criteria.

## Junction resistance on the critical sweep

`utils/junction_model.py` splits every crossing into two nodes joined by a junction edge
(rho = R_j / R0 a) and solves the graph sparsely; rho = 0 reproduces the stored wire-limited
R_eff to 1e-12 (contracted) and the tasks abort if it does not. `run_junction_task.py` sweeps
ten values of rho (uniform R_j) on the 33 x 30 critical-sweep networks and records R_eff, the
red segments / red junctions and their resistance shares; `run_junction_variants_task.py`
runs lognormal R_j at fixed mean (rho_bar in {0.1, 1, 10}, sigma in {0.5, 1, 1.5}, 5 draws)
and a metallic/semiconducting assignment (p_m = 1/3, kappa in {10, 100, 1000}, 5 assignments)
on the same networks. Draws use a generator seeded per (point, realization) independent of
the geometry stream. `notebooks/junction_resistance.ipynb` has the hand-solved case, the
validation and a laptop-scale preview. Results (2026-09-18, Apuana): both sweeps complete,
990 networks each.

## Paired grids

`run_onion_grid_task.py` reproduces the grid definition, the seed formula, and the
network-draw order of `run_effective_resistance_d_task.py` exactly. The consequence is
that a given `(N, a, d)` combination yields the *same network* in both datasets, so the
onion curves and the scalar resistances are paired one to one by construction rather than
matched after the fact. The whole matched-group analysis in
[`../docs/RESULTS.md`](../docs/RESULTS.md) depends on this.

The boosted rerun extends the same property. It reuses the same `stable_seed()` values, so
its first 6 realizations per combination are byte-identical to the original 6-realization
data (verified directly, maximum absolute difference 0.0). It is a strict extension of the
same draws rather than an independent resample.

This pairing only holds for data produced **after** the seed fix described in the main
[README](../README.md). The scalar grid in `experiments/exp_829/` predates it.

---

## A note on the results directories

The `*_results/` directories hold real data synced back from the cluster, not scratch
output. Running a task script locally overwrites the chunk it owns. Check what is there
before running anything that writes into these directories.

---

## Rerun after the merge-geometry fix (2026-08-31)

`pipeline_utils.distance_dependent_impedance` used to measure edge lengths on the
**merged** geometry, where `merge_boundary_nodes` parks each terminal at the average
position of the face it replaces. Under `R_edge = R0 * L_edge` that inflated the
resistance of every edge touching a terminal, biasing the whole-face scalar `R_eff`
**high**, and the bias grows with density: about 3% at `N=20, a=0.6`, about 31% at
`N=100, a=0.6`, up to 66% on individual realizations. It is not a constant offset, so it
moves fitted exponents, not just prefactors (a small local fit of `R_eff` against `N` at
`a=0.6` shifted from -1.075 to -1.219).

Everything computed through `run_onion_protocol_pinv` is unaffected, since that path
never merges. Only the scalar column needs regenerating.

**Order of operations:**

```bash
bash backup_results.sh                       # timestamped copy + MANIFEST.sha256
sbatch run_effective_resistance_d_array.sbatch
sbatch run_onion_grid_array.sbatch           # queues behind it: qos=simple caps 4 concurrent
python verify_rerun_vs_backup.py backups/<timestamp>
```

`verify_rerun_vs_backup.py` encodes what the fix must and must not do: the scalar
`R_eff` column has to move (downward, in general), while the onion curves and the network
draws (`n`, `perc_rate`) have to stay bit-for-bit identical, because the seeds did not
change and the pinv path does no merging. Curves that move mean something other than this
fix altered the pipeline, and the rerun should not be trusted.

**Outcome.** Both grids were regenerated on 2026-08-31 and verified against
`backups/20260831_111505/`: the scalar column moved, the curves and the draws did not.
The corrected scalars changed the matched groups (the five strongest went from
4, 6, 9, 11, 16 to 10, 28, 9, 7, 13), so `run_onion_grid_boost_task.py --prepare` was
re-run and `run_onion_grid_boost_array.sbatch` re-submitted; `onion_grid_boost_results/`
and `onion_grid_boost_plan.json` hold the post-correction boost. `experiments/exp_858/`
is the canonical analysis run for the onion study, `exp_868/` for the scalar `(N, a, d)`
grid and `exp_869/` for the inverse problem.

---

## Deduplicated pooling and validation re-run (2026-09-14)

Stable seeds mean a re-run of a sweep reproduces its trials exactly, and `experiments/`
holds several such re-runs of the post-fix grids (`exp_843`/`854`/`856`/`862` are one
baseline sweep, `exp_844`/`857` one extended sweep). `pool_and_fit()` in
`run_validation_task.py`, and the matching cell of `inverse_N_a_from_R_eff.ipynb`, now
skip any `(N, a)` cell whose `R_eff_values` vector has already been pooled, so the fit is
built on 2931 unique trials instead of the 6042 an earlier version counted; the plan
records `n_duplicate_cells_skipped`. The deduplicated fit
(`R_eff ≈ 22003 · N^-1.514 · a^-1.831`) changes one candidate by one stick
(`N = 347` to `346` at the 60 Ω, `a = 0.2` point).

The full validation was re-run on Apuana that day (`prepare_validation_plan.sbatch`, then
`run_validation_array.sbatch`). The 27 unchanged candidates reproduce the previous values
to a relative `1e-11` (BLAS noise between nodes); the previous run is in
`backups/20260914_190740/`. Note that the 680 Ω, `a = 0.2` candidate (`N = 70`,
`N·a² = 2.8`) reaches only 5 percolating networks before the 200-draw give-up, in both
runs. The three `.png` files that used to sit in `validation_results/` were figures from
the pre-correction fit and were removed; the thesis figures are regenerated from the
current data.

---

## Near-threshold sweep for the conductivity exponent (2026-09-14)

The critical-form fit of `inverse_N_a_from_R_eff.ipynb` section 3.2 spans the whole main
sweep (N a^2 from 3 to 85) in a box 1-5 stick lengths across, so its `t = 1.28` is an
effective exponent and its `c = 2.40` a finite-size threshold. `run_critical_sweep_task.py`
samples N a^2 from 4 to 20 in steps of 1-2, at a = 0.20, 0.15 and 0.10 (boxes 5, 6.7 and
10 stick lengths across; a = 0.10 stops at 12 to keep the dense solve under ~7000 nodes),
30 percolating realizations per point with a 6000-draw budget, one array task per point.
`critical_sweep_fit.py` fits `log R = log A_a - t log(N a^2 - c)` per box size and the
pooled form with the `a^q` prefactor, with cluster-bootstrap intervals, and `--xmax`
confines the fit to the critical region.

```bash
mkdir -p logs
python run_critical_sweep_task.py --plan          # the 33 points
sbatch run_critical_sweep_array.sbatch
python critical_sweep_fit.py --figure critical_sweep_results/collapse.png   # after syncing back
```

**Outcome (2026-09-15).** All 33 points completed with 30/30 realizations (longest task
132 min, `a = 0.15` at `N a^2 = 20`, 5720 nodes). Half-percolation at `N a^2 = 4.6, 4.9,
4.8` for the three boxes; `R_eff/(R0 a)` coincides across boxes above `N a^2 ~ 8` (the
`a^1` of the wire-limited model); the exponent trades off against the threshold offset
(`t` from 1.1 to 1.8 depending on convention), with the running exponent crossing over
toward 1 by `N a^2 ~ 20`. Numbers in `critical_sweep_results/critical_fit_summary.json` and
`docs/RESULTS.md`. `critical_sweep_running_exponent.png` there is the thesis figure.

---

## Curve inversion and the per-realization storage fix (2026-09-15)

`curve_inversion.py` regresses `(log N, log a, log d)` on the log shell means of the
onion grid (ridge and k-NN, grouped cross-validation by combination, plus an averaging
analysis over k realizations); results in `onion_grid_results/curve_inversion_summary.json`
and `docs/RESULTS.md` section 5.

While writing it, a storage defect surfaced: `run_onion_grid_task.py` and
`run_onion_grid_boost_task.py` stored each shell's `per_realization_means` with empty
shells skipped, so the lists of different shells are aligned by index only for
combinations in which every realization has every shell (137 of 359 in the grid, 9 of 69
in the boost). The ANOVA's per-realization normalization assumed alignment. Both scripts
now also write `per_realization_curves`, one dict per accepted realization in the order
of `R_eff_values` with `null` for empty shells; the existing chunk files predate this and
should be regenerated together with the boundary-rule change.

---

## Boundary rule change and full regeneration (2026-09-15)

`utils/point_generator.py` and `point_generator_3d.py` gained a `boundary` argument.
`"reject"` is the rule used until now (redraw any stick whose endpoints leave the box; it
depletes a rim of width ~a and, in thin 3D films, biases orientations into the film
plane). `"clip"`, now the default, treats the box as a window cut from a larger film:
stick centres are placed uniformly in the box enlarged by a/2 on every side, N centres per
unit area (volume), and each stick is clipped to the box. N is therefore a density of
stick centres; N a^2 keeps its meaning. `run_full_pipeline*` iterate over the sticks
actually present and record `n_sticks` and `boundary`. The bounding-box tolerance of the
2D intersection test went from 1e-3 to 1e-9 (clipped sticks ending on the same edge
produced near-miss spurious nodes at 1e-3; verified against brute force, 0 missed and
0 spurious in 40 networks under both rules).

Everything on this page was regenerated under `"clip"`: the 2D sweeps (notebook job;
`experiments_clip/` on the cluster, `experiments/exp_870` and `exp_871` locally), the
near-threshold sweep, the robustness batch, the scalar and onion `(N, a, d)` grids at
**24** realizations per combination (paired by seed, with the aligned
`per_realization_curves` record), the preview curves at the new combinations
`(150, 0.2, 0.10)`, `(150, 0.2, 0.15)`, `(125, 0.2, 0.10)` (measured rates 22.0%, 8.7%,
4.8%), and the inverse validation (22 unique candidates; a first run whose target
rounding and node cap produced four duplicate rows was discarded and is kept as
`validation_results_clip_dup30/`, and `run_validation_task.py` now records infeasible
candidates instead of moving them and refuses coincident targets or duplicate candidates).
The boost is not re-run: with the base grid at 24
realizations it feeds no result. The results directories in this folder are the clip-rule
ones; the rejection-rule ones are kept as `*_results_reject/` (and
`preview_curves_results_reject_rule/`, `preview_curves_decomposition_reject_rule.json`,
`figures/partA_constant_density_control_reject_rule.json`). Sweep summaries now carry `config.boundary`, and the pooling in
`run_validation_task.py` / `inverse_N_a_from_R_eff.ipynb` accepts only `"clip"` summaries,
so the old-rule `exp_843`-`exp_863` on the cluster cannot leak into the fit.
`export_supplement_networks.py` regenerates the twenty 2D networks of the thesis'
interactive supplement (`supplement_networks_clip.json`). Old-rule results are in
`backups/20260915_140512/`.

---

## Running a grid in one job (2026-09-15)

The Apuana docs (apuana.cin.ufpe.br, "Partições e Limites de Recursos") give the QoS
limits: `simple` allows 16 CPUs / 64 GB and caps a user at 4 concurrent jobs; `complex`
allows one job of 48 CPUs / 500 GB on `short-complex` (whole node, 2 days). The task
scripts are single-threaded Python, so a 20-task array on `simple` uses about four cores
at a time. `run_grid_parallel.py` runs the whole grid inside one `complex` job with a
process pool (`--workers 48`), one combination per process, BLAS pinned to one thread,
and writes the same 20 chunk files as the array; every combination seeds itself, so the
rows are identical to the array's (verified). `run_scalar_grid_parallel.sbatch` and
`run_onion_grid_parallel.sbatch` are the job scripts. `GRID_PARALLEL_TRIALS`,
`GRID_PARALLEL_LIMIT` and `GRID_PARALLEL_OUT` are smoke-test hooks.
