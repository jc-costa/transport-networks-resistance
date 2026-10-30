# Effective Resistance as an Inverse Descriptor of Random Stick Networks

Simulation and analysis code for an MSc dissertation on charge transport in random
conductive-stick networks, the idealized model behind materials such as nanowire and
nanotube transparent conductive films.

The project builds random networks of conducting sticks, turns the percolating cluster
into a resistor network, solves it exactly through the graph Laplacian, and studies how
the resulting effective resistance depends on the geometric parameters of the film. It
then asks the inverse question: given a measured resistance, can the geometry be
recovered?

Everything here is self-contained: the pipeline, the validation suites, the SLURM scripts
used for the large sweeps, and the raw output of every run.

---

## The question

A conductive film made of randomly dispersed sticks is characterized by a few geometric
parameters:

| Symbol | Meaning |
|---|---|
| `N` | number of sticks in the sample |
| `a` | stick length (aspect ratio relative to the unit box) |
| `d` | film thickness, for the 3D model |
| `R0` | resistance per unit length of a stick |

The **forward problem** is how the effective source-to-drain resistance `R_eff` depends on
`(N, a, d)`. The **inverse problem** is the one that matters experimentally: a
measurement gives a single number, `R_eff`, and the question is which geometries could
have produced it.

The central result of this project is that the inverse problem is genuinely
underdetermined if you only measure the scalar resistance, and that a richer
measurement protocol recovers information the scalar throws away.

---

## The model

**Network generation (2D).** Sticks of length `a` are deposited at uniform density `N`
per unit area, with uniform orientations, and observed through the unit square: centres
are drawn in the square enlarged by `a/2` on every side, `round(N·(1+a)²)` of them, and
every stick is clipped to the film (Liang–Barsky); a stick that misses the film is
dropped. `N` is therefore a density, the number of sticks actually in a realization is
recorded as `n_sticks`, and the film is not depleted near its edges. (The earlier
rejection rule, which redrew any stick leaving the square, is still available as
`boundary="reject"` but is used for nothing reported here.) Every pairwise intersection
becomes a graph node. The sticks are cut at those intersections into segments, and each
segment becomes an edge.

**Network generation (3D).** Two random finite segments in 3D are generically skew and
almost surely never intersect exactly, so the exact-intersection test has no 3D
equivalent. Sticks are instead treated as thin cylinders and two of them are bonded when
the closest distance between their segments falls at or below `contact_distance`, the
standard continuum-percolation bonding criterion for fibers (Pike and Seager 1974;
Balberg, Binenbaum and Wagner 1984). Each contact becomes one node, placed at the
midpoint of the two closest-approach points. Stick directions are sampled by normalizing
a 3D standard-normal vector, which is exactly uniform on the sphere. Sampling two
independent uniform angles instead would bias mass toward the poles, and
`percolation_3d.ipynb` tests explicitly for this.

**Film thickness.** Passing `box_dims=(1.0, 1.0, d)` confines the sticks to a slab
instead of the full unit cube, while the transport direction and the source/drain
convention stay untouched; the same clipping rule applies, with `N` centres in the slab
on average (volumetric density `N/d`). `d = 1` recovers the isotropic cube exactly, and
`d` going to zero approaches the 2D thin-sheet limit.

**Resistance model.** Each edge carries `R_edge = R0 * L_edge`, proportional to its own
Euclidean length, rather than a uniform per-edge resistance. `distance_dependent_resistance.ipynb`
defines and validates this model against hand-computable cases.

**Solving the circuit.** The percolating cluster is pruned of dead ends, the source and
drain boundary regions are each merged into a single node, and the weighted Laplacian
`L = Aᵀ G A` is assembled from the incidence matrix `A` and the edge-conductance diagonal
`G`. Removing the reference row and column gives an invertible reduced matrix, and
`R_eff` follows either from the solved node voltages or, equivalently, from the Laplacian
pseudoinverse:

```
R_eff(i, j) = (e_i - e_j)ᵀ L⁺ (e_i - e_j)
```

The pseudoinverse form matters for the onion protocol below: it costs one decomposition
per network instead of one dense solve per probed pair.

---

## What was found

Full numbers, figures, and the exact runs behind them are in **[docs/RESULTS.md](docs/RESULTS.md)**.
Everything below is from the runs regenerated under the clipping boundary rule
(2026-09-15), after the merge-geometry correction of 2026-08-31 (see the reproducibility
notes at the end); the pool of trials counts each distinct sweep once. In short:

**1. The forward scaling.** `R_eff` falls monotonically with both `N` and `a`. Pooled over
2919 unique percolating trials on 101 `(N, a)` points, the unconstrained fit gives

```
R_eff ≈ 28254 · N^(-1.694) · a^(-2.459)      R² = 0.904
```

**1b. The percolation form.** Fitted as `R_eff = A·a^q·(N·a² − c)^(−t)`, the same trials
give `q = 0.99`, the `a¹` of the wire-limited model, and `t = 1.44 [1.36, 1.51]` with a
free threshold `c = 2.28` that absorbs the rounding of the transition in a finite box
(`R² = 0.921`). A dedicated near-threshold sweep in boxes of 5 to 10 stick lengths places
the half-percolation density at `N·a² = 5.5–5.7`, the infinite-system threshold, shows
`R_eff/(R0·a)` collapsing across box sizes above `N·a² ≈ 8`, and gives `t = 1.10–1.23` with
the threshold pinned there. The dense wire-limited limit `R ∝ a/(N·a²)` does not fit.

**2. The `N·a²` collapse does not survive scrutiny.** Continuum stick-percolation theory
uses the excluded-area parameter `N·a²` to locate the percolation *threshold*, and it is
tempting to assume `R_eff` collapses onto the same variable away from the threshold. It
does not. The constrained single-variable fit is worse (`R² = 0.848` against `0.904`), and
the general fit's exponents give `p / 2q = 0.34` where a true collapse requires 1.0. The
inversion therefore uses the general fit.

**3. The scalar inversion is underdetermined, and it is validated.** A target `R_eff`
pins down a whole *curve* of `(N, a)` solutions rather than a point. Out-of-sample
validation on the 22 unique candidates the pipeline can reach (7 targets × 4 geometries,
minus 6 the curve places outside the node budget or below the stick floor; 30 fresh
networks each) reproduces the target to a median absolute error of 16%, 19 of 22 within
30%; the errors are about 10% for long sticks and largest at the percolation threshold and
where the straight-line power law misses the curvature of the critical form.

**4. Thickness enters, with a fitted exponent.** Over the 420-combination `(N, a, d)` grid
(341 combinations with at least one percolating realization, 7800 realizations at 24 per
combination),

```
R_eff ≈ 8.64e5 · N^(-1.892) · a^(-2.634) · d^(0.804)      R² = 0.854
```

against `R² = 0.635` without the `d` term. `R_eff` *rises* with `d` at fixed stick count
per unit area, because the same sticks are spread through more volume until percolation
is lost; at fixed density `N/d` it falls by a factor of three over a doubling of `d`, as
the exponents predict in direction and a 250-realization control confirms.

**5. Curve shape carries information the scalar collapses away.** This is the main
contribution. Instead of a single source-to-drain number, the **onion-layer protocol**
sweeps `R_eff(reference, j)` outward through concentric distance shells, producing a
curve rather than a scalar. Grouping `(N, a, d)` triples whose *scalar* `R_eff` matches
within 15%, precisely the triples the scalar inversion cannot separate, a one-way ANOVA
per group and shell asks whether their onion curves still differ.

On the full grid at 24 realizations per point, 337 of the 341 combinations form 32
matched groups and 234 group-and-shell tests; 81 (34.6%) are significant at `p < 0.05`,
against roughly 5% expected by chance, and 19 survive Bonferroni correction, with the
strongest groups at `p < 1e-5`. The fraction stays between 22% and 45% for matching
tolerances from 5% to 30%. A regression validated on combinations never seen in
training then says what the information is: the curve recovers the film thickness `d`
(out-of-sample `R²` rising from 0.15 on a single realization to 0.76 when 24 are
averaged) while the scalar recovers nothing about `d` however many are averaged; neither
observable separates `N` from `a` at fixed resistance.

One finding is reported as a correction rather than buried. On the earlier
rejection-rule grid the curve appeared to recover `a` (`R² = 0.55`). That was the depleted
rim of the earlier generator, one stick length wide, read as a signal; it disappeared
when the rule was corrected, and the recovery of `d` survived.

**6. What an onion curve is not.** The dependence of the pairwise resistance on probe
separation is linear in the sparse 3D networks, as in a wire, and sub-linear over most
of the box in dense 2D networks (local exponent 0.5–0.8, the parallel-path effect), with
a mild upturn at the far edge where the nodes on clipped sticks have fewer
continuations. An onion curve of a film a few stick lengths across is not the
resistance-versus-distance law of an infinite homogeneous film.

---

## Repository layout

```
.
├── notebooks/      Analysis notebooks, one per question. Start here.
├── utils/          The pipeline itself, imported by the notebooks.
├── cluster/        SLURM array jobs for the large sweeps, and their raw chunk output.
├── experiments/    Every run ever produced, auto-numbered exp_N/ with data and figures.
├── figures/        Standalone illustrative figures.
├── docs/           Results write-up and methodology notes.
└── Paths.jpg/.pdf  Reference diagram of the path data structures.
```

Each directory has its own README with a full index:
[notebooks](notebooks/README.md) ·
[utils](utils/README.md) ·
[cluster](cluster/README.md) ·
[experiments](experiments/README.md)

---

## Getting started

```bash
git clone <this-repo>
cd ColabProject
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open any notebook in `notebooks/`. The path-setup cell resolves `../utils` relative to the
notebook, so imports work with no further configuration.

**On Google Colab**, mount your Drive with the project at `MyDrive/ColabProject/`, then
uncomment the Colab `utils_path` line in the first cell and comment out the local one.

**A note on running notebooks.** Jupyter and Colab always start with an empty runtime, but
the outputs saved in the file stay visible from the previous run. Editing a cell in the
middle of a notebook and running only that cell will raise `NameError` for variables that
merely *look* like they already exist. Always use **Runtime > Run all** first, then edit
and re-run individual cells.

### Suggested reading order

| # | Notebook | What it establishes |
|---|---|---|
| 1 | [`calculating_the_impedance_of_a_graph.ipynb`](notebooks/calculating_the_impedance_of_a_graph.ipynb) | The incidence-matrix method, from scratch |
| 2 | [`distance_dependent_resistance.ipynb`](notebooks/distance_dependent_resistance.ipynb) | The `R_edge = R0 · L` edge model, validated |
| 3 | [`test_random_network_impedance.ipynb`](notebooks/test_random_network_impedance.ipynb) | The whole 2D pipeline end to end |
| 4 | [`effective_resistance_vs_N_a.ipynb`](notebooks/effective_resistance_vs_N_a.ipynb) | The forward problem in 2D |
| 5 | [`inverse_N_a_from_R_eff.ipynb`](notebooks/inverse_N_a_from_R_eff.ipynb) | The inverse problem, and why it is underdetermined |
| 6 | [`percolation_3d.ipynb`](notebooks/percolation_3d.ipynb) | The 3D contact-based generalization |
| 7 | [`effective_resistance_vs_d.ipynb`](notebooks/effective_resistance_vs_d.ipynb) | Film thickness as a third parameter |
| 8 | [`multimeter_r_eff.ipynb`](notebooks/multimeter_r_eff.ipynb) | The onion-layer probe protocol |
| 9 | [`r_eff_ij_vs_N_a_d.ipynb`](notebooks/r_eff_ij_vs_N_a_d.ipynb) | **The main result** |

---

## Reproducing the results

The small notebooks run locally in seconds. The large grids were produced on the Apuana
cluster (CIn/UFPE) as SLURM array jobs, and their raw output is committed under
`cluster/*_results/`, so the analysis notebooks reload it instead of recomputing.

```bash
cd cluster
mkdir -p logs
sbatch run_effective_resistance_d_array.sbatch   # scalar R_eff over the (N, a, d) grid
sbatch run_onion_grid_array.sbatch               # onion curves over the same grid
sbatch run_critical_sweep_array.sbatch           # near-threshold sweep, 33 points
sbatch run_preview_curves_array.sbatch           # 150 preview realizations
sbatch prepare_validation_plan.sbatch            # pool + fit + freeze the 28 inverse candidates
sbatch run_validation_array.sbatch               # 30 fresh networks per candidate
```

Each task writes one `chunk_NN.json`. Once the full set is present, the corresponding
notebook detects it automatically and loads instead of recomputing. The two large grids
can also be run as a single 48-worker job on the `short-complex` partition
(`run_scalar_grid_parallel.sbatch`, `run_onion_grid_parallel.sbatch`), which writes the
same chunk files. See [cluster/README.md](cluster/README.md) for the details of each job.

---

## Verification

The physics is checked against cases with known answers rather than trusted:

- Hand-built networks with closed-form resistance (single edge, ladder, triple parallel,
  three-branch star) are reproduced exactly, in both 2D and 3D.
- A **Wheatstone bridge**, the simplest topology that is not series/parallel reducible, is
  cross-checked independently through the Laplacian pseudoinverse.
- **2D/3D consistency**: a real random 2D network embedded at `z = 0` and run through the
  3D contact-based pipeline with a near-zero contact distance reproduces the 2D result
  exactly.
- **Invariance tests**: injecting self-loops or electrically inert node-loops must leave
  `R_eff` unchanged, and pruning dead ends must not alter it either.
- **Isotropy**: 3D stick directions are tested for uniformity on the sphere.
- The pseudoinverse fast path and the per-probe solve path are checked to agree to
  numerical precision.

Unit tests live in `utils/test_*.py` and are exercised throughout the notebooks:

```bash
cd utils
python test_merge.py
python test_multimeter.py
python test_percolation_3d.py
```

---

## Reproducibility note

Earlier versions of the sweep scripts seeded the random draws with
`hash((N, a, d, 'tag'))`. Python randomizes string hashing per process unless
`PYTHONHASHSEED` is fixed, so any seed tuple containing a string silently produced a
different seed on every run, including across the parallel tasks of a single array job.
The code looked deterministic and was not.

This is fixed by `stable_seed()` in `utils/pipeline_utils.py`, which derives seeds through
`hashlib`. Data collected before the fix (the scalar `(N, a, d)` grid in
`experiments/exp_829/`) remains a valid random sample and is used as such, but it cannot
be regenerated byte for byte from today's seed formula. Everything collected afterwards
can, and the cluster results were spot-checked on a laptop by redrawing individual units
from their seeds (they reproduce bit for bit).

---

## Citation

```bibtex
@mastersthesis{costa2026percolation,
  author  = {Jefferson Carlos Lima da Costa},
  title   = {Effective Resistance as an Inverse Descriptor of Random Geometric Networks},
  school  = {Centro de Inform\'atica, Universidade Federal de Pernambuco},
  year    = {2026}
}
```

## License

[MIT](LICENSE), Copyright (c) 2026 Jefferson Carlos Lima da Costa.

**Merge-geometry correction (2026-08-31).** `distance_dependent_impedance` used to measure
edge lengths on the merged geometry, where each terminal sits at the centroid of the face
it replaces, which inflated every whole-face `R_eff` by an amount that grows with density.
Edge lengths are now measured on the pre-merge geometry, with the harmonic-mean length for
collapsed parallel edges. Every grid was regenerated after the fix (`cluster/backups/`
holds the pre-fix data with checksums, and `cluster/verify_rerun_vs_backup.py` checks that
only the scalar column moved). The numbers in this README and in `docs/RESULTS.md` are
post-correction; runs numbered below `exp_843` are not.

**One sweep, once (2026-09-14).** Because seeds are stable, re-running a sweep reproduces
its trials exactly, and the `experiments/` archive holds several such re-runs of the same
grids. The pooling in `inverse_N_a_from_R_eff.ipynb` and `cluster/run_validation_task.py`
now skips any `(N, a)` cell whose `R_eff_values` vector has already been pooled, so the
fit sees each trial once.

**Clipping boundary rule (2026-09-15).** The generator used to redraw any stick whose
second endpoint fell outside the box, which depleted a rim of width `~a`, biased the
orientations there and made `N` a count of whole sticks rather than a density. It now
places centres at uniform density in the box enlarged by `a/2` and clips every stick to
the film (`utils/point_generator.py`, `utils/point_generator_3d.py`, `boundary="clip"`),
and the intersection tolerance was tightened from `1e-3` to `1e-9`, since clipped sticks
that end near each other at the edge were otherwise occasionally joined. Every sweep was
regenerated under the new rule, at 24 realizations per combination for the 3D grids
(previously 6), with the onion grid storing each realization's curve as its own record
(`per_realization_curves`). The rejection-rule results are kept under
`cluster/*_results_reject/` and `experiments/exp_843`–`exp_869`; every summary records
its boundary rule and the pooling scripts refuse to mix the two. The onion-grid boost
(`run_onion_grid_boost_task.py`) was retired by the higher base count and feeds no result.
