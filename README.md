# Effective Resistance of Random Stick-Percolation Networks

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

**Network generation (2D).** `N` sticks of length `a` are dropped at uniformly random
positions and orientations in the unit square. Every pairwise intersection becomes a
graph node. The sticks are cut at those intersections into segments, and each segment
becomes an edge.

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
convention stay untouched. `d = 1` recovers the isotropic cube exactly, and `d` going to
zero approaches the 2D thin-sheet limit.

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
In short:

**1. The forward scaling.** `R_eff` falls monotonically with both `N` and `a`. Pooled over
2835 individual percolating trials, the unconstrained fit gives

```
R_eff ≈ 10246 · N^(-1.28) · a^(-1.75)      R² = 0.848
```

**2. The `N·a²` collapse does not survive scrutiny.** Continuum stick-percolation theory
uses the excluded-area parameter `N·a²` to locate the percolation *threshold*, and it is
tempting to assume `R_eff` collapses onto the same variable away from the threshold. It
does not. The constrained single-variable fit is clearly worse (`R² = 0.776` against
`0.848`), and the general fit's exponents give `p / 2q = 0.37` where a true collapse
requires 1.0. The inversion therefore uses the general fit, and this was checked rather
than assumed.

**3. The scalar inversion is underdetermined.** Even with the better fit, a target `R_eff`
pins down a whole *curve* of `(N, a)` solutions rather than a point, because more short
sticks behave much like fewer long ones. Picking one point off that curve requires a
constraint from outside the model, such as a fabrication limit. Out-of-sample validation
redraws fresh networks at candidate values and confirms the curve.

**4. Thickness enters, but the scalar cannot see it.** Holding `N` fixed, `R_eff` rises
monotonically with `d` and the percolation success rate falls, across every `(N, a)`
baseline tested. This is the direct consequence of spreading the same material through a
larger volume. Since the scalar inversion never used `d` at all, adding a genuine third
parameter makes the underdetermination strictly worse.

**5. Curve shape carries information the scalar collapses away.** This is the main
contribution. Instead of a single source-to-drain number, the **onion-layer protocol**
sweeps `R_eff(reference, j)` outward through concentric distance shells, producing a
curve rather than a scalar. Grouping `(N, a, d)` triples whose *scalar* `R_eff` matches
within 15%, precisely the triples the scalar inversion cannot separate, a one-way ANOVA
per group and shell asks whether their onion curves still differ.

On the full 420-combination grid at 6 realizations per point, 39 of 214 tests are
significant at `p < 0.05`, against roughly 5% expected by chance. Re-running the five
strongest groups at 50 realizations sharpens this decisively: 35 of 40 tests significant,
31 surviving Bonferroni correction, with minimum p-values falling from about `1e-4` to as
low as `1e-48`.

The answer is yes: the shape of the onion curve distinguishes geometries that the scalar
resistance maps to the same value.

One caveat is reported honestly rather than buried. Group 16 stays weak even at 50
realizations (38% of its shells significant), which points to a genuinely marginal effect
there rather than a sampling artifact.

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
sbatch run_onion_grid_boost_array.sbatch         # 50 realizations on the strongest groups
```

Each task writes one `chunk_NN.json`. Once the full set is present, the corresponding
notebook detects it automatically and loads instead of recomputing. See
[cluster/README.md](cluster/README.md) for the details of each job.

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
can. The boosted rerun exploits this: because it reuses the same `stable_seed()` values,
its first 6 realizations per combination are byte-identical to the original data, making
it a strict extension of the same draws rather than an independent resample.

---

## Citation

```bibtex
@mastersthesis{costa2026percolation,
  author  = {Jefferson Carlos Lima da Costa},
  title   = {Effective Resistance of Random Stick-Percolation Networks},
  school  = {Centro de Inform\'atica, Universidade Federal de Pernambuco},
  year    = {2026}
}
```

## License

[MIT](LICENSE), Copyright (c) 2026 Jefferson Carlos Lima da Costa.
