# Notebooks

Every analysis in this project lives in a notebook. The `utils/` package holds the
pipeline; these notebooks pose the questions, run them, and record the answers.

All notebooks run unchanged both locally and on Google Colab. The path-setup cell in each
one resolves `../utils` relative to the notebook for local use; on Colab, uncomment the
Colab `utils_path` line and comment out the local one.

> **Before running anything.** Jupyter and Colab always start with an empty runtime, but
> the outputs saved in the file remain visible from the previous run. Running a cell in
> the middle of a notebook without running the ones above it raises `NameError` for
> variables that merely look like they already exist, and the stale plot below the cell
> does not change because the cell crashed before drawing a new one. Use
> **Runtime > Run all** first, then edit and re-run individual cells.
>
> If a cell that worked earlier suddenly raises `TypeError: got an unexpected keyword
> argument`, the code in `utils/` changed while the session was live and Python is still
> holding the old import. Re-run the imports cell, which force-reloads the changed
> modules, then re-run the failing cell.

---

## The main line of the thesis

Read these in order. They build on one another.

| # | Notebook | What it establishes |
|---|---|---|
| 1 | `calculating_the_impedance_of_a_graph.ipynb` | The incidence-matrix method for equivalent resistance, derived from scratch |
| 2 | `exemplo_impedance.ipynb` | A fully worked example, building the Laplacian by hand and verifying the result |
| 3 | `distance_dependent_resistance.ipynb` | Defines and validates the `R_edge = R0 · L_edge` model used everywhere afterwards, against hand-computable cases and six real random networks |
| 4 | `test_random_network_impedance.ipynb` | The complete 2D workflow: random network, percolation, pruning, impedance |
| 5 | `effective_resistance_vs_N_a.ipynb` | The forward problem in 2D. Part A is a dense 4×3 grid, Part B an adaptive grid sized for the cluster |
| 6 | `inverse_N_a_from_R_eff.ipynb` | The inverse problem. Fits the scaling, tests the `N·a²` collapse rather than assuming it, traces the solution curve for a target, and validates on fresh networks |
| 7 | `percolation_3d.ipynb` | The 3D generalization via a contact-distance bonding criterion, with the full validation suite and a 2D cross-check |
| 8 | `effective_resistance_vs_d.ipynb` | Film thickness as a third parameter. Parts A and B are local, Part C is the 420-combination cluster grid |
| 9 | `multimeter_r_eff.ipynb` | The probe protocol: `R_eff(i, j)` between arbitrary nodes, and the onion-layer sweep |
| 10 | **`r_eff_ij_vs_N_a_d.ipynb`** | **The main result.** Does onion-curve shape disambiguate geometries the scalar `R_eff` cannot separate? |

Results and figures from these are written up in [`../docs/RESULTS.md`](../docs/RESULTS.md).

---

## Multi-terminal extensions

Generalizing from one source and one drain to several independently biased terminals.

| Notebook | Description |
|---|---|
| `percolation_3d_multi_terminal_faces.ipynb` | Each source and drain face subdivided into several independent terminals by `(y, z)` quadrant, each at its own prescribed voltage |
| `percolation_3d_spherical_terminals.ipynb` | Terminals as small spherical regions placed anywhere in the cube, not confined to the `x` faces |
| `percolation_2d_multi_terminal.ipynb` | The same multi-terminal capability ported back down to 2D. Most of the machinery was already dimension-agnostic, so only two pieces needed 2D siblings |

---

## Validation and test suites

These exist to make the main results trustworthy. Each targets one way the pipeline could
be silently wrong.

| Notebook | What it checks |
|---|---|
| `test_pruning_examples.ipynb` | Pruning against manual examples, augmented networks, and small-`a` random networks |
| `test_pruning_with_dangling_bonds.ipynb` | Removal of dangling nodes and bonds |
| `test_simple_percolation.ipynb` | Basic percolation and connectivity |
| `test_self_loops.ipynb` | Injecting and removing a self-loop on hand-built networks must leave `R_eff` unchanged |
| `test_new_self_loops_random_networks.ipynb` | The same invariance through the full random-network pipeline |
| `test_self_loops_node_specific.ipynb` | Self-loops on specific chosen nodes, with the affected node highlighted |
| `test_self_loops_multi_node.ipynb` | Electrically inert node-loops (linear, branched, chorded subgraphs) attached to a host node, verified before, during, and after cleanup |
| `test_150_random_networks.ipynb` | 150 random networks through the full pipeline, for batch statistics |
| `test_paths_in_subregions.ipynb` | Path detection inside specific subregions |
| `testing_points_in_subregions.ipynb` | Point location and geometric region detection |
| `testing_data_structures.ipynb` | The node and edge data structures |
| `node_edge_mapping_analysis.ipynb` | Structures mapping nodes to edges and back |
| `node_edge_visualization_with_ids.ipynb` | Visualizations annotated with edge and node IDs |
| `PiaNo.ipynb` | The original working notebook: network generation and path analysis |

`test_self_loops_random_networks.ipynb` is an empty stub, superseded by
`test_new_self_loops_random_networks.ipynb`. It is kept only so links to it from older
notes do not break.

---

## Notebooks with a cluster part

Several notebooks are split so the cheap version runs locally and the full version runs as
a SLURM job, reloading committed results instead of recomputing. The pattern is a
`LOCAL_PREVIEW` flag plus a cache check against `cluster/*_results/`.

| Notebook | Cluster section | Job |
|---|---|---|
| `effective_resistance_vs_N_a.ipynb` | Part B | `run_effective_resistance_sweep.sbatch` |
| `effective_resistance_vs_d.ipynb` | Part C | `run_effective_resistance_d_array.sbatch` |
| `inverse_N_a_from_R_eff.ipynb` | Section 6 | `run_validation_array.sbatch` |
| `r_eff_ij_vs_N_a_d.ipynb` | Parts B, C | `run_onion_grid_array.sbatch` (or `run_onion_grid_parallel.sbatch`); Part D needs the retired boost run and is skipped (`RUN_BOOST=False`) |
| `test_150_random_networks.ipynb` | Section 4 | `run_random_networks_array.sbatch` |

See [`../cluster/README.md`](../cluster/README.md).
