# `utils/`: the pipeline

Plain Python modules with no packaging layer, imported by the notebooks after they add
this directory to `sys.path`. Dependencies are numpy, matplotlib, scipy, and reportlab for
the PDF reports.

A deliberate design point: the 3D generalization did **not** fork the pipeline. Only the
two genuinely geometric steps (stick generation and contact detection) have 3D siblings.
Everything downstream, path finding, pruning, edge discovery, boundary merging, and the
Laplacian solve, is the same code operating on coordinate tuples of either length. The
multi-terminal work follows the same principle.

---

## The pipeline, in order of execution

| Stage | Module | Notes |
|---|---|---|
| 1. Generate sticks | `point_generator.py` | 2D: random positions and angles in the unit square |
| | `point_generator_3d.py` | 3D: takes `box_dims` for anisotropic boxes such as a thin film `[0,1]×[0,1]×[0,d]`. Directions are sampled by normalizing a 3D standard normal, which is exactly uniform on the sphere |
| 2. Find connections | `intersection_utils.py` | 2D: exact line-segment intersections |
| | `intersection_utils_3d.py` | 3D: the contact criterion, since exact intersections almost surely do not occur between random 3D segments |
| 3. Find percolating paths | `path_analysis.py` | BFS from source to drain |
| | `boundary_utils.py` | Classifies boundary nodes as source or drain by `x` coordinate |
| 4. Clean up | `pruning.py` | Removes dead ends. Iterative node-based `prune_dead_end_nodes`, plus the legacy segment-based `prune_dangling_bonds` |
| | `self_loops.py` | Detects, removes, and (for testing) injects `(i, i)` edges |
| | `edge_discovery.py` | Consecutive edges along a stick, and dead-end segment detection |
| 5. Solve the circuit | `impedance_calculator.py` | Equivalent resistance via `L = AᵀGA`. Contains `merge_boundary_nodes()` and the `ImpedanceCalculator` class |
| | `laplacian_utils.py` | Builds the weighted Laplacian, extracts the reduced internal-node matrix, checks conditioning and singularity. Provides the pseudoinverse path used by the onion protocol |
| | `multi_terminal_solver.py` | The same Dirichlet problem with an arbitrary number of prescribed-voltage terminals rather than exactly two |

## Orchestration

| Module | Description |
|---|---|
| `pipeline_utils.py` | `run_impedance()` and `run_full_pipeline()` tie generation, pruning, self-loop checks, and the solve into one call. Also holds **`stable_seed()`**, the hashlib-based seed derivation that replaced the non-reproducible `hash()` formula |
| `pipeline_utils_3d.py` | The 3D sibling. Mirrors the 2D call sequence exactly, swapping only the two geometry steps at the front |
| `random_network_batch.py` | Batch runner for many random networks, extracted from `test_150_random_networks.ipynb` so that notebook stays a thin driver |
| `experiment_manager.py` | Creates numbered `experiments/exp_N/` folders and saves into them. Works locally and on Colab. Provides `setup_experiment()` and `load_network()` |

## Measurement

| Module | Description |
|---|---|
| `multimeter.py` | "Touch two probes to the network." `measure_r_eff(i, j)` between any two existing nodes without altering the network, unlike the boundary merge used for the standard source-to-drain calculation. Also `onion_layers()` and `run_onion_protocol()` for sweeping `R_eff` outward through concentric shells, plus `run_onion_protocol_pinv()`, the fast path that costs one pseudoinverse per network instead of one dense solve per probe |

## Visualization

| Module | Description |
|---|---|
| `network_visualization.py` | The main plotting surface: raw lines, intersections, paths before and after pruning, networks colored by resistance |
| `network_visualization_3d.py` | The 3D counterpart. Only the straight-edge pattern is ported; the 2D curved parallel-edge bowing and self-loop arcs use canvas APIs with no unique 3D analogue and are intentionally not attempted |
| `node_loop_visualization.py` | Voltage-colored panels and the three-panel baseline/looped/cleaned report for node-loop tests |
| `filtered_paths.py` | Node numbering and filtered percolation-path views |
| `plotting.py` | Standalone figure functions, output to `figures/` |

## Loop topology

| Module | Description |
|---|---|
| `self_loop_utils.py` | Add, remove, and count self-loops on the graph structures, and check percolation in their presence |
| `loop_topology_utils.py` | Attaches and removes small closed subgraphs (linear, branched, chorded) to a host node. Electrically inert, but built from real nodes and edges, so `R_eff` must not change |
| `node_loop_test_runner.py` | Runs the whole node-loop test: baseline, inject, re-measure, clean up, verify, persist, plot |
| `admittance_checks.py` | Verifies the reduced Laplacian stays invertible through injection and removal, cross-checking voltages against `ImpedanceCalculator` |

## Reporting

| Module | Description |
|---|---|
| `result_exporter.py` | Builds the export structures for nodes, edges, paths, and impedance curves |
| `pdf_generator.py` | Per-experiment and cumulative PDF reports |
| `rebuild_experiments.py` | Command-line tool to regenerate figures for existing experiments from their saved JSON. Run with `python utils/rebuild_experiments.py` |
| `region_checks.py` | Geometric tests for points lying within regions and boundaries |

## Tests

Each prints its own pass and fail lines rather than using a test framework.

```bash
cd utils
python test_merge.py
python test_multimeter.py
python test_percolation_3d.py
```

`test_percolation_3d.py` runs only its self-contained geometry, symmetry, and isotropy
checks from the command line. Its `build_*_case` functions return hand-built stick
configurations for a caller to push through the pipeline, and are exercised from
`notebooks/percolation_3d.ipynb` section 3. `test_networks.py` is likewise a fixture
module, supplying networks to the self-loop notebooks rather than running on its own.

| Module | Covers |
|---|---|
| `test_merge.py` | `merge_boundary_nodes()` on a 2×2 grid |
| `test_multimeter.py` | `measure_r_eff()` and the onion protocol against closed-form cases: a ring, a disconnected pair, a resistor ladder |
| `test_networks.py` | Hand-built 2D networks with known resistance, used by the self-loop suite |
| `test_percolation_3d.py` | The 3D suite: contact geometry, and hand-built topologies with known `R_eff` |

---

## Legacy files

Kept for provenance, not used by current work.

| File | Status |
|---|---|
| `pipeline_utils_src.py` | Earlier variant of `pipeline_utils.py` |
| `plotting_old.py` | Superseded by `network_visualization.py` |
| `utility_reset.py` | Empty placeholder |
| `nextSteps.txt` | Working notes |
| `path_pairs_of_lines vs  path_pairs_of_edges.txt` | Explains the distinction between the line-centric and node-centric views of a path. Still worth reading alongside `Paths.pdf` |
