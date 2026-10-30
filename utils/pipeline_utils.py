"""
Pipeline utilities for network impedance calculations.
Provides run_full_pipeline (complete workflow) and run_impedance (convenience wrapper).
"""

import hashlib
import numpy as np


def stable_seed(*parts, modulus=100000):
    """
    Deterministic, process-stable seed from (N, a, d, ...)-style parts.

    Python's built-in hash() randomizes str/bytes hashing per-process by
    default (PYTHONHASHSEED) -- fine for a tuple of pure numbers (int/float
    hashing is NOT randomized), but as soon as a tag string like 'ext' joins
    the tuple, hash((N, a, d, 'ext')) gives a DIFFERENT value every time a
    new Python process starts, even for identical inputs. That silently
    broke reproducibility for every sweep that seeded networks with a
    string-tagged hash() call (the 3D (N, a, d) grid and its cluster
    scripts) while leaving the string-free 2D (N, a) grid unaffected.

    This hashes the string form of `parts` through hashlib (stable across
    interpreters/processes/PYTHONHASHSEED settings) instead of relying on
    Python's built-in hash() for anything but pure numeric tuples.

    Parameters
    ----------
    *parts : object
        Values identifying one sweep point, joined as ``str(p)`` with "_".
        Floats should be rounded by the caller (e.g. ``round(a, 3)``), since
        ``str(0.30000000000000004)`` and ``str(0.3)`` give different seeds.
    modulus : int, optional
        Upper bound (exclusive) of the returned seed.

    Returns
    -------
    int
        SHA-256 of the joined string, reduced modulo `modulus`; a valid
        argument to ``np.random.seed``.
    """
    key = "_".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest, 16) % modulus


def run_impedance(intersections, graph, segment_to_nodes, k=1.0, label=""):
    """Solve a network with uniform edge resistance and print a short report.

    Parameters
    ----------
    intersections : sequence of coordinate sequences, shape (n, d)
        Node positions; source and drain are the nodes of minimum and
        maximum x, as chosen by ``ImpedanceCalculator``.
    graph : list of list of int
        Incident edge IDs per node.
    segment_to_nodes : dict of int to (int, int)
        Edge IDs mapped to endpoint pairs. Self-loops are allowed.
    k : float, optional
        Resistance of every edge, in ohms.
    label : str, optional
        Tag printed above the report.

    Returns
    -------
    dict
        The result of ``ImpedanceCalculator.calculate(k=k)``.

    Notes
    -----
    The report recomputes the reduced matrix with
    ``laplacian_utils.compute_reduced_matrix`` to print its condition number,
    and flags it singular above 1e10, the threshold at which
    ``ImpedanceCalculator`` falls back to least squares.
    """
    from impedance_calculator import ImpedanceCalculator
    from laplacian_utils import compute_reduced_matrix
    from self_loop_utils import count_self_loops, check_percolation

    calc = ImpedanceCalculator(intersections, graph, segment_to_nodes)
    result = calc.calculate(k=k, verbose=False)

    src = calc.source_node
    drn = calc.drain_node

    L, M, cond, is_sing = compute_reduced_matrix(
        intersections, segment_to_nodes, src, drn, k=k
    )
    n_loops = count_self_loops(segment_to_nodes)
    perc = check_percolation(segment_to_nodes, src, drn)

    tag = f"  [{label}]" if label else ""
    print(f"{tag}")
    print(f"    Nodes: {len(intersections)},  Edges: {len(segment_to_nodes)}"
          f"  (self-loops: {n_loops})")
    print(f"    Source node: {src}  (x={intersections[src][0]:.3f})")
    print(f"    Drain  node: {drn}  (x={intersections[drn][0]:.3f})")
    print(f"    Percolation: {'YES' if perc else 'NO'}")
    if M.size > 0:
        print(f"    Condition number of M: {cond:.3e}  "
              f"  Singular: {'YES (fallback LSQ)' if is_sing else 'no'}")
    print(f"    Impedance R = {result['impedance']:.6f} Ohm")
    return result


def run_full_pipeline(points_1, points_2, angles, N, a,
                      title="Network", k=1000, verbose=True, save=False):
    """Run the 2D two-terminal pipeline on one stick configuration.

    Steps: exact stick intersections, source (x < 0.1) and drain (x > 0.9)
    nodes, ``find_path`` from every source node, consecutive edges along the
    percolating paths, dead-end pruning, a source-to-drain connectivity check,
    the connected source-drain component, local renumbering, the boundary
    merge, and a uniform-resistance solve.

    Parameters
    ----------
    points_1, points_2 : list of (float, float)
        Stick endpoints in the unit square.
    angles : list of float
        Stick orientations in radians, aligned with the endpoints.
    N : int
        Density parameter of the draw. Under the ``clip`` boundary rule the
        number of sticks can differ from N; loops use ``len(points_1)``.
    a : float
        Stick length, also used by ``identify_intersections``.
    title : str, optional
        Label for printed output.
    k : float, optional
        Uniform resistance per edge, in ohms.
    verbose : bool, optional
        Print progress. Some early-exit messages print regardless.
    save : bool, optional
        Unused; kept for call-site compatibility.

    Returns
    -------
    results : dict
        Always holds ``has_percolation`` and the intersection data. When the
        pruned network still connects source and drain, it also holds the
        ``pre_merge_*`` network (renumbered, before the boundary merge), the
        ``merged_*`` network with ``edge_conductance`` multipliers, and
        ``impedance`` with the full ``calc_result``.

    Notes
    -----
    The merged terminals sit at the mean position of their face nodes. For
    the distance-dependent model use ``distance_dependent_impedance``, which
    measures edge lengths on the pre-merge geometry.
    """
    from intersection_utils import identify_intersections
    from path_analysis import find_path, build_connected_path, rebuild_local_graph
    from pruning import prune_dead_end_nodes, build_pruned_structures
    from path_analysis import verify_pruned_percolation
    from edge_discovery import discover_consecutive_edges, check_dead_ends
    from impedance_calculator import ImpedanceCalculator, merge_boundary_nodes

    # N is the density parameter of the draw (stick centres per unit area);
    # the number of sticks actually present may differ from it under the
    # clipping boundary rule, so every loop below runs over the lists.
    n_sticks = len(points_1)
    results = {"title": title, "N": N, "n_sticks": n_sticks, "a": a, "k": k}

    intersections_points = []
    intersected_line_1 = []
    intersected_line_2 = []
    graph = []
    intersections_points, intersected_line_1, intersected_line_2, graph = \
        identify_intersections(
            points_1, points_2,
            intersections_points, intersected_line_1, intersected_line_2,
            graph, angles, a, n_sticks
        )
    results["intersections_points"] = intersections_points
    results["graph"] = graph

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  Sticks: {N},  a = {a:.4f}")
        print(f"  Intersections found: {len(intersections_points)}")

    if len(intersections_points) == 0:
        print("  No intersections. Skipping.")
        results["has_percolation"] = False
        return results

    boundary_condition_source = [i for i, (x, y) in enumerate(intersections_points) if x < 0.1]
    boundary_condition_drain = [i for i, (x, y) in enumerate(intersections_points) if x > 0.9]

    if verbose:
        print(f"  Source nodes (x<0.1): {len(boundary_condition_source)}")
        print(f"  Drain  nodes (x>0.9): {len(boundary_condition_drain)}")

    if not boundary_condition_source or not boundary_condition_drain:
        print("  Boundary conditions not met. No percolation possible.")
        results["has_percolation"] = False
        return results

    source_segments = set()
    drain_segments = set()
    for node_idx in boundary_condition_source:
        source_segments.update(graph[node_idx])
    for node_idx in boundary_condition_drain:
        drain_segments.update(graph[node_idx])
    results["source_segments"] = source_segments
    results["drain_segments"] = drain_segments

    path_nodes = []
    path_pairs_of_lines = []
    path_pairs_of_edges = []
    all_path_perc = []

    for i in boundary_condition_source:
        already_covered = any(i in path_nodes[k_] for k_ in range(len(path_nodes)))
        if not already_covered:
            nodes, pol, poe, pp = find_path(intersections_points, graph, i)
            path_nodes.append(nodes)
            path_pairs_of_lines.append(pol)
            path_pairs_of_edges.append(poe)
            all_path_perc.append(pp)

    results["all_path_perc"] = all_path_perc
    results["path_nodes"] = path_nodes
    has_percolation = any(p == 1 for p in all_path_perc)
    results["has_percolation"] = has_percolation

    if not has_percolation:
        print("  No percolating path found.")
        return results

    if verbose:
        n_perc = sum(1 for p in all_path_perc if p == 1)
        print(f"  Percolating paths: {n_perc}")

    perc_indices = [i for i, p in enumerate(all_path_perc) if p == 1]
    perc_path_nodes = [path_nodes[i] for i in perc_indices]
    all_perc_nodes_set = set(n for lst in perc_path_nodes for n in lst)

    segment_edge_list = discover_consecutive_edges(
        graph, intersections_points, all_perc_nodes_set, points_1, points_2
    )
    results["segment_edge_list"] = segment_edge_list

    dangling_intern, has_dead_ends = check_dead_ends(
        segment_edge_list, source_segments, drain_segments
    )
    results["has_dead_ends"] = has_dead_ends
    results["dangling_intern"] = dangling_intern

    if verbose:
        print(f"  Edges before pruning: {len(segment_edge_list)}")

    filtered_segment_edge_list, source_node_set, drain_node_set = \
        prune_dead_end_nodes(segment_edge_list, intersections_points, verbose=verbose)
    results["filtered_segment_edge_list"] = filtered_segment_edge_list

    pruned_segments, filtered_nodes, filtered_pair_of_edges_flat, pruning_stats = \
        build_pruned_structures(segment_edge_list, filtered_segment_edge_list)
    results["pruning_stats"] = pruning_stats
    results["filtered_nodes"] = filtered_nodes

    if verbose:
        print(f"  Edges after  pruning: {len(filtered_segment_edge_list)}")

    proceed_to_impedance = verify_pruned_percolation(
        filtered_segment_edge_list, intersections_points,
        source_segments, drain_segments
    )
    results["proceed_to_impedance"] = proceed_to_impedance

    if proceed_to_impedance and len(filtered_segment_edge_list) > 0:
        segment_to_nodes, filtered_nodes = build_connected_path(
            filtered_segment_edge_list, filtered_nodes, intersections_points
        )
        node_mapping, path_intersections, new_segment_to_nodes, path_graph = \
            rebuild_local_graph(filtered_nodes, intersections_points, segment_to_nodes)

        source_nodes_pruned = [n for n in filtered_nodes if intersections_points[n][0] < 0.1]
        drain_nodes_pruned = [n for n in filtered_nodes if intersections_points[n][0] > 0.9]
        source_boundary_new = [node_mapping[n] for n in source_nodes_pruned if n in node_mapping]
        drain_boundary_new = [node_mapping[n] for n in drain_nodes_pruned if n in node_mapping]

        # Pre-merge (but post-pruning/rebuild) structures -- kept around so
        # callers can inspect what merge_boundary_nodes is about to collapse,
        # and so multi-terminal pipelines (subdivided faces, terminals
        # anywhere) can reuse contact-detection/percolation/pruning/rebuild
        # without duplicating it. Mirrors pipeline_utils_3d.py's
        # run_full_pipeline_3d exactly.
        results["pre_merge_intersections"] = path_intersections
        results["pre_merge_segment_to_nodes"] = new_segment_to_nodes
        results["pre_merge_source_nodes"] = source_boundary_new
        results["pre_merge_drain_nodes"] = drain_boundary_new

        (merged_intersections, merged_graph, merged_segment_to_nodes,
         merged_source_node, merged_drain_node, merge_mapping, edge_conductance) = \
            merge_boundary_nodes(
                path_intersections, path_graph, new_segment_to_nodes,
                source_boundary_new, drain_boundary_new
            )

        results["merged_intersections"] = merged_intersections
        results["merged_graph"] = merged_graph
        results["merged_segment_to_nodes"] = merged_segment_to_nodes
        results["merged_source_node"] = merged_source_node
        results["merged_drain_node"] = merged_drain_node
        results["edge_conductance"] = edge_conductance

        calc = ImpedanceCalculator(
            merged_intersections, merged_graph, merged_segment_to_nodes,
            source_boundary_nodes=source_boundary_new,
            drain_boundary_nodes=drain_boundary_new,
            use_virtual_nodes=False,
            edge_conductance=edge_conductance,
        )
        result = calc.calculate(k=k, verbose=False)
        results["impedance"] = result["impedance"]
        results["calc_result"] = result

        if verbose:
            print(f"\n  >>> Impedance = {result['impedance']:.4f} Ohm <<<")

    return results


def run_full_pipeline_multi_terminal_faces(points_1, points_2, angles, N, a,
                                           n_subdivisions_per_face=2,
                                           terminal_voltages=None,
                                           drop_empty_terminals=True,
                                           title="Network (multi-terminal faces)",
                                           k=1000, verbose=True):
    """
    2D sibling of pipeline_utils_3d.py's run_full_pipeline_3d_multi_terminal_faces.
    Same x<0.1/x>0.9 boundary convention as run_full_pipeline, but each face
    (a set of nodes that only varies in y, since 2D has no z axis) is
    subdivided into several independent terminals via
    multi_terminal_solver.split_face_into_segments -- the 2D analog of
    split_face_into_quadrants -- instead of being merged into a single
    source/drain node. Each terminal gets its own prescribed voltage.

    Reuses run_full_pipeline ENTIRELY unchanged (including its pre_merge_*
    fields, added precisely so this function doesn't need to duplicate
    intersection-detection/percolation/pruning/rebuild logic) -- only the
    final grouping/merge/solve differs, replacing merge_boundary_nodes +
    ImpedanceCalculator with merge_terminal_groups + solve_multi_terminal.

    Returns a dict mirroring run_full_pipeline's shape, plus:
      terminal_groups, terminal_node_indices, terminal_voltages,
      merged_intersections, merged_segment_to_nodes, edge_conductance,
      solve (the full solve_multi_terminal result dict)
    """
    from impedance_calculator import merge_terminal_groups
    from multi_terminal_solver import split_face_into_segments, solve_multi_terminal

    base = run_full_pipeline(points_1, points_2, angles, N, a,
                             title=title, k=k, verbose=verbose)
    results = dict(base)
    results["n_subdivisions_per_face"] = n_subdivisions_per_face

    if not base.get("has_percolation") or "pre_merge_segment_to_nodes" not in base:
        results["has_percolation"] = False
        return results

    intersections_points = base["pre_merge_intersections"]
    segment_to_nodes = base["pre_merge_segment_to_nodes"]
    source_nodes = base["pre_merge_source_nodes"]
    drain_nodes = base["pre_merge_drain_nodes"]

    source_groups = split_face_into_segments(source_nodes, intersections_points,
                                             n_subdivisions=n_subdivisions_per_face)
    drain_groups = split_face_into_segments(drain_nodes, intersections_points,
                                            n_subdivisions=n_subdivisions_per_face)
    n_source_nonempty = sum(1 for g in source_groups if g)
    n_drain_nonempty = sum(1 for g in drain_groups if g)
    all_groups = source_groups + drain_groups

    if drop_empty_terminals:
        all_groups = [g for g in all_groups if g]
    elif any(len(g) == 0 for g in all_groups):
        results["has_percolation"] = False
        results["multi_terminal_error"] = "empty terminal group (pass drop_empty_terminals=True to allow this)"
        return results

    if len(all_groups) < 2:
        results["has_percolation"] = False
        results["multi_terminal_error"] = f"only {len(all_groups)} non-empty terminal group(s), need >= 2"
        return results

    if terminal_voltages is None:
        # All source-face terminals at the same (high) voltage, all drain-face
        # terminals at the same (low) voltage -- NOT alternating by list
        # position, which would (whenever the two faces don't have matching
        # non-empty-segment counts) assign the source's voltage to a
        # drain-face segment, making it electrically act as a second source
        # even though it's geometrically on the drain face.
        terminal_voltages = [1.0] * n_source_nonempty + [0.0] * n_drain_nonempty
    elif len(terminal_voltages) != len(all_groups):
        raise ValueError(f"terminal_voltages length ({len(terminal_voltages)}) must match the number of "
                         f"non-empty terminal groups ({len(all_groups)})")

    (merged_intersections, merged_graph, merged_segment_to_nodes,
     terminal_node_indices, node_mapping, edge_conductance) = merge_terminal_groups(
        intersections_points, [], segment_to_nodes, all_groups
    )

    solve = solve_multi_terminal(merged_intersections, merged_segment_to_nodes,
                                 terminal_node_indices, terminal_voltages, k=k)

    results["terminal_groups"] = all_groups
    results["terminal_node_indices"] = terminal_node_indices
    results["terminal_voltages"] = terminal_voltages
    results["merged_intersections"] = merged_intersections
    results["merged_graph"] = merged_graph
    results["merged_segment_to_nodes"] = merged_segment_to_nodes
    results["edge_conductance"] = edge_conductance
    results["solve"] = solve

    if verbose:
        print(f"\n  Multi-terminal faces: {len(all_groups)} terminals "
              f"({n_source_nonempty}/{n_subdivisions_per_face} source segments non-empty, "
              f"{n_drain_nonempty}/{n_subdivisions_per_face} drain segments non-empty)")
        print(f"  Terminal voltages: {terminal_voltages}")
        print(f"  Current imbalance: {solve['current_imbalance']:.3e} (should be ~0)")

    return results


def run_full_pipeline_circular_terminals(points_1, points_2, angles, N, a,
                                         terminal_centers, terminal_radii, terminal_voltages,
                                         title="Network (circular terminals)",
                                         k=1000, verbose=True):
    """
    2D sibling of pipeline_utils_3d.py's run_full_pipeline_3d_spherical_terminals:
    terminals as small circular regions placed ANYWHERE in the unit square
    (not confined to the x<0.1/x>0.9 faces), so the x-threshold-hardwired
    machinery in find_path/prune_dead_end_nodes/build_connected_path can't be
    reused. Uses the same generic siblings the 3D version uses (they were
    already dimension-agnostic, no changes needed):
      identify_intersections -> multi_terminal_solver.
      assign_nodes_to_spherical_terminals (plain Euclidean distance, works in
      any dimension despite the name) -> path_analysis.connected_components
      -> pick the component common to all terminals ->
      discover_consecutive_edges -> pruning.prune_dead_end_nodes_generic
      (protects any terminal node) -> dedupe into a clean segment_to_nodes
      -> rebuild_local_graph -> merge_terminal_groups -> solve_multi_terminal.

    Returns a dict with the same shape as
    run_full_pipeline_multi_terminal_faces's return value.
    """
    from intersection_utils import identify_intersections
    from path_analysis import connected_components, rebuild_local_graph
    from pruning import prune_dead_end_nodes_generic
    from edge_discovery import discover_consecutive_edges
    from impedance_calculator import merge_terminal_groups
    from multi_terminal_solver import (
        assign_nodes_to_spherical_terminals, validate_multi_terminal_percolation, solve_multi_terminal,
    )

    results = {"title": title, "N": N, "a": a, "k": k,
              "terminal_centers": terminal_centers, "terminal_radii": terminal_radii}

    intersections_points, intersected_line_1, intersected_line_2, graph = \
        identify_intersections(
            points_1, points_2, [], [], [], [], angles, a, N
        )
    results["intersections_points"] = intersections_points
    results["graph"] = graph

    if verbose:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
        print(f"  Sticks: {N},  a = {a:.4f}")
        print(f"  Intersections found: {len(intersections_points)}")

    if len(intersections_points) == 0:
        results["has_percolation"] = False
        return results

    terminal_groups = assign_nodes_to_spherical_terminals(intersections_points, terminal_centers, terminal_radii)
    components = connected_components(graph)
    node_to_component = {n: c_idx for c_idx, comp in enumerate(components) for n in comp}

    if not validate_multi_terminal_percolation(terminal_groups, node_to_component):
        results["has_percolation"] = False
        return results

    # Which single component contains at least one node from every terminal
    # group (validated above to exist) -- take it as the working network.
    common_components = set.intersection(*[
        {node_to_component[n] for n in g if n in node_to_component} for g in terminal_groups
    ])
    chosen_component = next(iter(common_components))
    component_node_set = set(components[chosen_component])

    if verbose:
        print(f"  Terminal groups (sizes): {[len(g) for g in terminal_groups]}")
        print(f"  Chosen connected component: {len(component_node_set)} nodes")

    results["has_percolation"] = True
    results["terminal_groups_raw"] = terminal_groups

    segment_edge_list = discover_consecutive_edges(
        graph, intersections_points, component_node_set, points_1, points_2
    )
    results["segment_edge_list"] = segment_edge_list

    protected_nodes = set(n for g in terminal_groups for n in g)
    filtered_segment_edge_list = prune_dead_end_nodes_generic(
        segment_edge_list, protected_nodes, verbose=verbose
    )
    results["filtered_segment_edge_list"] = filtered_segment_edge_list

    # Re-validate: pruning can still isolate a terminal even if it had
    # contacts in the chosen component before pruning.
    terminal_groups_pruned = [
        [n for n in g if any(n in edge for _, edge in filtered_segment_edge_list)]
        for g in terminal_groups
    ]
    surviving_nodes = set(n for _, edge in filtered_segment_edge_list for n in edge)
    node_to_component_pruned = {n: 0 for n in surviving_nodes}  # single component by construction
    if not validate_multi_terminal_percolation(terminal_groups_pruned, node_to_component_pruned):
        results["has_percolation"] = False
        results["multi_terminal_error"] = "a terminal was isolated by dead-end pruning"
        return results

    unique_edges = sorted(set((min(n1, n2), max(n1, n2)) for _, (n1, n2) in filtered_segment_edge_list))
    segment_to_nodes = {i: edge for i, edge in enumerate(unique_edges)}
    filtered_nodes = sorted(set(n for edge in unique_edges for n in edge))

    node_mapping, path_intersections, new_segment_to_nodes, path_graph = rebuild_local_graph(
        filtered_nodes, intersections_points, segment_to_nodes
    )

    terminal_groups_local = [
        [node_mapping[n] for n in g if n in node_mapping] for g in terminal_groups_pruned
    ]

    (merged_intersections, merged_graph, merged_segment_to_nodes,
     terminal_node_indices, merge_mapping, edge_conductance) = merge_terminal_groups(
        path_intersections, path_graph, new_segment_to_nodes, terminal_groups_local
    )

    solve = solve_multi_terminal(merged_intersections, merged_segment_to_nodes,
                                 terminal_node_indices, terminal_voltages, k=k)

    results["terminal_groups"] = terminal_groups_local
    results["terminal_node_indices"] = terminal_node_indices
    results["terminal_voltages"] = terminal_voltages
    results["merged_intersections"] = merged_intersections
    results["merged_graph"] = merged_graph
    results["merged_segment_to_nodes"] = merged_segment_to_nodes
    results["edge_conductance"] = edge_conductance
    results["solve"] = solve

    if verbose:
        print(f"\n  Terminals: {len(terminal_node_indices)}   Voltages: {terminal_voltages}")
        print(f"  Current imbalance: {solve['current_imbalance']:.3e} (should be ~0)")

    return results


def generate_percolating_network(N, a, k_uniform=1000.0, max_attempts=200, seed=None,
                                 boundary="clip"):
    """Draw random 2D stick networks until one percolates.

    Parameters
    ----------
    N : int
        Stick density (see ``point_generator.generate_points``).
    a : float
        Stick length.
    k_uniform : float, optional
        Uniform edge resistance for the solve inside ``run_full_pipeline``.
    max_attempts : int, optional
        Number of draws before giving up.
    seed : int, optional
        If given, ``np.random.seed(seed)`` is called once before the first
        draw; otherwise the global NumPy generator continues from its state.
    boundary : {"clip", "reject"}, optional
        Boundary rule passed to ``generate_points``.

    Returns
    -------
    results : dict or None
        The ``run_full_pipeline`` result of the first percolating draw, with
        ``points_1``, ``points_2``, ``angles`` and ``boundary`` added, or
        None if no draw percolated.
    attempts : int
        Draws used, counting the successful one; ``max_attempts`` on failure.

    Notes
    -----
    All randomness comes from the global ``np.random`` state, so the result
    depends on every earlier draw made with it in the same process.
    """
    import io
    import contextlib
    from point_generator import generate_points

    if seed is not None:
        np.random.seed(seed)

    for attempt in range(1, max_attempts + 1):
        points_1, points_2, angles = [], [], []
        generate_points(N, points_1, points_2, angles, a, enforce_bounds=True, boundary=boundary)
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_full_pipeline(points_1, points_2, angles, N, a,
                                        k=k_uniform, verbose=False)
        if results.get("has_percolation") and "merged_segment_to_nodes" in results:
            results["points_1"] = points_1
            results["points_2"] = points_2
            results["angles"] = angles
            results["boundary"] = boundary
            return results, attempt

    return None, max_attempts


def distance_dependent_impedance(results, R0, use_merged_geometry=False):
    """
    Recompute the impedance of a run_full_pipeline() result using the
    distance-dependent resistance model (R_edge = R0 * L_edge) instead of the
    uniform-k model already stored in results['impedance'].

    Edge lengths are measured on the PRE-MERGE geometry. merge_boundary_nodes
    parks each merged terminal at the average position of the face it replaces,
    so measuring lengths on the merged coordinates would re-measure every edge
    that touches a terminal, changing the resistance of wire segments that never
    moved. Shorting a face is node identification: the face nodes are tied
    together by an ideal conductor and the wires keep their real lengths, which
    is what merged_edge_true_lengths recovers (including the correct
    parallel combination for collapsed edges). The difference is not academic:
    on (N=40, a=0.6) percolating networks the centroid geometry overstates the
    whole-face R_eff by 2-17% depending on the realization.

    Parameters
    ----------
    results : dict
        A run_full_pipeline / run_full_pipeline_3d result.
    R0 : float
        Resistance per unit length.
    use_merged_geometry : bool
        Set True to reproduce the older (physically incorrect) numbers that
        measured edge lengths on the merged centroid coordinates. Only useful
        for comparing against results recorded before this was fixed.
    """
    from impedance_calculator import (ImpedanceCalculator, build_merge_node_mapping,
                                      merged_edge_true_lengths)

    calc = ImpedanceCalculator(
        results["merged_intersections"], results["merged_graph"],
        results["merged_segment_to_nodes"], edge_conductance=results["edge_conductance"],
    )

    edge_lengths = None
    if not use_merged_geometry:
        if "pre_merge_segment_to_nodes" not in results:
            raise KeyError(
                "results has no pre_merge_* fields, so the true edge lengths cannot be "
                "recovered; pass use_merged_geometry=True to accept the merged centroid "
                "geometry instead")
        node_mapping, _, _, _ = build_merge_node_mapping(
            len(results["pre_merge_intersections"]),
            results["pre_merge_source_nodes"], results["pre_merge_drain_nodes"])
        edge_lengths = merged_edge_true_lengths(
            results["pre_merge_intersections"], results["pre_merge_segment_to_nodes"],
            node_mapping, results["merged_segment_to_nodes"],
            edge_conductance=results["edge_conductance"])

    return calc.calculate(k=R0, distance_dependent=True, edge_lengths=edge_lengths)
