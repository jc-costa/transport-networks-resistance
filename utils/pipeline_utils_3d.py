"""
3D sibling of pipeline_utils.py. Mirrors run_full_pipeline/
generate_percolating_network's call sequence exactly, swapping only the two
3D-specific geometry steps (generate_points_3d, identify_contacts_3d) at the
front. Everything downstream -- find_path, pruning, edge discovery, boundary
merging, the Laplacian/impedance solve -- is the SAME code used by the 2D
pipeline (utils/path_analysis.py, utils/pruning.py, utils/edge_discovery.py,
utils/impedance_calculator.py): none of it is duplicated here, since it was
already dimension-agnostic (or made so) and works unchanged on 3-tuples.

`distance_dependent_impedance` (in pipeline_utils.py) is likewise reused
as-is: it only reads results["merged_intersections"]/["merged_graph"]/
["merged_segment_to_nodes"]/["edge_conductance"], which run_full_pipeline_3d
populates in the same shape as the 2D pipeline.
"""

import numpy as np


def run_full_pipeline_3d(points_1, points_2, N, a, contact_distance,
                         title="Network 3D", k=1000, verbose=True):
    """
    Run the complete 3D pipeline on a given network of sticks.
    Returns a dict with all intermediate results for inspection -- same
    shape as run_full_pipeline's return value, plus a "contact_gaps" entry.
    """
    from intersection_utils_3d import identify_contacts_3d
    from path_analysis import find_path, build_connected_path, rebuild_local_graph
    from pruning import prune_dead_end_nodes, build_pruned_structures
    from path_analysis import verify_pruned_percolation
    from edge_discovery import discover_consecutive_edges, check_dead_ends
    from impedance_calculator import (
        ImpedanceCalculator, merge_boundary_nodes, build_incidence_matrix, build_laplacian
    )

    results = {"title": title, "N": N, "n_sticks": len(points_1), "a": a,
               "contact_distance": contact_distance, "k": k}

    intersections_points, graph, contact_gaps = identify_contacts_3d(
        points_1, points_2, contact_distance
    )
    results["intersections_points"] = intersections_points
    results["graph"] = graph
    results["contact_gaps"] = contact_gaps

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  Sticks: {N},  a = {a:.4f},  contact_distance = {contact_distance:.4f}")
        print(f"  Contacts found: {len(intersections_points)}")

    if len(intersections_points) == 0:
        print("  No contacts. Skipping.")
        results["has_percolation"] = False
        return results

    boundary_condition_source = [i for i, pt in enumerate(intersections_points) if pt[0] < 0.1]
    boundary_condition_drain = [i for i, pt in enumerate(intersections_points) if pt[0] > 0.9]

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
        # callers can inspect what merge_boundary_nodes is about to collapse:
        # any edge whose two endpoints are BOTH in source_boundary_new (or
        # BOTH in drain_boundary_new) becomes a self-loop after merging and is
        # silently dropped inside merge_boundary_nodes (never appears in
        # merged_segment_to_nodes).
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


def identify_self_loop_edges(res):
    """
    Which pre-merge edges (from res["pre_merge_segment_to_nodes"]) will
    become self-loops once merge_boundary_nodes collapses every source-region
    node into one and every drain-region node into another -- i.e. edges
    whose two endpoints are both in pre_merge_source_nodes, or both in
    pre_merge_drain_nodes. These are dropped silently inside
    merge_boundary_nodes (`if new_n1 == new_n2: continue`) and never appear
    in merged_segment_to_nodes.

    Returns a list of (edge_idx, (n1, n2)) pre-merge edges that are
    self-loop candidates.
    """
    source_set = set(res["pre_merge_source_nodes"])
    drain_set = set(res["pre_merge_drain_nodes"])
    self_loops = []
    for e, (n1, n2) in res["pre_merge_segment_to_nodes"].items():
        if (n1 in source_set and n2 in source_set) or (n1 in drain_set and n2 in drain_set):
            self_loops.append((e, (n1, n2)))
    return self_loops


def run_full_pipeline_3d_multi_terminal_faces(points_1, points_2, N, a, contact_distance,
                                              n_subdivisions_per_face=4,
                                              terminal_voltages=None,
                                              drop_empty_terminals=True,
                                              title="Network 3D (multi-terminal faces)",
                                              k=1000, verbose=True):
    """
    Same x<0.1/x>0.9 boundary convention as run_full_pipeline_3d, but each
    face is subdivided into several independent terminals (by (y, z)
    quadrant, see multi_terminal_solver.split_face_into_quadrants) instead
    of being merged into a single source/drain node -- each terminal gets
    its own prescribed voltage.

    Reuses run_full_pipeline_3d ENTIRELY unchanged (including its
    pre_merge_* fields, added precisely so this function doesn't need to
    duplicate contact-detection/percolation/pruning/rebuild logic) -- only
    the final grouping/merge/solve differs, replacing merge_boundary_nodes
    + ImpedanceCalculator with merge_terminal_groups + solve_multi_terminal.

    Returns a dict mirroring run_full_pipeline_3d's shape, plus:
      terminal_groups, terminal_node_indices, terminal_voltages,
      merged_intersections, merged_segment_to_nodes, edge_conductance,
      solve (the full solve_multi_terminal result dict)
    """
    from impedance_calculator import merge_terminal_groups
    from multi_terminal_solver import split_face_into_quadrants, solve_multi_terminal

    base = run_full_pipeline_3d(points_1, points_2, N, a, contact_distance,
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

    source_groups = split_face_into_quadrants(source_nodes, intersections_points,
                                              n_subdivisions=n_subdivisions_per_face)
    drain_groups = split_face_into_quadrants(drain_nodes, intersections_points,
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
        # non-empty-quadrant counts) assign the source's voltage to a
        # drain-face quadrant, making it electrically act as a second source
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
              f"({n_source_nonempty}/{n_subdivisions_per_face} source quadrants non-empty, "
              f"{n_drain_nonempty}/{n_subdivisions_per_face} drain quadrants non-empty)")
        print(f"  Terminal voltages: {terminal_voltages}")
        print(f"  Current imbalance: {solve['current_imbalance']:.3e} (should be ~0)")

    return results


def run_full_pipeline_3d_spherical_terminals(points_1, points_2, N, a, contact_distance,
                                             terminal_centers, terminal_radii, terminal_voltages,
                                             title="Network 3D (spherical terminals)",
                                             k=1000, verbose=True):
    """
    Terminals as small spherical regions placed ANYWHERE in the unit cube
    (not confined to the x<0.1/x>0.9 faces) -- so the x-threshold-hardwired
    machinery in find_path/prune_dead_end_nodes/build_connected_path can't
    be reused. Uses the generic siblings instead:
      identify_contacts_3d (unchanged) -> assign_nodes_to_spherical_terminals
      -> connected_components (generic, no x-threshold) -> pick the
      component common to all terminals -> discover_consecutive_edges
      (already generic, just needs a node set) -> prune_dead_end_nodes_generic
      (protects any terminal node, not an x-range) -> dedupe into a clean
      segment_to_nodes (the same idiom build_connected_path uses internally)
      -> rebuild_local_graph (unchanged, purely topological) ->
      merge_terminal_groups -> solve_multi_terminal.

    Returns a dict with the same shape as
    run_full_pipeline_3d_multi_terminal_faces's return value.
    """
    from intersection_utils_3d import identify_contacts_3d
    from path_analysis import connected_components, rebuild_local_graph
    from pruning import prune_dead_end_nodes_generic, build_pruned_structures
    from edge_discovery import discover_consecutive_edges
    from impedance_calculator import merge_terminal_groups
    from multi_terminal_solver import (
        assign_nodes_to_spherical_terminals, validate_multi_terminal_percolation, solve_multi_terminal,
    )

    results = {"title": title, "N": N, "a": a, "contact_distance": contact_distance, "k": k,
              "terminal_centers": terminal_centers, "terminal_radii": terminal_radii}

    intersections_points, graph, contact_gaps = identify_contacts_3d(points_1, points_2, contact_distance)
    results["intersections_points"] = intersections_points
    results["graph"] = graph
    results["contact_gaps"] = contact_gaps

    if verbose:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
        print(f"  Sticks: {N},  a = {a:.4f},  contact_distance = {contact_distance:.4f}")
        print(f"  Contacts found: {len(intersections_points)}")

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

    # Dedupe raw (stick, (n1,n2)) edges into a clean segment_to_nodes dict --
    # same idiom path_analysis.build_connected_path uses internally, minus
    # the "reachable from source AND to drain" filtering (already handled
    # by restricting to the chosen connected component above).
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


def generate_percolating_network_3d(N, a, contact_distance, box_dims=(1.0, 1.0, 1.0),
                                    k_uniform=1000.0, max_attempts=200, seed=None, boundary="clip"):
    """
    Draw random 3D stick networks (N sticks, length a, contact_distance),
    confined to `box_dims` (unit cube by default; pass e.g. (1.0, 1.0, d)
    for a film of thickness d), via run_full_pipeline_3d until one
    percolates, or give up after max_attempts. Mirrors
    generate_percolating_network exactly.

    Source/drain are still the x<0.1 / x>0.9 faces (run_full_pipeline_3d),
    so this only makes physical sense for box_dims[0] == 1.0 -- shrinking
    box_dims[2] (thickness) leaves the transport direction (x) and its
    boundary convention untouched.

    Returns (results, attempts) where results is the run_full_pipeline_3d
    dict for the percolating network (or None if none percolated within
    max_attempts), and attempts is how many draws it took.
    """
    import io
    import contextlib
    from point_generator_3d import generate_points_3d

    if seed is not None:
        np.random.seed(seed)

    for attempt in range(1, max_attempts + 1):
        points_1, points_2, directions = [], [], []
        try:
            generate_points_3d(N, points_1, points_2, directions, a, box_dims=box_dims,
                               enforce_bounds=True, boundary=boundary)
        except RuntimeError:
            # A single stick occasionally can't find in-bounds endpoints within
            # generate_points_3d's own per-segment attempt budget (rare tail
            # event, more common as a approaches the box diagonal or box_dims
            # gets thin) -- treat it as one more failed draw, not a crash.
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_full_pipeline_3d(points_1, points_2, N, a, contact_distance,
                                           k=k_uniform, verbose=False)
        if results.get("has_percolation") and "merged_segment_to_nodes" in results:
            results["points_1"] = points_1
            results["points_2"] = points_2
            results["directions"] = directions
            results["box_dims"] = box_dims
            results["boundary"] = boundary
            return results, attempt

    return None, max_attempts
