"""
Pipeline utilities for network impedance calculations.
Provides run_full_pipeline (complete workflow) and run_impedance (convenience wrapper).
"""

import numpy as np

def run_impedance(intersections, graph, segment_to_nodes, k=1.0, label=""):
    """
    Convenience wrapper: create ImpedanceCalculator, run, and print summary.
    Returns the full result dict from ImpedanceCalculator.calculate().
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
    """
    Run the complete pipeline on a given network.
    Returns a dict with all intermediate results for inspection.
    """
    from point_generator import calculate_angles
    from intersection_utils import identify_intersections
    from path_analysis import find_path, build_connected_path, rebuild_local_graph
    from pruning import (
        prune_dead_end_nodes, build_pruned_structures, verify_pruned_percolation
    )
    from edge_discovery import discover_consecutive_edges, check_dead_ends
    from impedance_calculator import (
        ImpedanceCalculator, merge_boundary_nodes, build_incidence_matrix, build_laplacian
    )

    results = {"title": title, "N": N, "a": a, "k": k}

    intersections_points = []
    intersected_line_1 = []
    intersected_line_2 = []
    graph = []
    intersections_points, intersected_line_1, intersected_line_2, graph = \
        identify_intersections(
            points_1, points_2,
            intersections_points, intersected_line_1, intersected_line_2,
            graph, angles, a, N
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
