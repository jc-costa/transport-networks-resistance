def remover_tupla_e_nos(filtered_path_pairs_of_lines, filtered_path_nodes, filtered_path_pairs_of_edges, dangling_segment):
    """
    Remove edges (tuples) that contain a dangling segment from the network.
    
    Parameters:
    - dangling_segment: The segment ID (line number) to remove
    
    This function removes edge pairs that contain the dangling segment.
    Since segments appear as tuples (seg1, seg2) in filtered_path_pairs_of_lines,
    we need to remove any tuple that contains the dangling segment.
    """
    # Step 1: Remove tuples containing dangling_segment from filtered_path_pairs_of_lines
    # Work backwards through paths to avoid index issues when removing items
    for i in range(len(filtered_path_pairs_of_lines) - 1, -1, -1):
        path = filtered_path_pairs_of_lines[i]
        # Remove edges that contain the dangling segment
        new_path = [edge for edge in path if dangling_segment not in edge]
        if len(new_path) < len(path):
            # Some edges were removed
            if len(new_path) == 0:
                # The entire path was removed
                filtered_path_pairs_of_lines.pop(i)
                if i < len(filtered_path_nodes):
                    filtered_path_nodes.pop(i)
                if i < len(filtered_path_pairs_of_edges):
                    filtered_path_pairs_of_edges.pop(i)
            else:
                # Some edges remain in the path
                filtered_path_pairs_of_lines[i] = new_path

    return filtered_path_pairs_of_lines, filtered_path_nodes, filtered_path_pairs_of_edges

def verify_dangling_sticks(filtered_path_pairs_of_lines, source_segments, drain_segments):
    """
    Identify dangling segments (segments that appear only once in the network).
    
    This function looks at all segment pairs in the network and identifies segments
    that appear only once, which means they are dead ends (dangling bonds).
    
    Segments at source or drain boundaries should be kept even if they appear once,
    as they provide the connection to the boundaries. Only internal dangling segments
    (not at boundaries) are considered for removal.
    
    Parameters:
    - filtered_path_pairs_of_lines: List of paths with edge pairs
    - source_segments: Set of segment IDs that touch the source boundary
    - drain_segments: Set of segment IDs that touch the drain boundary
    """
    dangling_sticks = []
    all_segments = []

    # Collect all segments from all paths
    for k in range(0, len(filtered_path_pairs_of_lines)):
        if len(filtered_path_pairs_of_lines[k]) > 0:
            edges1, edges2 = zip(*filtered_path_pairs_of_lines[k])
            all_segments.extend(edges1)
            all_segments.extend(edges2)
    
    # Find segments that appear only once (these are potential dangling bonds)
    segment_counts = {}
    for seg in all_segments:
        segment_counts[seg] = segment_counts.get(seg, 0) + 1
    
    dangling_sticks = [seg for seg, count in segment_counts.items() if count <= 1]

    print(f"  Dangling segments found: {dangling_sticks}")
    
    # Separate dangling segments into source, drain, and internal
    dangling_source = [seg for seg in dangling_sticks if seg in source_segments]
    dangling_drain = [seg for seg in dangling_sticks if seg in drain_segments]
    dangling_intern = [seg for seg in dangling_sticks if seg not in source_segments and seg not in drain_segments]
    
    print(f"  - At source boundary: {dangling_source} (keep)")
    print(f"  - At drain boundary: {dangling_drain} (keep)")
    print(f"  - Internal dangling: {dangling_intern} (remove)")
    print(f"  Total to remove: {len(dangling_intern)}")

    # Only return internal dangling segments for removal
    return dangling_intern


def prune_dangling_bonds(path_pairs_of_lines, path_nodes, path_pairs_of_edges, source_segments, drain_segments):
    """
    Prune dangling bonds from the network iteratively.
    
    Parameters:
    - path_pairs_of_lines: List of paths, where each path is a list of tuples (line1, line2)
    - path_nodes: List of nodes in each path
    - path_pairs_of_edges: List of edge pairs for each path
    - source_segments: Set of segment IDs that connect to the source boundary
    - drain_segments: Set of segment IDs that connect to the drain boundary
    
    Returns:
    - Pruned versions of the input lists
    """
    # Make copies to avoid modifying originals
    filtered_pair_of_lines = [list(path) for path in path_pairs_of_lines]
    filtered_path_nodes = [list(nodes) for nodes in path_nodes]
    filtered_pair_of_edges = [list(pairs) for pairs in path_pairs_of_edges]
    
    iteration = 0
    max_iterations = 100  # Prevent infinite loops
    while iteration < max_iterations:
        iteration += 1
        print(f"\nPruning iteration {iteration}:")
        dangling_intern = verify_dangling_sticks(filtered_pair_of_lines, source_segments, drain_segments)
        if not dangling_intern:
            print("No more dangling bonds found. Pruning complete!")
            break

        # Remove ALL dangling bonds found in this iteration
        for intern in dangling_intern:
            filtered_pair_of_lines, filtered_path_nodes, filtered_pair_of_edges = remover_tupla_e_nos(
                filtered_pair_of_lines, filtered_path_nodes, filtered_pair_of_edges, intern
            )
            print(f"Removed dangling segment {intern}")
        
        # Continue to next iteration to check if new dangling bonds were created
    
    return filtered_pair_of_lines, filtered_path_nodes, filtered_pair_of_edges


def prune_dead_end_nodes(segment_edge_list, intersections_points, verbose=True):
    """
    Remove dead-end nodes iteratively from a segment_edge_list.

    A dead-end is a node with degree 1 that is NOT at the source (x < 0.1)
    or drain (x > 0.9) boundary.

    This function is FULLY RECURSIVE with two levels:
    1. INNER LOOP: Removes all current dead-ends in each iteration
    2. OUTER LOOP: After inner loop completes, re-checks if new dead-ends
       appeared (from degree changes) and repeats the entire process
    
    This ensures that cascading dead-ends are fully removed, even in
    complex networks like Small-a random networks.

    Parameters
    ----------
    segment_edge_list : list of (segment, (n1, n2))
        Edge list from edge discovery.
    intersections_points : list
        All intersection point coordinates.
    verbose : bool
        Print progress.

    Returns
    -------
    filtered : list of (segment, (n1, n2))
        Pruned edge list.
    source_node_set : set
        Nodes with x < 0.1 in the percolating component.
    drain_node_set : set
        Nodes with x > 0.9 in the percolating component.
    """
    filtered = list(segment_edge_list)

    # Classify boundary nodes
    all_nodes = set()
    for seg, (n1, n2) in segment_edge_list:
        all_nodes.add(n1)
        all_nodes.add(n2)

    source_node_set = {n for n in all_nodes if intersections_points[n][0] < 0.1}
    drain_node_set = {n for n in all_nodes if intersections_points[n][0] > 0.9}

    outer_iteration = 0
    max_outer_iterations = 50
    total_pruning_log = []

    # ═══ OUTER LOOP: Re-run pruning until no new dead-ends appear ═══
    while outer_iteration < max_outer_iterations:
        outer_iteration += 1
        
        if verbose and outer_iteration > 1:
            print(f"\n  ⟳ Outer iteration {outer_iteration}: re-checking for new dead-ends...")

        inner_iteration = 0
        max_inner_iterations = 100
        pruning_log = []

        # ── INNER LOOP: Remove dead-ends iteratively ──────────────────────
        while inner_iteration < max_inner_iterations:
            inner_iteration += 1

            # Recompute degrees and adjacency
            node_degree = {}
            node_edges  = {}
            for seg, (n1, n2) in filtered:
                node_degree[n1] = node_degree.get(n1, 0) + 1
                node_degree[n2] = node_degree.get(n2, 0) + 1
                node_edges.setdefault(n1, []).append((seg, (n1, n2)))
                node_edges.setdefault(n2, []).append((seg, (n1, n2)))

            # Find internal dead-end nodes (degree 1, not on boundaries)
            dead_end_nodes = [
                node for node, degree in node_degree.items()
                if degree == 1
                and intersections_points[node][0] >= 0.1
                and intersections_points[node][0] <= 0.9
            ]

            # Collect edges to remove (only genuine degree-1 dead-end nodes.
            # NOTE: a segment contributing only a single edge to `filtered`
            # is NOT, by itself, a dead end -- its endpoints may still have
            # degree > 1 via other segments, in which case that edge is a
            # load-bearing bridge rather than a dangling appendage. Removing
            # edges on that basis alone can disconnect an otherwise-clean
            # percolating network, so it is intentionally not done here.)
            edges_to_remove = set()
            for node in dead_end_nodes:
                for entry in node_edges.get(node, []):
                    edges_to_remove.add(entry)

            # If nothing to remove, break
            if not dead_end_nodes:
                if inner_iteration == 1:
                    if verbose and outer_iteration == 1:
                        print("  ✓ No dead-end nodes found. Network is already clean.")
                break

            if verbose:
                node_coords = [
                    f"{n} ({intersections_points[n][0]:.3f},{intersections_points[n][1]:.3f})"
                    for n in sorted(dead_end_nodes)[:5]
                ]
                print(f"  Iteration {outer_iteration}.{inner_iteration}: removing "
                      f"{len(dead_end_nodes)} dead-end node(s) = "
                      f"{len(edges_to_remove)} edge(s)")
                if node_coords:
                    print(f"    → Dead-end nodes: [{', '.join(node_coords)}...]")

            pruning_log.append({
                'outer': outer_iteration,
                'inner': inner_iteration,
                'removed_nodes': sorted(dead_end_nodes),
                'removed_edges': list(edges_to_remove),
            })

            # Remove the dead-end edges
            filtered = [
                (seg, edge) for seg, edge in filtered
                if (seg, edge) not in edges_to_remove
            ]

            if verbose:
                print(f"    → Remaining edges: {len(filtered)}")

        total_pruning_log.extend(pruning_log)

        # ── Check if we need another outer iteration ──────────────────────
        # Recompute node degrees
        node_degree_final = {}
        for seg, (n1, n2) in filtered:
            node_degree_final[n1] = node_degree_final.get(n1, 0) + 1
            node_degree_final[n2] = node_degree_final.get(n2, 0) + 1

        remaining_dead_ends = [
            node for node, degree in node_degree_final.items()
            if degree == 1
            and intersections_points[node][0] >= 0.1
            and intersections_points[node][0] <= 0.9
        ]

        if not remaining_dead_ends:
            # All dead-ends removed - we're done!
            break
        else:
            # New dead-ends appeared, need another outer iteration
            if verbose:
                print(f"    ⚠ Found {len(remaining_dead_ends)} new dead-end node(s), "
                      f"continuing outer loop...")

    # ── Final summary ─────────────────────────────────────────────────────
    if total_pruning_log:
        total_removed = sum(len(log['removed_nodes']) for log in total_pruning_log)
        total_edges_removed = sum(len(log['removed_edges']) for log in total_pruning_log)
        if verbose:
            print(f"\n  ✓ Pruning complete: {total_removed} node(s) and "
                  f"{total_edges_removed} edge(s) removed across {outer_iteration} outer iteration(s).")

        # Final verification
        node_degree_final = {}
        for seg, (n1, n2) in filtered:
            node_degree_final[n1] = node_degree_final.get(n1, 0) + 1
            node_degree_final[n2] = node_degree_final.get(n2, 0) + 1

        remaining_dead_ends = [
            node for node, degree in node_degree_final.items()
            if degree == 1
            and intersections_points[node][0] >= 0.1
            and intersections_points[node][0] <= 0.9
        ]

        if remaining_dead_ends and verbose:
            print(f"  ⚠ WARNING: {len(remaining_dead_ends)} dead-end node(s) still remain!")
            print(f"    This may indicate the algorithm reached max_outer_iterations ({max_outer_iterations})")
        elif verbose:
            print(f"  ✓ Verified: No internal dead-ends remain.")

    return filtered, source_node_set, drain_node_set


def prune_dead_end_nodes_generic(segment_edge_list, protected_nodes, verbose=True):
    """
    Generalization of prune_dead_end_nodes for an arbitrary number of
    terminal groups (not just source x<0.1 / drain x>0.9): same
    inner/outer iterative removal of degree-1 nodes, except a node is
    protected from pruning whenever it's in `protected_nodes` (the union of
    ALL terminal groups), instead of prune_dead_end_nodes' internal
    `intersections_points[node][0]` threshold check. Coordinates aren't
    needed at all here, since "is this node a terminal" is now an explicit
    set membership test rather than something derived from position.

    Parameters
    ----------
    segment_edge_list : list of (segment, (n1, n2))
    protected_nodes : set[int]
        Union of all terminal groups' node indices -- never pruned
        regardless of degree.
    verbose : bool

    Returns
    -------
    filtered : list of (segment, (n1, n2))
        Pruned edge list.
    """
    filtered = list(segment_edge_list)

    outer_iteration = 0
    max_outer_iterations = 50
    total_removed_nodes = 0
    total_removed_edges = 0

    while outer_iteration < max_outer_iterations:
        outer_iteration += 1

        if verbose and outer_iteration > 1:
            print(f"\n  ⟳ Outer iteration {outer_iteration}: re-checking for new dead-ends...")

        inner_iteration = 0
        max_inner_iterations = 100

        while inner_iteration < max_inner_iterations:
            inner_iteration += 1

            node_degree = {}
            node_edges = {}
            for seg, (n1, n2) in filtered:
                node_degree[n1] = node_degree.get(n1, 0) + 1
                node_degree[n2] = node_degree.get(n2, 0) + 1
                node_edges.setdefault(n1, []).append((seg, (n1, n2)))
                node_edges.setdefault(n2, []).append((seg, (n1, n2)))

            dead_end_nodes = [
                node for node, degree in node_degree.items()
                if degree == 1 and node not in protected_nodes
            ]

            edges_to_remove = set()
            for node in dead_end_nodes:
                for entry in node_edges.get(node, []):
                    edges_to_remove.add(entry)

            if not dead_end_nodes:
                if inner_iteration == 1 and verbose and outer_iteration == 1:
                    print("  ✓ No dead-end nodes found. Network is already clean.")
                break

            if verbose:
                print(f"  Iteration {outer_iteration}.{inner_iteration}: removing "
                      f"{len(dead_end_nodes)} dead-end node(s) = {len(edges_to_remove)} edge(s)")

            total_removed_nodes += len(dead_end_nodes)
            total_removed_edges += len(edges_to_remove)

            filtered = [
                (seg, edge) for seg, edge in filtered
                if (seg, edge) not in edges_to_remove
            ]

        node_degree_final = {}
        for seg, (n1, n2) in filtered:
            node_degree_final[n1] = node_degree_final.get(n1, 0) + 1
            node_degree_final[n2] = node_degree_final.get(n2, 0) + 1

        remaining_dead_ends = [
            node for node, degree in node_degree_final.items()
            if degree == 1 and node not in protected_nodes
        ]

        if not remaining_dead_ends:
            break
        elif verbose:
            print(f"    ⚠ Found {len(remaining_dead_ends)} new dead-end node(s), "
                  f"continuing outer loop...")

    if verbose and total_removed_nodes:
        print(f"\n  ✓ Pruning complete: {total_removed_nodes} node(s) and "
              f"{total_removed_edges} edge(s) removed across {outer_iteration} outer iteration(s).")

    return filtered


def build_pruned_structures(segment_edge_list, filtered_segment_edge_list):
    """
    Build summary data structures after pruning.

    Returns
    -------
    pruned_segments : set
        Segments remaining after pruning.
    filtered_nodes : list
        Sorted list of unique node indices.
    filtered_pair_of_edges_flat : list
        Unique (n1, n2) edge pairs.
    stats : dict
        Before/after counts and removed elements.
    """
    pruned_segments = set(seg for seg, edge in filtered_segment_edge_list)
    filtered_pair_of_edges_flat = list(
        set(edge for seg, edge in filtered_segment_edge_list)
    )

    filtered_nodes = set()
    for seg, (n1, n2) in filtered_segment_edge_list:
        filtered_nodes.add(n1)
        filtered_nodes.add(n2)
    filtered_nodes = sorted(filtered_nodes)

    edges_before = set((min(n1, n2), max(n1, n2)) for seg, (n1, n2) in segment_edge_list)
    edges_after = set((min(n1, n2), max(n1, n2)) for seg, (n1, n2) in filtered_segment_edge_list)
    removed_edges = edges_before - edges_after

    nodes_before = set()
    for seg, (n1, n2) in segment_edge_list:
        nodes_before.add(n1)
        nodes_before.add(n2)
    nodes_after = set(filtered_nodes)
    removed_nodes = nodes_before - nodes_after

    segments_before = set(seg for seg, edge in segment_edge_list)
    segments_removed = segments_before - pruned_segments

    stats = {
        'nodes_before': nodes_before,
        'nodes_after': nodes_after,
        'removed_nodes': removed_nodes,
        'edges_before': edges_before,
        'edges_after': edges_after,
        'removed_edges': removed_edges,
        'segments_before': segments_before,
        'segments_after': pruned_segments,
        'segments_removed': segments_removed,
    }

    return pruned_segments, filtered_nodes, filtered_pair_of_edges_flat, stats
