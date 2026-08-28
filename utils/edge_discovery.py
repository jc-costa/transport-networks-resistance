"""
Edge Discovery Module

Discovers all edges in the percolating component using consecutive-only
connectivity: for each stick, sort percolating nodes by projection along
the stick direction and connect only consecutive pairs.
"""
from collections import defaultdict, Counter
import numpy as np


def discover_consecutive_edges(graph, intersections_points, perc_node_set, points_1, points_2):
    """
    Discover all edges in the percolating component using consecutive-only logic.

    For each stick, collect all percolating nodes on it, sort by projection
    along the stick direction, and connect only consecutive pairs.

    Parameters
    ----------
    graph : list
        graph[i] = list of stick indices that node i lies on.
    intersections_points : list
        List of coordinate tuples (2D or 3D) for all intersection nodes.
    perc_node_set : set
        Set of node indices belonging to the percolating component.
    points_1 : list
        Start points of each stick [(x, y[, z]), ...].
    points_2 : list
        End points of each stick [(x, y[, z]), ...].

    Returns
    -------
    segment_edge_list : list of (stick, (n1, n2))
        Each entry is (stick_index, (node_a, node_b)) for consecutive nodes.
    """
    stick_to_perc_nodes = defaultdict(set)
    for node in perc_node_set:
        for stick in graph[node]:
            stick_to_perc_nodes[stick].add(node)

    segment_edge_list = []
    seen_edges = set()

    for stick, nodes_on_stick in stick_to_perc_nodes.items():
        if len(nodes_on_stick) < 2:
            continue
        direction = np.asarray(points_2[stick]) - np.asarray(points_1[stick])

        def projection(n, _direction=direction):
            return np.dot(np.asarray(intersections_points[n]), _direction)

        sorted_on_stick = sorted(nodes_on_stick, key=projection)
        for idx in range(len(sorted_on_stick) - 1):
            n1, n2 = sorted_on_stick[idx], sorted_on_stick[idx + 1]
            edge_key = (min(n1, n2), max(n1, n2))
            dedup_key = (stick, edge_key)
            if dedup_key not in seen_edges:
                seen_edges.add(dedup_key)
                segment_edge_list.append((stick, (n1, n2)))

    return segment_edge_list


def check_dead_ends(segment_edge_list, source_segments, drain_segments):
    """
    Check for internal dangling segments (dead-ends) in the edge list.

    A segment is dangling if at least one of its endpoint nodes has
    degree 1 (i.e. that node has no other connection into the network)
    and the segment does not touch the source or drain boundary.

    Note: a segment that contributes only a single edge to
    `segment_edge_list` is NOT, by itself, dangling -- its endpoints may
    still have degree > 1 via other segments, in which case that edge is
    a load-bearing bridge rather than a dead end. Flagging it purely
    because the stick "appears once" (the previous behaviour of this
    function) can misclassify bridge edges as dangling and, if pruned,
    disconnect an otherwise-clean percolating network.

    Parameters
    ----------
    segment_edge_list : list of (stick, (n1, n2))
    source_segments : set of int
    drain_segments : set of int

    Returns
    -------
    dangling_intern : list of int
        Internal dangling segment indices.
    has_dead_ends : bool
    """
    node_degree = Counter()
    for seg, (n1, n2) in segment_edge_list:
        node_degree[n1] += 1
        node_degree[n2] += 1

    dangling_intern = sorted({
        seg for seg, (n1, n2) in segment_edge_list
        if (node_degree[n1] == 1 or node_degree[n2] == 1)
        and seg not in source_segments and seg not in drain_segments
    })

    return dangling_intern, len(dangling_intern) > 0
