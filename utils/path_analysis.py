from collections import deque


def connected_components(graph):
    """Group intersection nodes connected through shared stick IDs.

    Parameters
    ----------
    graph : sequence of pairs of int
        ``graph[i]`` contains the two stick IDs meeting at node i.
        This is the intersection representation, not a neighbor list.

    Returns
    -------
    components : list of list of int
        Node indices in breadth-first discovery order, grouped by component.
        Components are sorted by decreasing size, with ties retaining
        discovery order. An empty graph returns an empty list.

    Notes
    -----
    No source/drain boundary condition is imposed, so these components also
    support terminals placed away from the boundary faces.
    """
    n = len(graph)
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue
        component = [start]
        visited[start] = True
        queue = deque([start])
        while queue:
            current = queue.popleft()
            target_edge = graph[current]
            for i, edge in enumerate(graph):
                if not visited[i] and (target_edge[0] in edge or target_edge[1] in edge):
                    visited[i] = True
                    component.append(i)
                    queue.append(i)
        components.append(component)

    components.sort(key=len, reverse=True)
    return components


def find_path(intersections_points, graph, start_node):
    """Traverse shared-stick connections and record whether a drain is reached.

    Parameters
    ----------
    intersections_points : sequence of coordinate sequences, shape (n, d)
        Intersection coordinates in the unit-width film, with x first.
        Only x is used; d may be 2 or 3.
    graph : sequence of pairs of int, length n
        Stick IDs meeting at each intersection, aligned with the coordinates.
        Equal stick pairs are treated as the same entry during traversal.
    start_node : int
        Valid node index, normally chosen from the source region x < 0.1.

    Returns
    -------
    path_nodes : list of int
        Discovered nodes, starting with start_node, in traversal order.
    path_pairs_of_lines : list
        Distinct entries from graph encountered during traversal.
    path_pairs_of_edges : list of tuple of int
        Parent/child node pairs recorded on discovery. These are traversal
        edges, not all consecutive electrical segments along the sticks.
    path_perc : int
        1 if a newly discovered node has x > 0.9, otherwise 0. The starting
        node alone does not set this flag, even if it lies in the drain.

    Notes
    -----
    The input ordering determines discovery order. Coordinates and indices
    are assumed valid; an empty graph has no valid starting node.
    """
    visited = set()
    queue = deque([start_node])
    path_nodes = [start_node]
    path_pairs_of_lines = []
    path_pairs_of_edges = []
    path_perc = 0

    while queue:
        current_node = queue.popleft()

        if current_node not in visited:
            visited.add(current_node)
            target_edge = graph[current_node]

            if target_edge not in path_pairs_of_lines:
                path_pairs_of_lines.append(target_edge)

            for i, edge in enumerate(graph):
                if i != current_node and (target_edge[0] in edge or target_edge[1] in edge) and edge not in path_pairs_of_lines:
                    if i not in path_nodes:
                        queue.append(i)
                        path_nodes.append(i)
                        path_pairs_of_lines.append(edge)
                        path_pairs_of_edges.append((current_node,i))
                        if intersections_points[i][0] > 0.9:
                          path_perc = 1

    return path_nodes, path_pairs_of_lines, path_pairs_of_edges, path_perc


def verify_pruned_percolation(filtered_segment_edge_list, intersections_points,
                              source_segments, drain_segments):
    """Check for a source-to-drain connection in the retained electrical edges.

    Parameters
    ----------
    filtered_segment_edge_list : sequence of (int, (int, int))
        Retained (stick ID, endpoint-node pair) entries.
    intersections_points : sequence of coordinate sequences, shape (n, d)
        Global node coordinates in the unit-width film, with x first.
    source_segments, drain_segments : set of int
        Stick IDs touching the source and drain regions, respectively.

    Returns
    -------
    proceed : bool
        True when retained sticks include both boundary sets and a retained
        node with x < 0.1 connects to one with x > 0.9. Empty edges or a
        missing boundary return False. Nodes exactly on a threshold are
        not boundary nodes.
    """
    if not filtered_segment_edge_list:
        return False

    pruned_segments = set(seg for seg, edge in filtered_segment_edge_list)
    has_source = bool(pruned_segments & source_segments)
    has_drain = bool(pruned_segments & drain_segments)

    if not (has_source and has_drain):
        return False

    filtered_nodes = set()
    for seg, (n1, n2) in filtered_segment_edge_list:
        filtered_nodes.add(n1)
        filtered_nodes.add(n2)

    adj = {n: set() for n in filtered_nodes}
    for seg, (n1, n2) in filtered_segment_edge_list:
        adj[n1].add(n2)
        adj[n2].add(n1)

    source_nodes = [n for n in filtered_nodes if intersections_points[n][0] < 0.1]
    drain_nodes = [n for n in filtered_nodes if intersections_points[n][0] > 0.9]

    if not source_nodes or not drain_nodes:
        return False

    visited = set(source_nodes)
    queue = deque(source_nodes)
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return any(d in visited for d in drain_nodes)


def build_connected_path(filtered_segment_edge_list, filtered_nodes, intersections_points):
    """Retain components that connect both boundary regions and deduplicate edges.

    Parameters
    ----------
    filtered_segment_edge_list : sequence of (int, (int, int))
        Retained (stick ID, endpoint-node pair) entries.
    filtered_nodes : sequence of int
        Global indices of retained nodes, used to select boundary nodes.
    intersections_points : sequence of coordinate sequences, shape (n, d)
        Global coordinates in the unit-width film. Source nodes have
        x < 0.1 and drain nodes have x > 0.9; thresholds are strict.

    Returns
    -------
    segment_to_nodes : dict
        Consecutive edge IDs mapped to unique, sorted global endpoint pairs.
        Repeated pairs, including parallel edges, collapse to one pair.
    filtered_nodes : list of int
        Sorted nodes reachable from both boundary sets. More than one
        component may qualify. Returns an empty dict and list when no
        component connects the two regions.
    """
    pruned_adj = {n: {} for n in filtered_nodes}
    for seg, (n1, n2) in filtered_segment_edge_list:
        pruned_adj.setdefault(n1, {})[n2] = seg
        pruned_adj.setdefault(n2, {})[n1] = seg

    source_nodes = [n for n in filtered_nodes if intersections_points[n][0] < 0.1]
    drain_nodes = [n for n in filtered_nodes if intersections_points[n][0] > 0.9]

    # BFS from source
    reachable_from_source = set(source_nodes)
    queue = deque(source_nodes)
    while queue:
        node = queue.popleft()
        for neighbor in pruned_adj.get(node, {}):
            if neighbor not in reachable_from_source:
                reachable_from_source.add(neighbor)
                queue.append(neighbor)

    # BFS from drain
    reachable_to_drain = set(drain_nodes)
    queue = deque(drain_nodes)
    while queue:
        node = queue.popleft()
        for neighbor in pruned_adj.get(node, {}):
            if neighbor not in reachable_to_drain:
                reachable_to_drain.add(neighbor)
                queue.append(neighbor)

    connected_nodes = reachable_from_source & reachable_to_drain

    connected_edges = []
    for seg, (n1, n2) in filtered_segment_edge_list:
        if n1 in connected_nodes and n2 in connected_nodes:
            connected_edges.append((seg, (n1, n2)))

    unique_edges = sorted(set((min(n1, n2), max(n1, n2)) for seg, (n1, n2) in connected_edges))
    segment_to_nodes = {i: edge for i, edge in enumerate(unique_edges)}

    return segment_to_nodes, sorted(connected_nodes)


def rebuild_local_graph(filtered_nodes, intersections_points, segment_to_nodes):
    """Renumber retained nodes and edges for impedance calculation.

    Parameters
    ----------
    filtered_nodes : sequence of int
        Unique global node indices, including every edge endpoint.
    intersections_points : sequence of coordinate sequences, shape (n, d)
        Global node coordinates; coordinates and units are preserved.
    segment_to_nodes : dict of int to (int, int)
        Edge IDs mapped to pairs of global node indices.

    Returns
    -------
    node_mapping : dict
        Global-to-local indices assigned in sorted global-node order.
    path_intersections : list
        Coordinates in local-node order, referencing the original entries.
    new_segment_to_nodes : dict
        Consecutive edge IDs mapped to local endpoint pairs. New IDs follow
        sorted old edge IDs; dictionary insertion order follows the input.
    path_graph : list
        Incident edge IDs for each local node. A self-loop is listed twice
        at its node, once for each endpoint.

    Notes
    -----
    Empty nodes and edges produce empty outputs. Every endpoint must occur
    in filtered_nodes; a missing endpoint raises KeyError.
    """
    nodes_in_path = sorted(filtered_nodes)
    node_mapping = {old: new for new, old in enumerate(nodes_in_path)}

    path_intersections = [intersections_points[i] for i in nodes_in_path]

    old_segments = sorted(segment_to_nodes.keys())
    segment_mapping = {old: new for new, old in enumerate(old_segments)}

    new_segment_to_nodes = {}
    for old_seg, (old_n1, old_n2) in segment_to_nodes.items():
        new_seg = segment_mapping[old_seg]
        new_segment_to_nodes[new_seg] = (node_mapping[old_n1], node_mapping[old_n2])

    path_graph = [[] for _ in range(len(nodes_in_path))]
    for new_seg, (n1, n2) in new_segment_to_nodes.items():
        path_graph[n1].append(new_seg)
        path_graph[n2].append(new_seg)

    return node_mapping, path_intersections, new_segment_to_nodes, path_graph


def build_graph_from_segments(n_nodes, segment_to_nodes):
    """Build per-node incident-edge lists from an edge-to-endpoints mapping.

    Parameters
    ----------
    n_nodes : int
        Number of nodes; every endpoint must lie in ``range(n_nodes)``.
    segment_to_nodes : dict of int to (int, int)
        Edge IDs mapped to their two endpoint node indices.

    Returns
    -------
    graph : list of list of int
        ``graph[i]`` lists the IDs of the edges incident to node i, in the
        iteration order of ``segment_to_nodes``. A self-loop is listed twice
        at its node, as in :func:`rebuild_local_graph`.

    Notes
    -----
    ``ImpedanceCalculator`` takes this list as its ``graph`` argument; the
    two-point solve in ``multimeter.measure_r_eff`` does not read it.
    """
    g = [[] for _ in range(n_nodes)]
    for e, (n1, n2) in segment_to_nodes.items():
        g[n1].append(e)
        g[n2].append(e)
    return g

