from collections import deque


def connected_components(graph):
    """
    Partition ALL nodes into connected components, using the same
    shared-stick adjacency rule find_path uses internally (two nodes i, j
    are adjacent iff graph[i] and graph[j] -- each a (stick_a, stick_b)
    pair -- share a stick index) -- but with no x>0.9 "reached the drain"
    flag and no single designated start_node, since a multi-terminal network
    's terminals aren't confined to one face and there's no single
    source/drain distinction to trace.

    Returns list[list[int]], one list of node indices per component, sorted
    largest-first.
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
    visited = set()
    queue = deque([start_node])
    path_nodes = [start_node]
    path_pairs_of_lines = []
    path_pairs_of_edges = []
    path_perc = 0

    while queue:
        # print(f'valor de queue antes = {queue}')
        current_node = queue.popleft()
        # print(f'valor de queue depois = {queue}')

        if current_node not in visited:
            visited.add(current_node)
            # print(f'nós visitados {visited}')
            target_edge = graph[current_node]
            # print(f'valor de target edge = {target_edge}')

            #Appending the target edge to path_pairs_of_lines if it's not already there
            if target_edge not in path_pairs_of_lines:
                path_pairs_of_lines.append(target_edge)
                # print(f'valor de path_pairs_of_lines = {path_pairs_of_lines}')

            #Visiting each edge and appending the nodes
            for i, edge in enumerate(graph):
                if i != current_node and (target_edge[0] in edge or target_edge[1] in edge) and edge not in path_pairs_of_lines:
                    # print(f'valor do i = {i}')
                    # print(f'valor do current node = {current_node}')
                    if i not in path_nodes:
                        queue.append(i)
                        # print(f'valor da queue após append = {queue}')
                        path_nodes.append(i)
                        # print(f'valor do path_pairs_of_nodes{path_nodes}')
                        path_pairs_of_lines.append(edge)
                        # print(f'valor do path_pairs_of_lines = {path_pairs_of_lines}')
                        path_pairs_of_edges.append((current_node,i))
                        # print(f'valor do path_pairs_of_edges = {path_pairs_of_edges}')
                        if intersections_points[i][0] > 0.9:
                          path_perc = 1

    return path_nodes, path_pairs_of_lines, path_pairs_of_edges, path_perc


def verify_pruned_percolation(filtered_segment_edge_list, intersections_points,
                              source_segments, drain_segments):
    """
    Verify that the pruned path still percolates (source reachable to drain).

    Parameters
    ----------
    filtered_segment_edge_list : list of (segment, (n1, n2))
    intersections_points : list of (x, y)
    source_segments : set
    drain_segments : set

    Returns
    -------
    proceed : bool
        True if the pruned network still percolates.
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
    """
    Filter to only the connected component between source and drain.

    Parameters
    ----------
    filtered_segment_edge_list : list of (segment, (n1, n2))
    filtered_nodes : list of int
    intersections_points : list of (x, y)

    Returns
    -------
    segment_to_nodes : dict
        {index: (n1, n2)} mapping with unique edges.
    filtered_nodes : list
        Sorted list of nodes on the connected path.
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
    """
    Rebuild the graph with consecutive node indices for impedance calculation.

    Parameters
    ----------
    filtered_nodes : list of int
        Global node indices on the connected path.
    intersections_points : list of (x, y)
    segment_to_nodes : dict
        {seg_index: (global_n1, global_n2)}

    Returns
    -------
    node_mapping : dict
        {global_index: local_index}
    path_intersections : list
        Coordinate list for local nodes.
    new_segment_to_nodes : dict
        {seg_index: (local_n1, local_n2)}
    path_graph : list
        path_graph[local_i] = list of segment indices.
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

