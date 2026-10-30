"""
Impedance Calculator Module

Calculates the equivalent resistance (impedance) of a percolation path 
using the incidence matrix method as described in the book (Figure 2.7).

Flow:
1. Build incidence matrix A (edges x nodes)
2. Laplacian L = A^T * A
3. Reduce matrix M (remove source/drain)
4. Boundary vector b = -L_ib * V_boundary
5. Solve V = M^(-1) * b
6. Currents I = -A * V
7. Impedance R = k / I_total
"""

import numpy as np


def _infer_dim(intersections, default=2):
    """Number of coordinates per node (2 for 2D, 3 for 3D), inferred from the
    first available point so boundary/virtual-node placeholders generalize
    without hardcoding a dimensionality."""
    for point in intersections:
        return len(point)
    return default


def build_merge_node_mapping(n_nodes, source_boundary_nodes, drain_boundary_nodes):
    """
    The old-index -> new-index renumbering that merge_boundary_nodes applies:
    internal nodes first (ascending), then the single merged source node, then
    the single merged drain node.

    Factored out of merge_boundary_nodes so that code needing to relate a merged
    network back to the network it came from (e.g.
    merged_edge_true_lengths) can rebuild the mapping instead of
    guessing it, with no risk of the two drifting apart.

    Returns
    -------
    node_mapping : dict  old node index -> new node index
    merged_source_idx, merged_drain_idx : int
    internal_nodes : list of int  (old indices, ascending)
    """
    source_set = set(source_boundary_nodes)
    drain_set = set(drain_boundary_nodes)
    internal_nodes = [i for i in range(n_nodes) if i not in source_set and i not in drain_set]

    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(internal_nodes)}

    merged_source_idx = len(internal_nodes)
    for old_idx in source_boundary_nodes:
        node_mapping[old_idx] = merged_source_idx

    merged_drain_idx = len(internal_nodes) + 1
    for old_idx in drain_boundary_nodes:
        node_mapping[old_idx] = merged_drain_idx

    return node_mapping, merged_source_idx, merged_drain_idx, internal_nodes


def merged_edge_true_lengths(pre_intersections, pre_segment_to_nodes, node_mapping,
                             merged_segment_to_nodes, edge_conductance=None):
    """
    Effective length of every edge of a MERGED network, measured on the
    geometry the network actually had before the merge.

    Why this exists: merge_boundary_nodes puts each merged terminal at the
    AVERAGE position of the face it replaces. Under the uniform model that is
    harmless (resistance does not depend on position at all), but under the
    distance-dependent model R_edge = R0 * L_edge it silently rewrites the
    resistance of every edge touching a terminal -- wire segments that never
    moved would be re-measured to the centroid. Shorting a face is node
    IDENTIFICATION: the nodes are tied together by an ideal conductor, and the
    wires keep their real lengths. This function recovers those real lengths.

    Parallel edges are handled the way parallel resistors actually combine.
    merge_boundary_nodes collapses the m edges that land on the same merged node
    pair into ONE edge with conductance multiplier m, and ImpedanceCalculator
    then uses G = m / (R0 * L). For that to equal the true parallel conductance
    sum(1 / (R0 * L_i)), the effective length must be the harmonic-style
    combination L_eff = m / sum(1 / L_i), which is what is returned here (it
    reduces to L itself when m == 1).

    Parameters
    ----------
    pre_intersections : list
        Node coordinates BEFORE the merge (2D or 3D).
    pre_segment_to_nodes : dict
        edge_idx -> (node_a, node_b) before the merge.
    node_mapping : dict
        old node index -> merged node index (see build_merge_node_mapping).
    merged_segment_to_nodes : dict
        edge_idx -> (node_a, node_b) after the merge.
    edge_conductance : dict, optional
        Merged edge -> parallel multiplier, as returned by merge_boundary_nodes.
        Only used to sanity-check that the recovered grouping matches.

    Returns
    -------
    numpy.ndarray of length len(merged_segment_to_nodes): the effective length
    of each merged edge (inf for an edge whose originals all had zero length,
    which ImpedanceCalculator already treats as carrying no current).
    """
    pre_inter = np.asarray(pre_intersections, dtype=float)

    # merged endpoint pair -> merged edge index
    pair_to_edge = {}
    for e, (n1, n2) in merged_segment_to_nodes.items():
        pair_to_edge[(min(n1, n2), max(n1, n2))] = e

    inv_length_sum = {e: 0.0 for e in merged_segment_to_nodes}
    counts = {e: 0 for e in merged_segment_to_nodes}

    for n1, n2 in pre_segment_to_nodes.values():
        if n1 == n2:
            continue
        m1, m2 = node_mapping[n1], node_mapping[n2]
        if m1 == m2:
            continue  # both endpoints landed on the same terminal: shorted out
        e = pair_to_edge.get((min(m1, m2), max(m1, m2)))
        if e is None:
            continue
        counts[e] += 1
        length = float(np.linalg.norm(pre_inter[n1] - pre_inter[n2]))
        if length > 0:
            inv_length_sum[e] += 1.0 / length

    if edge_conductance:
        for e, multiplier in edge_conductance.items():
            if counts.get(e, 0) != multiplier:
                raise ValueError(
                    f"merged edge {e} groups {counts.get(e, 0)} pre-merge edge(s) but its "
                    f"conductance multiplier is {multiplier}; the merged network and the "
                    f"pre-merge network passed in do not correspond")

    lengths = np.full(len(merged_segment_to_nodes), np.inf)
    for e in merged_segment_to_nodes:
        if inv_length_sum[e] > 0:
            lengths[e] = counts[e] / inv_length_sum[e]
    return lengths


def merge_boundary_nodes(intersections, graph, segment_to_nodes,
                         source_boundary_nodes, drain_boundary_nodes):
    """
    Merge all source boundary nodes into ONE physical node, and all drain 
    boundary nodes into ONE physical node.
    
    This modifies the actual network topology (not virtual nodes).
    Parallel edges between merged nodes are preserved and counted.
    
    Parameters:
    -----------
    intersections : list
        List of coordinates for each node (2D `[x, y]` or 3D `[x, y, z]`)
    graph : list
        List where graph[i] = segments at node i
    segment_to_nodes : dict
        Mapping from segment index to (node_a, node_b) tuple
    source_boundary_nodes : list
        List of node indices in source region (will be merged into one)
    drain_boundary_nodes : list
        List of node indices in drain region (will be merged into one)
    
    Returns:
    --------
    new_intersections : list
        Updated list of node positions
    new_graph : list
        Updated graph (segments at each node)
    new_segment_to_nodes : dict
        Updated segment-to-nodes mapping
    source_node : int
        Index of the merged source node
    drain_node : int
        Index of the merged drain node
    node_mapping : dict
        Mapping from old node indices to new node indices
    edge_conductance : dict
        Multiplier for each edge (number of parallel resistors)
    """
    node_mapping, merged_source_idx, merged_drain_idx, internal_nodes = \
        build_merge_node_mapping(len(intersections), source_boundary_nodes,
                                 drain_boundary_nodes)

    # Build new intersections list
    new_intersections = []
    
    # Internal nodes keep their positions
    for old_idx in internal_nodes:
        new_intersections.append(list(intersections[old_idx]))
    
    dim = _infer_dim(intersections)

    # Merged source: average position of source boundary nodes
    if source_boundary_nodes:
        src_pos = np.mean([intersections[i] for i in source_boundary_nodes], axis=0)
        new_intersections.append(src_pos.tolist())
    else:
        new_intersections.append([0.0] + [0.5] * (dim - 1))

    # Merged drain: average position of drain boundary nodes
    if drain_boundary_nodes:
        drn_pos = np.mean([intersections[i] for i in drain_boundary_nodes], axis=0)
        new_intersections.append(drn_pos.tolist())
    else:
        new_intersections.append([1.0] + [0.5] * (dim - 1))
    
    # Count parallel edges between each pair of nodes
    # Multiple edges between same pair = parallel resistors
    edge_counts = {}  # (n1, n2) -> count
    
    for old_edge, (old_n1, old_n2) in segment_to_nodes.items():
        new_n1 = node_mapping[old_n1]
        new_n2 = node_mapping[old_n2]
        
        # Skip self-loops (both nodes merged to same point)
        if new_n1 == new_n2:
            continue
        
        # Normalize edge direction
        edge_key = (min(new_n1, new_n2), max(new_n1, new_n2))
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
    
    # Create segment_to_nodes with conductance multipliers
    new_segment_to_nodes = {}
    edge_conductance = {}  # edge_idx -> number of parallel wires
    
    new_edge_idx = 0
    for edge_key, count in edge_counts.items():
        n1, n2 = edge_key
        new_segment_to_nodes[new_edge_idx] = (n1, n2)
        edge_conductance[new_edge_idx] = count
        new_edge_idx += 1
    
    # Build new graph (segments at each node)
    n_new_nodes = len(new_intersections)
    new_graph = [[] for _ in range(n_new_nodes)]
    
    for edge_idx, (n1, n2) in new_segment_to_nodes.items():
        new_graph[n1].append(edge_idx)
        new_graph[n2].append(edge_idx)
    
    return (new_intersections, new_graph, new_segment_to_nodes,
            merged_source_idx, merged_drain_idx, node_mapping, edge_conductance)


def merge_terminal_groups(intersections, graph, segment_to_nodes, terminal_groups):
    """
    Generalizes merge_boundary_nodes from exactly 2 groups (source, drain) to
    an arbitrary number of terminal groups -- each group of node indices is
    merged into ONE new node (position = mean of the group), for a
    multi-terminal network with more than one independent source/sink.

    Same parallel-edge counting / edge_conductance-multiplier logic as
    merge_boundary_nodes, just iterating over `terminal_groups` instead of
    two hardcoded groups. merge_boundary_nodes itself is left untouched;
    this is a separate function so existing callers/behavior can't be
    affected by this generalization.

    Parameters
    ----------
    intersections : list
        List of coordinates for each node (2D or 3D).
    graph : list
        List where graph[i] = segments at node i.
    segment_to_nodes : dict
        Mapping from segment index to (node_a, node_b) tuple.
    terminal_groups : list of list of int
        Each inner list is one terminal's node indices, merged into one new
        node. Groups must be pairwise disjoint.

    Returns
    -------
    new_intersections : list
    new_graph : list
    new_segment_to_nodes : dict
    terminal_node_indices : list of int
        One merged node index per input group, same order as terminal_groups.
    node_mapping : dict
        Mapping from old node indices to new node indices.
    edge_conductance : dict
        Multiplier for each edge (number of parallel resistors).
    """
    all_terminal_nodes = set()
    for group in terminal_groups:
        group_set = set(group)
        if group_set & all_terminal_nodes:
            raise ValueError("terminal_groups must be pairwise disjoint")
        all_terminal_nodes |= group_set

    n_nodes = len(intersections)
    internal_nodes = [i for i in range(n_nodes) if i not in all_terminal_nodes]

    node_mapping = {}
    for new_idx, old_idx in enumerate(internal_nodes):
        node_mapping[old_idx] = new_idx

    terminal_node_indices = []
    for g_idx, group in enumerate(terminal_groups):
        if not group:
            raise ValueError(f"terminal group {g_idx} is empty -- cannot merge "
                             "(check validity/percolation upstream before merging)")
        merged_idx = len(internal_nodes) + g_idx
        terminal_node_indices.append(merged_idx)
        for old_idx in group:
            node_mapping[old_idx] = merged_idx

    new_intersections = []
    for old_idx in internal_nodes:
        new_intersections.append(list(intersections[old_idx]))
    for group in terminal_groups:
        pos = np.mean([intersections[i] for i in group], axis=0)
        new_intersections.append(pos.tolist())

    # Count parallel edges between each pair of new nodes (same idiom as
    # merge_boundary_nodes: multiple edges between the same pair collapse
    # into one edge with a conductance multiplier)
    edge_counts = {}
    for old_edge, (old_n1, old_n2) in segment_to_nodes.items():
        new_n1 = node_mapping[old_n1]
        new_n2 = node_mapping[old_n2]
        if new_n1 == new_n2:
            continue
        edge_key = (min(new_n1, new_n2), max(new_n1, new_n2))
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

    new_segment_to_nodes = {}
    edge_conductance = {}
    new_edge_idx = 0
    for edge_key, count in edge_counts.items():
        n1, n2 = edge_key
        new_segment_to_nodes[new_edge_idx] = (n1, n2)
        edge_conductance[new_edge_idx] = count
        new_edge_idx += 1

    n_new_nodes = len(new_intersections)
    new_graph = [[] for _ in range(n_new_nodes)]
    for edge_idx, (n1, n2) in new_segment_to_nodes.items():
        new_graph[n1].append(edge_idx)
        new_graph[n2].append(edge_idx)

    return (new_intersections, new_graph, new_segment_to_nodes,
            terminal_node_indices, node_mapping, edge_conductance)


def build_incidence_matrix(n_nodes, n_edges, segment_to_nodes):
    """
    Build the incidence matrix A (edges x nodes).
    
    A[e, n1] = -1 and A[e, n2] = +1 for edge e = (n1, n2); a self-loop
    (n1 == n2) leaves row e at zero.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the graph
    n_edges : int
        Number of edges in the graph
    segment_to_nodes : dict
        Mapping from segment index to (node_from, node_to) tuple
    
    Returns:
    --------
    A : numpy.ndarray
        Incidence matrix of shape (n_edges, n_nodes). Edge IDs index rows,
        so they must lie in range(n_edges).
    """
    A = np.zeros((n_edges, n_nodes))
    
    for edge, (n1, n2) in segment_to_nodes.items():
        if n1 == n2:
            # Self-loop: no electrical effect (both terminals are the same node,
            # so voltage difference is zero and no current flows). Leave the
            # row as zeros so the segment contributes nothing to the Laplacian
            # and current calculations while remaining in segment_to_nodes for
            # visualisation.
            continue
        A[edge, n1] = -1  # Edge starts from n1
        A[edge, n2] = +1  # Edge ends at n2
    
    return A


def build_laplacian(A):
    """
    Build the Laplacian matrix L = A^T * A.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Incidence matrix
    
    Returns:
    --------
    L : numpy.ndarray
        Laplacian matrix
    """
    return A.T @ A


class ImpedanceCalculator:
    """
    Calculator for equivalent resistance of a percolation path.
    
    Uses the incidence matrix method following the book's approach.
    Supports virtual source/drain nodes for proper parallel path handling.

    Method: with incidence matrix A and edge conductances G, the weighted
    Laplacian is L = A^T diag(G) A. The source is held at V = 1 and the
    drain at V = 0; the internal voltages solve M V_int = -L_ib V_b, where M
    is the internal block of L. The impedance is 1 / I_total, with I_total
    the net current into the drain.

    Assumptions and numerical caveats:

    - Without virtual nodes, source and drain are the nodes of minimum and
      maximum x (first index on ties); after ``merge_boundary_nodes`` these
      are the merged terminals. Callers may overwrite ``source_node`` and
      ``drain_node`` before ``calculate`` (as ``multimeter.measure_r_eff``
      does).
    - If the drain is not reachable from the source, ``calculate`` returns
      an infinite impedance without solving.
    - M is solved with ``np.linalg.solve`` when its 2-norm condition number is
      at most 1e10 and with least squares otherwise; ``is_singular`` reports
      which. An internal component cut off from both terminals makes M
      singular.
    - An impedance is reported as infinite when I_total <= 1e-10, an absolute
      threshold, so resistances above about 1e10 ohm read as open circuits.
    - In the distance-dependent model an edge of zero (or non-finite) length
      gets zero conductance, i.e. it is treated as open, not as a short.
    """
    
    def __init__(self, intersections, graph, segment_to_nodes, 
                 source_boundary_nodes=None, drain_boundary_nodes=None,
                 use_virtual_nodes=False, edge_conductance=None):
        """
        Initialize the calculator.
        
        Parameters:
        -----------
        intersections : list
            List of [x, y] coordinates for each node
        graph : list
            List where graph[i] = segments at node i
        segment_to_nodes : dict
            Mapping from segment index to (node_a, node_b) tuple
        source_boundary_nodes : list, optional
            List of node indices at source boundary. If provided with 
            use_virtual_nodes=True, a virtual source node will be added
            connecting to all these nodes.
        drain_boundary_nodes : list, optional
            List of node indices at drain boundary. If provided with
            use_virtual_nodes=True, a virtual drain node will be added
            connecting to all these nodes.
        use_virtual_nodes : bool
            If True and boundary nodes are provided, add virtual source/drain
            nodes that connect to all boundary nodes (zero resistance connections).
            This properly handles multiple entry/exit points.
        edge_conductance : dict, optional
            Mapping from edge index to conductance multiplier (for parallel edges).
            If edge has n parallel resistors, its conductance is n/k instead of 1/k.
        """
        self.intersections = list(intersections)
        self.graph = [list(g) for g in graph]
        self.segment_to_nodes = dict(segment_to_nodes)
        self.use_virtual_nodes = use_virtual_nodes
        self.edge_conductance = edge_conductance or {}  # Default: no multipliers
        
        # Store original counts before adding virtual nodes
        self.original_n_nodes = len(intersections)
        self.original_n_edges = len(segment_to_nodes)
        
        # Find source and drain nodes based on x-coordinate
        self.source_node = None
        self.drain_node = None
        
        # Handle virtual nodes for multiple boundary points
        if use_virtual_nodes and source_boundary_nodes and drain_boundary_nodes:
            self._add_virtual_nodes(source_boundary_nodes, drain_boundary_nodes)
        else:
            self._find_boundary_nodes()
        
        self.n_nodes = len(self.intersections)
        self.n_edges = len(self.segment_to_nodes)
    
    def _add_virtual_nodes(self, source_boundary_nodes, drain_boundary_nodes):
        """
        Add virtual source and drain nodes connected to all boundary nodes.
        
        This creates:
        - A virtual SOURCE node connected to all source_boundary_nodes
        - A virtual DRAIN node connected to all drain_boundary_nodes
        
        The connections are zero-resistance (handled in the matrix).
        """
        # Virtual source node index (after all real nodes)
        self.source_node = len(self.intersections)
        self.drain_node = len(self.intersections) + 1
        
        # Add virtual node positions (outside the [0,1] domain for visualization)
        # Source: far left, Drain: far right
        dim = _infer_dim(self.intersections)
        self.intersections.append([-0.1] + [0.5] * (dim - 1))  # Virtual source position
        self.intersections.append([1.1] + [0.5] * (dim - 1))   # Virtual drain position
        
        # Add empty graph entries for virtual nodes
        self.graph.append([])  # Virtual source
        self.graph.append([])  # Virtual drain
        
        # Add edges from virtual source to all source boundary nodes
        next_edge = max(self.segment_to_nodes.keys()) + 1 if self.segment_to_nodes else 0
        
        self.virtual_source_edges = []
        for node in source_boundary_nodes:
            self.segment_to_nodes[next_edge] = (self.source_node, node)
            self.graph[self.source_node].append(next_edge)
            self.graph[node].append(next_edge)
            self.virtual_source_edges.append(next_edge)
            next_edge += 1
        
        # Add edges from all drain boundary nodes to virtual drain
        self.virtual_drain_edges = []
        for node in drain_boundary_nodes:
            self.segment_to_nodes[next_edge] = (node, self.drain_node)
            self.graph[node].append(next_edge)
            self.graph[self.drain_node].append(next_edge)
            self.virtual_drain_edges.append(next_edge)
            next_edge += 1
        
        self.source_boundary_nodes = source_boundary_nodes
        self.drain_boundary_nodes = drain_boundary_nodes
    
    def _find_boundary_nodes(self):
        """Find source (min x) and drain (max x) nodes."""
        min_x = float('inf')
        max_x = float('-inf')
        
        for i, point in enumerate(self.intersections):
            x = point[0]
            if x < min_x:
                min_x = x
                self.source_node = i
            if x > max_x:
                max_x = x
                self.drain_node = i
        
        self.virtual_source_edges = []
        self.virtual_drain_edges = []
    
    def _check_connectivity(self, verbose=False):
        """
        Check if the network is connected using BFS.
        Returns (is_connected, reachable_nodes).
        """
        if self.n_nodes == 0:
            return True, set()
        
        # Build adjacency list from segment_to_nodes
        adj = {i: set() for i in range(self.n_nodes)}
        for edge, (n1, n2) in self.segment_to_nodes.items():
            adj[n1].add(n2)
            adj[n2].add(n1)
        
        # BFS from source node
        visited = set()
        queue = [self.source_node]
        visited.add(self.source_node)
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        is_connected = self.drain_node in visited
        
        if verbose:
            print(f"\nConnectivity check:")
            print(f"  Reachable nodes: {len(visited)}/{self.n_nodes}")
            print(f"  Drain reachable: {is_connected}")
        
        return is_connected, visited
    
    def calculate(self, k=1.0, distance_dependent=False, edge_lengths=None, verbose=False):
        """
        Calculate the equivalent impedance.

        Parameters:
        -----------
        k : float
            Resistance of each segment (all equal for real segments), unless
            distance_dependent=True, in which case k is used as R0 and each
            edge's resistance is R0 * L_edge (L_edge = Euclidean distance
            between the edge's two nodes).
        edge_lengths : array-like, optional
            Explicit length for each edge, used instead of measuring the
            distance between this network's own node coordinates. Only
            meaningful together with distance_dependent=True, and needed when
            the stored coordinates no longer describe the real conductors --
            after merge_boundary_nodes, where each terminal sits at the average
            position of the face it replaced (see merged_edge_true_lengths).
        distance_dependent : bool
            If True, resistance per edge scales with its physical length
            instead of being the same (k) for every edge.
        verbose : bool
            Print detailed calculation steps

        Returns:
        --------
        result : dict
            Dictionary containing:
            - 'impedance': Equivalent resistance
            - 'voltages': Voltage at each node
            - 'currents': Current through each edge
            - 'edge_lengths': Euclidean length of each edge
            - 'edge_resistances': Resistance of each edge
            - 'A': Incidence matrix
            - 'L': Laplacian matrix
        """
        if verbose:
            print(f"\nSource node: {self.source_node}")
            print(f"Drain node: {self.drain_node}")
            print(f"Number of nodes: {self.n_nodes}")
            print(f"Number of edges: {self.n_edges}")
            if self.use_virtual_nodes:
                print(f"Using virtual nodes: YES")
                print(f"  Virtual source edges: {self.virtual_source_edges}")
                print(f"  Virtual drain edges: {self.virtual_drain_edges}")
        
        # Pre-check: Verify network connectivity
        is_connected, reachable = self._check_connectivity(verbose=verbose)
        
        if not is_connected:
            if verbose:
                print("\n[WARNING]  WARNING: Network is not connected!")
                print("    Drain is not reachable from source - returning infinite impedance.")
            return {
                'impedance': float('inf'),
                'voltages': np.zeros(self.n_nodes),
                'currents': np.zeros(self.n_edges),
                'edge_lengths': np.zeros(self.n_edges),
                'edge_resistances': np.full(self.n_edges, np.inf),
                'A': np.zeros((self.n_edges, self.n_nodes)),
                'L': np.zeros((self.n_nodes, self.n_nodes)),
                'I_total': 0.0
            }

        # Step 1: Build incidence matrix A
        A = build_incidence_matrix(self.n_nodes, self.n_edges, self.segment_to_nodes)

        if verbose:
            print(f"\nIncidence matrix A ({self.n_edges}x{self.n_nodes}):")
            print(A)

        # Step 2: Build conductance matrix (diagonal)
        # G[e] = 1/R[e] for each edge
        # Virtual edges have very high conductance (low resistance)
        # Parallel edges: conductance = n/k where n is the number of parallel wires
        # distance_dependent: R[e] = k * L[e] instead of R[e] = k for every edge
        virtual_edges = set(self.virtual_source_edges + self.virtual_drain_edges)
        k_virtual = k * 1e-6  # Very small resistance for virtual edges

        if edge_lengths is None:
            edge_lengths = np.zeros(self.n_edges)
            for e, (n1, n2) in self.segment_to_nodes.items():
                if n1 != n2:
                    edge_lengths[e] = np.linalg.norm(
                        np.array(self.intersections[n2]) - np.array(self.intersections[n1])
                    )
        else:
            edge_lengths = np.asarray(edge_lengths, dtype=float)
            if len(edge_lengths) != self.n_edges:
                raise ValueError(f"edge_lengths has {len(edge_lengths)} entries but the "
                                 f"network has {self.n_edges} edges")

        G = np.zeros(self.n_edges)
        for e in range(self.n_edges):
            if e in virtual_edges:
                G[e] = 1.0 / k_virtual  # High conductance for virtual
            else:
                # Get conductance multiplier (number of parallel resistors)
                multiplier = self.edge_conductance.get(e, 1)
                if distance_dependent:
                    L_e = edge_lengths[e]
                    G[e] = multiplier / (k * L_e) if 0 < L_e < np.inf else 0.0
                else:
                    G[e] = multiplier / k  # n parallel resistors = n x conductance

        G_matrix = np.diag(G)
        edge_resistances = np.divide(1.0, G, out=np.full(self.n_edges, np.inf), where=G > 0)

        if verbose:
            print(f"\nConductance matrix G (diagonal):")
            if distance_dependent:
                print(f"  Distance-dependent mode: R[e] = {k} * L[e]")
            else:
                print(f"  Base conductance = 1/{k} = {1/k:.6f}")
            if self.edge_conductance:
                print(f"  Edge conductance multipliers: {self.edge_conductance}")
            if self.use_virtual_nodes:
                print(f"  Virtual edges: conductance = 1/{k_virtual} = {1/k_virtual:.2e}")
        
        # Step 3: Build weighted Laplacian L = A^T * G * A
        L = A.T @ G_matrix @ A
        
        if verbose:
            print(f"\nWeighted Laplacian L = A^T * G * A:")
            print(L)
        
        # Step 4: Identify boundary and internal nodes
        # Boundary: source and drain
        # Internal: all others
        boundary_nodes = [self.drain_node, self.source_node]
        internal_nodes = [i for i in range(self.n_nodes) if i not in boundary_nodes]
        
        if verbose:
            print(f"\nBoundary nodes: {boundary_nodes}")
            print(f"Internal nodes: {internal_nodes}")
        
        # Step 4: Extract submatrices
        # M = L_ii (internal-internal block)
        # L_ib = internal-boundary block
        n_internal = len(internal_nodes)
        n_boundary = len(boundary_nodes)
        
        M = np.zeros((n_internal, n_internal))
        L_ib = np.zeros((n_internal, n_boundary))
        
        for i, ni in enumerate(internal_nodes):
            for j, nj in enumerate(internal_nodes):
                M[i, j] = L[ni, nj]
            for j, nj in enumerate(boundary_nodes):
                L_ib[i, j] = L[ni, nj]
        
        if verbose:
            print(f"\nReduced matrix M (internal nodes):")
            print(M)
            print(f"\nL_ib (internal-boundary):")
            print(L_ib)
        
        # Step 5: Set boundary voltages
        # V_drain = 0, V_source = 1
        V_boundary = np.zeros(n_boundary)
        for i, node in enumerate(boundary_nodes):
            if node == self.source_node:
                V_boundary[i] = 1.0
            else:  # drain
                V_boundary[i] = 0.0
        
        if verbose:
            print(f"\nBoundary voltages: {V_boundary}")
        
        # Step 6: Calculate boundary vector b = -L_ib * V_boundary
        b = -L_ib @ V_boundary
        
        if verbose:
            print(f"\nBoundary vector b = -L_ib * V_boundary:")
            print(b)
        
        # Step 7: Solve for internal voltages: M * V_internal = b
        is_singular = False
        condition_number = 1.0

        if n_internal > 0:
            try:
                condition_number = np.linalg.cond(M)
                is_singular = condition_number > 1e10
            except np.linalg.LinAlgError:
                condition_number = float('inf')
                is_singular = True

            try:
                if is_singular:
                    if verbose:
                        print(f"\n[WARNING]  Matrix M is ill-conditioned "
                              f"(cond={condition_number:.2e})!")
                        print("    Using least-squares solution...")
                    V_internal, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                else:
                    V_internal = np.linalg.solve(M, b)
            except np.linalg.LinAlgError as e:
                is_singular = True
                if verbose:
                    print(f"\n[WARNING]  Linear algebra error: {e}")
                    print("    Using least-squares solution...")
                V_internal, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
        else:
            V_internal = np.array([])
        
        if verbose:
            print(f"\nInternal voltages V = M^(-1) * b:")
            print(V_internal)
        
        # Step 8: Assemble full voltage vector
        V_full = np.zeros(self.n_nodes)
        for i, node in enumerate(boundary_nodes):
            V_full[node] = V_boundary[i]
        for i, node in enumerate(internal_nodes):
            V_full[node] = V_internal[i]
        
        if verbose:
            print(f"\nFull voltage vector:")
            for i in range(self.n_nodes):
                print(f"  V[{i}] = {V_full[i]:.4f}")
        
        # Step 9: Calculate currents I = G * (-A * V)
        # Current through edge e is: I[e] = G[e] * (V[n2] - V[n1])
        voltage_drops = -A @ V_full  # Voltage drop across each edge
        I = G_matrix @ voltage_drops  # Current = conductance x voltage drop
        
        if verbose:
            print(f"\nEdge currents I = G * (-A * V):")
            for e in range(self.n_edges):
                is_virtual = e in virtual_edges
                marker = " (virtual)" if is_virtual else ""
                print(f"  I[{e}] = {I[e]:.6f}{marker}")
        
        # Step 10: Calculate total current into drain
        # Sum currents flowing into the drain node
        # Self-loops (n1 == n2) are skipped because their incidence-matrix row
        # is zero, so I[edge] is already zero from Step 9.  We skip them here
        # explicitly as well for robustness.
        I_total = 0.0
        for edge, (n1, n2) in self.segment_to_nodes.items():
            if n1 == n2:
                continue
            if n2 == self.drain_node:
                I_total += I[edge]
            elif n1 == self.drain_node:
                I_total -= I[edge]
        
        I_total = abs(I_total)
        
        if verbose:
            print(f"\nTotal current into drain: I_total = {I_total:.6f}")
        
        # Step 11: Calculate impedance R = V / I_total = 1 / I_total
        # (since V_source - V_drain = 1V)
        if I_total > 1e-10:
            R = 1.0 / I_total
        else:
            R = float('inf')
        
        if verbose:
            print(f"\nImpedance R = V / I_total = 1 / {I_total:.6f} = {R:.4f} Ohm")
        
        return {
            'impedance': R,
            'voltages': V_full,
            'currents': I,
            'edge_lengths': edge_lengths,
            'edge_resistances': edge_resistances,
            'A': A,
            'L': L,
            'I_total': I_total,
            'is_singular': is_singular,
            'condition_number': condition_number,
        }
    
    def print_summary(self, result):
        """Print a summary of the calculation results."""
        print("=" * 60)
        print("IMPEDANCE CALCULATION SUMMARY")
        print("=" * 60)
        print(f"\nGraph: {self.n_nodes} nodes, {self.n_edges} edges")
        if self.use_virtual_nodes:
            print(f"  (includes 2 virtual nodes: source at {self.source_node}, drain at {self.drain_node})")
            print(f"  Original network: {self.original_n_nodes} nodes, {self.original_n_edges} edges")
        print(f"Source node: {self.source_node}")
        print(f"Drain node: {self.drain_node}")
        print(f"\nVoltages:")
        for i, v in enumerate(result['voltages']):
            node_type = ""
            if i == self.source_node:
                node_type = " (VIRTUAL SOURCE)" if self.use_virtual_nodes else " (SOURCE)"
            elif i == self.drain_node:
                node_type = " (VIRTUAL DRAIN)" if self.use_virtual_nodes else " (DRAIN)"
            print(f"  V[{i}]{node_type} = {v:.4f}")
        print(f"\nCurrents (real edges only):")
        virtual_edges = set(self.virtual_source_edges + self.virtual_drain_edges)
        for i, c in enumerate(result['currents']):
            if i not in virtual_edges:
                n1, n2 = self.segment_to_nodes[i]
                print(f"  I[{i}] (Node {n1} -> Node {n2}) = {c:.6f}")
        print(f"\nTotal current: {result['I_total']:.6f}")
        print(f"\n>>> IMPEDANCE: R = {result['impedance']:.4f} Ohm <<<")
        print("=" * 60)


def calculate_path_impedance(intersections, graph, segment_to_nodes, k=1.0, verbose=False,
                             source_boundary_nodes=None, drain_boundary_nodes=None,
                             use_virtual_nodes=False, distance_dependent=False):
    """
    Convenience function to calculate impedance in one call.

    Parameters:
    -----------
    intersections : list
        List of coordinates for each node (2D `[x, y]` or 3D `[x, y, z]`)
    graph : list
        List where graph[i] = segments at node i
    segment_to_nodes : dict
        Mapping from segment index to (node_a, node_b) tuple
    k : float
        Resistance of each segment, or R0 (resistance per unit length) when
        distance_dependent=True
    verbose : bool
        Print detailed steps
    source_boundary_nodes : list, optional
        List of node indices at source boundary (for virtual node mode)
    drain_boundary_nodes : list, optional
        List of node indices at drain boundary (for virtual node mode)
    use_virtual_nodes : bool
        If True, add virtual source/drain nodes connected to all boundary nodes
    distance_dependent : bool
        If True, resistance per edge is k * L_edge instead of a uniform k

    Returns:
    --------
    R : float
        Equivalent impedance
    """
    calc = ImpedanceCalculator(
        intersections, graph, segment_to_nodes,
        source_boundary_nodes=source_boundary_nodes,
        drain_boundary_nodes=drain_boundary_nodes,
        use_virtual_nodes=use_virtual_nodes
    )
    result = calc.calculate(k=k, distance_dependent=distance_dependent, verbose=verbose)
    return result['impedance']


def print_merge_report(path_intersections, merged_intersections,
                       merge_mapping, merged_source_node, merged_drain_node,
                       merged_segment_to_nodes, edge_conductance,
                       nodes_in_path, intersections_points,
                       source_boundary_new, drain_boundary_new):
    """
    Print a detailed report of what happened during the merge operation.

    Returns (nodes_merged_to_source, nodes_merged_to_drain) lists of dicts
    for use in downstream steps.
    """
    nodes_merged_to_source = []
    nodes_merged_to_drain = []
    internal_nodes_mapping = []

    for old_local_idx, new_merged_idx in sorted(merge_mapping.items()):
        original_global_node = nodes_in_path[old_local_idx]
        x_pos = intersections_points[original_global_node][0]
        y_pos = intersections_points[original_global_node][1]
        entry = {'local': old_local_idx, 'global': original_global_node,
                 'x': x_pos, 'y': y_pos}
        if new_merged_idx == merged_source_node:
            nodes_merged_to_source.append(entry)
        elif new_merged_idx == merged_drain_node:
            nodes_merged_to_drain.append(entry)
        else:
            entry['merged'] = new_merged_idx
            internal_nodes_mapping.append(entry)

    print("="*70)
    print("MERGE REPORT")
    print("="*70)
    print(f"\n  BEFORE: {len(path_intersections)} nodes")
    print(f"    {len(nodes_merged_to_source)} SOURCE (x < 0.1), "
          f"{len(nodes_merged_to_drain)} DRAIN (x > 0.9), "
          f"{len(internal_nodes_mapping)} INTERNAL")
    print(f"\n  AFTER:  {len(merged_intersections)} nodes")
    print(f"    1 merged SOURCE (index {merged_source_node}), "
          f"1 merged DRAIN (index {merged_drain_node}), "
          f"{len(internal_nodes_mapping)} INTERNAL")
    print(f"\n  Edges: {len(merged_segment_to_nodes)}")
    parallel = sum(1 for v in edge_conductance.values() if v > 1)
    if parallel:
        print(f"    ({parallel} with parallel merges)")

    return nodes_merged_to_source, nodes_merged_to_drain


def print_detailed_impedance(merged_segment_to_nodes, merged_intersections,
                             merged_source_node, merged_drain_node,
                             edge_conductance, result, k):
    """
    Print the full incidence matrix, Laplacian, voltages, currents, and
    final impedance — the pedagogical breakdown from the notebook.
    """
    n_edges = len(merged_segment_to_nodes)
    n_nodes = len(merged_intersections)
    voltages = result['voltages']
    currents = result['currents']

    print("="*80)
    print("DETAILED IMPEDANCE CALCULATION")
    print("="*80)
    print(f"\n  MERGED NETWORK: {n_nodes} nodes, {n_edges} edges, k = {k} ohm")

    # Incidence matrix
    A = np.zeros((n_edges, n_nodes))
    for seg_idx, (n1, n2) in merged_segment_to_nodes.items():
        A[seg_idx, n1] = 1
        A[seg_idx, n2] = -1

    print("\n" + "-"*80)
    print("INCIDENCE MATRIX A (edges x nodes)")
    print("-"*80)
    header = "         "
    for i in range(n_nodes):
        if i == merged_source_node:
            header += "  S  "
        elif i == merged_drain_node:
            header += "  D  "
        else:
            header += f" I{i}  "
    print(header)
    for e in range(n_edges):
        row = f"  e{e}:   ["
        row += "  ".join(f"{int(A[e, n]):3d}" for n in range(n_nodes))
        row += f" ]  (nodes {merged_segment_to_nodes[e][0]}<->{merged_segment_to_nodes[e][1]})"
        print(row)

    # Conductance
    print("\n" + "-"*80)
    print("CONDUCTANCE OF EACH EDGE")
    print("-"*80)
    G_diag = []
    for seg_idx in range(n_edges):
        mult = edge_conductance.get(seg_idx, 1)
        g = mult / k
        G_diag.append(g)
        print(f"  Edge {seg_idx}: G = {mult}/{k} = {g:.6f} S  (x{mult} parallel)")

    # Laplacian
    G_matrix = np.diag(G_diag)
    L_matrix = A.T @ G_matrix @ A
    print("\n" + "-"*80)
    print("LAPLACIAN L = At.G.A")
    print("-"*80)
    header = "         "
    for i in range(n_nodes):
        if i == merged_source_node:
            header += "    S   "
        elif i == merged_drain_node:
            header += "    D   "
        else:
            header += f"   I{i}   "
    print(header)
    for i in range(n_nodes):
        if i == merged_source_node:
            lbl = "  S: "
        elif i == merged_drain_node:
            lbl = "  D: "
        else:
            lbl = f" I{i}: "
        row = lbl + "[" + "  ".join(f"{L_matrix[i, j]:7.4f}" for j in range(n_nodes)) + " ]"
        print(row)

    # Solution
    print("\n" + "-"*80)
    print("SOLUTION: VOLTAGES AND CURRENTS")
    print("-"*80)
    print(f"\n  Boundary: V[{merged_source_node}]=1.0 V, V[{merged_drain_node}]=0.0 V")
    print(f"\n  Voltages:")
    for i in range(n_nodes):
        tag = "SOURCE" if i == merged_source_node else ("DRAIN" if i == merged_drain_node else "INTERNAL")
        print(f"    V[{i}] = {voltages[i]:.6f} V  [{tag}]")

    print(f"\n  Currents:")
    total_source_current = 0.0
    for seg_idx, (n1, n2) in merged_segment_to_nodes.items():
        mult = edge_conductance.get(seg_idx, 1)
        g = mult / k
        I_e = currents[seg_idx]
        print(f"    Edge {seg_idx}: I = {g:.4f} x ({voltages[n1]:.4f} - {voltages[n2]:.4f}) = {I_e:.6f} A")
        if n1 == merged_source_node:
            total_source_current += I_e
        elif n2 == merged_source_node:
            total_source_current -= I_e

    R_calc = 1.0 / total_source_current if total_source_current != 0 else float('inf')
    print("\n" + "-"*80)
    print("FINAL RESULT")
    print("-"*80)
    print(f"  I_total = {total_source_current:.6f} A")
    print(f"  R_equivalent = 1.0 / {total_source_current:.6f} = {R_calc:.4f} ohm")
    print(f"  Verification (ImpedanceCalculator): {result['impedance']:.4f} ohm")

    return R_calc, total_source_current


# # Test when run directly
# if __name__ == "__main__":
#     print("Testing impedance_calculator module...")
#     print()
    
#     # Book example (Figure 2.7) - Expected R = 625 Ohm
#     # 
#     # From the book's Laplacian matrix:
#     # L = [[3, -1, -1, -1],
#     #      [-1, 3, -1, -1],
#     #      [-1, -1, 2, 0],
#     #      [-1, -1, 0, 2]]
#     #
#     # This tells us:
#     # - Node 0 (drain) connected to: 1, 2, 3
#     # - Node 1 connected to: 0, 2, 3  
#     # - Node 2 connected to: 0, 1 (NOT to 3)
#     # - Node 3 (source) connected to: 0, 1 (NOT to 2)
#     #
#     # 5 edges: (0-1), (0-2), (0-3), (1-2), (1-3)
#     # Note: there IS a direct source-drain edge!
#     #
#     intersections = [
#         [0.95, 0.50],   # Node 0 (drain, book Node 1)
#         [0.55, 0.70],   # Node 1 (book Node 2)
#         [0.55, 0.30],   # Node 2 (book Node 3)
#         [0.05, 0.50],   # Node 3 (source, book Node 4)
#     ]
    
#     graph = [
#         [0, 1, 2],   # Node 0 (drain): edges to nodes 1, 2, 3
#         [0, 3, 4],   # Node 1: edges to nodes 0, 2, 3
#         [1, 3],      # Node 2: edges to nodes 0, 1
#         [2, 4],      # Node 3 (source): edges to nodes 0, 1
#     ]
    
#     # 5 edges matching the book's Laplacian:
#     segment_to_nodes = {
#         0: (0, 1),   # Edge 0: Drain (0) - Node 1 (1)
#         1: (0, 2),   # Edge 1: Drain (0) - Node 2 (2)
#         2: (0, 3),   # Edge 2: Drain (0) - Source (3) DIRECT!
#         3: (1, 2),   # Edge 3: Node 1 (1) - Node 2 (2)
#         4: (1, 3),   # Edge 4: Node 1 (1) - Source (3)
#     }
    
#     calc = ImpedanceCalculator(intersections, graph, segment_to_nodes)
#     result = calc.calculate(k=1000, verbose=True)
    
#     print()
#     calc.print_summary(result)
    
#     print()
#     print(f"Expected: R = 625.0 Ohm")
#     print(f"Calculated: R = {result['impedance']:.1f} Ohm")
#     if abs(result['impedance'] - 625.0) < 1:
#         print("[OK] TEST PASSED!")
#     else:
#         print("[FAIL] TEST FAILED!")
