"""
Multi-terminal generalization of ImpedanceCalculator.calculate() (see
utils/impedance_calculator.py): instead of exactly one source node fixed at
1V and one drain node fixed at 0V, this solves the same weighted-Laplacian
Dirichlet problem for an arbitrary number of terminal nodes, each with its
own prescribed voltage.

Most of calculate()'s internals were already fully general (the M/L_ib
reduction and solve never assumed exactly 2 boundary nodes) -- only the
boundary-node list length, the fixed [1.0, 0.0] voltage vector, and the
single source->drain current/impedance computation were hardcoded to the
2-terminal case. solve_multi_terminal reuses build_incidence_matrix
unchanged and duplicates the small (~15-line) conductance-diagonal
construction rather than refactoring ImpedanceCalculator, per the decision
to leave the existing 2-terminal code untouched.

Also provides the geometry/bookkeeping helpers shared by both new
notebooks (spherical terminals anywhere vs. subdivided x<0.1/x>0.9 faces):
terminal-membership assignment, voltage assignment, and the percolation/
validity criterion for a multi-terminal draw.
"""
import numpy as np

from impedance_calculator import build_incidence_matrix


def solve_multi_terminal(intersections, segment_to_nodes, terminal_nodes, terminal_voltages,
                         k=1.0, distance_dependent=False, edge_conductance=None):
    """
    Solve the resistor network with Dirichlet boundary conditions at
    `terminal_nodes` (each fixed to the corresponding value in
    `terminal_voltages`), for internal (non-terminal) node voltages, edge
    currents, and the net current flowing into each terminal.

    Parameters
    ----------
    intersections : list of coordinate tuples (2D or 3D)
    segment_to_nodes : dict edge_idx -> (n1, n2)
    terminal_nodes : list[int], length >= 2, the (already-merged) terminal
        node indices
    terminal_voltages : list[float], same length/order as terminal_nodes
    k : resistance scale (uniform per-edge resistance, or R0 if
        distance_dependent=True)
    distance_dependent : if True, edge resistance = k * edge_length instead
        of a uniform k
    edge_conductance : dict edge_idx -> parallel-edge multiplier (default 1),
        as produced by merge_terminal_groups/merge_boundary_nodes

    Returns
    -------
    dict with:
      voltages           : full voltage vector, length n_nodes
      currents           : per-edge current, length n_edges (same sign
                           convention as ImpedanceCalculator.calculate:
                           I[e] = G[e] * (V[n1] - V[n2]))
      terminal_currents  : {terminal_node: net_current_into_node}
      current_imbalance  : sum(terminal_currents.values()) -- must be ~0 by
                           current conservation (Kirchhoff's current law
                           applied to the whole network); a cheap, strong
                           sanity check for ANY topology/terminal count
      A, L, M, L_ib, edge_lengths, edge_resistances, is_singular,
      condition_number

    Notes
    -----
    Same linear algebra as ``ImpedanceCalculator.calculate``: least squares
    replaces ``np.linalg.solve`` when cond(M) > 1e10, and under the
    distance-dependent model a zero-length edge gets zero conductance (open).
    Voltages are only determined for nodes connected to at least one
    terminal; use ``validate_multi_terminal_percolation`` before solving.
    """
    if len(terminal_nodes) != len(terminal_voltages):
        raise ValueError("terminal_nodes and terminal_voltages must have the same length")
    if len(set(terminal_nodes)) != len(terminal_nodes):
        raise ValueError("terminal_nodes must be distinct")

    n_nodes = len(intersections)
    n_edges = len(segment_to_nodes)
    edge_conductance = edge_conductance or {}

    A = build_incidence_matrix(n_nodes, n_edges, segment_to_nodes)

    inter = np.asarray(intersections, dtype=float)
    edge_lengths = np.zeros(n_edges)
    for e, (n1, n2) in segment_to_nodes.items():
        if n1 != n2:
            edge_lengths[e] = np.linalg.norm(inter[n2] - inter[n1])

    G = np.zeros(n_edges)
    for e in range(n_edges):
        multiplier = edge_conductance.get(e, 1)
        if distance_dependent:
            L_e = edge_lengths[e]
            G[e] = multiplier / (k * L_e) if L_e > 0 else 0.0
        else:
            G[e] = multiplier / k
    G_matrix = np.diag(G)
    edge_resistances = np.divide(1.0, G, out=np.full(n_edges, np.inf), where=G > 0)

    L = A.T @ G_matrix @ A

    boundary_nodes = list(terminal_nodes)
    internal_nodes = [i for i in range(n_nodes) if i not in set(boundary_nodes)]
    n_internal = len(internal_nodes)
    n_boundary = len(boundary_nodes)

    M = np.zeros((n_internal, n_internal))
    L_ib = np.zeros((n_internal, n_boundary))
    for i, ni in enumerate(internal_nodes):
        for j, nj in enumerate(internal_nodes):
            M[i, j] = L[ni, nj]
        for j, nj in enumerate(boundary_nodes):
            L_ib[i, j] = L[ni, nj]

    V_boundary = np.array(terminal_voltages, dtype=float)
    b = -L_ib @ V_boundary

    is_singular = False
    condition_number = 1.0
    if n_internal > 0:
        try:
            condition_number = float(np.linalg.cond(M))
            is_singular = condition_number > 1e10
        except np.linalg.LinAlgError:
            condition_number = float('inf')
            is_singular = True

        if is_singular:
            V_internal, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
        else:
            V_internal = np.linalg.solve(M, b)
    else:
        V_internal = np.array([])

    V_full = np.zeros(n_nodes)
    for i, node in enumerate(boundary_nodes):
        V_full[node] = V_boundary[i]
    for i, node in enumerate(internal_nodes):
        V_full[node] = V_internal[i]

    voltage_drops = -A @ V_full
    I = G_matrix @ voltage_drops

    terminal_currents = {}
    for t in terminal_nodes:
        total = 0.0
        for edge, (n1, n2) in segment_to_nodes.items():
            if n1 == n2:
                continue
            if n2 == t:
                total += I[edge]
            elif n1 == t:
                total -= I[edge]
        terminal_currents[t] = total

    return {
        'voltages': V_full,
        'currents': I,
        'terminal_currents': terminal_currents,
        'current_imbalance': float(sum(terminal_currents.values())),
        'A': A,
        'L': L,
        'M': M,
        'L_ib': L_ib,
        'edge_lengths': edge_lengths,
        'edge_resistances': edge_resistances,
        'is_singular': is_singular,
        'condition_number': condition_number,
    }


def multi_terminal_pinv_currents(intersections, segment_to_nodes, terminal_nodes, terminal_voltages,
                                 k=1.0, distance_dependent=False, edge_conductance=None):
    """
    Independent cross-check of solve_multi_terminal: builds the weighted
    Laplacian directly from node coordinates and edge lengths (no
    build_incidence_matrix), and computes per-edge currents directly from
    voltage differences (no incidence matrix) -- shares no code path with
    solve_multi_terminal, in the same spirit as laplacian_utils.
    effective_resistance_pinv cross-checking ImpedanceCalculator.

    Returns dict with 'voltages', 'terminal_currents', 'current_imbalance'
    (same shapes/meaning as solve_multi_terminal's).
    """
    inter = np.asarray(intersections, dtype=float)
    edge_conductance = edge_conductance or {}
    n = len(inter)

    conductances = {}
    L = np.zeros((n, n))
    for e, (n1, n2) in segment_to_nodes.items():
        if n1 == n2:
            continue
        multiplier = edge_conductance.get(e, 1)
        if distance_dependent:
            length = np.linalg.norm(inter[n2] - inter[n1])
            g = multiplier / (k * length) if length > 0 else 0.0
        else:
            g = multiplier / k
        conductances[e] = (n1, n2, g)
        L[n1, n1] += g; L[n2, n2] += g
        L[n1, n2] -= g; L[n2, n1] -= g

    boundary_nodes = list(terminal_nodes)
    internal_nodes = [i for i in range(n) if i not in set(boundary_nodes)]

    M = L[np.ix_(internal_nodes, internal_nodes)]
    L_ib = L[np.ix_(internal_nodes, boundary_nodes)]
    V_boundary = np.array(terminal_voltages, dtype=float)
    b = -L_ib @ V_boundary
    V_internal = np.linalg.solve(M, b) if len(internal_nodes) > 0 else np.array([])

    V_full = np.zeros(n)
    for i, node in enumerate(boundary_nodes):
        V_full[node] = V_boundary[i]
    for i, node in enumerate(internal_nodes):
        V_full[node] = V_internal[i]

    terminal_currents = {t: 0.0 for t in terminal_nodes}
    for e, (n1, n2, g) in conductances.items():
        current = g * (V_full[n1] - V_full[n2])  # current flowing n1 -> n2
        if n1 in terminal_currents:
            terminal_currents[n1] -= current
        if n2 in terminal_currents:
            terminal_currents[n2] += current

    return {
        'voltages': V_full,
        'terminal_currents': terminal_currents,
        'current_imbalance': float(sum(terminal_currents.values())),
    }


def summarize_terminal_roles(terminal_node_indices, terminal_voltages, terminal_currents):
    """
    Turn a solve_multi_terminal result into an explicit per-terminal role
    table -- SOURCE (current flows OUT of the terminal into the network) or
    DRAIN (current flows IN from the network) -- instead of leaving the
    reader to infer it from the sign of terminal_currents (see
    solve_multi_terminal's docstring: terminal_currents[t] is the net
    current INTO t, so t < 0 means current leaves t for the network).

    Role is a computed CONSEQUENCE of the solve, not something the caller
    sets: two terminals can share the same prescribed voltage and still end
    up with opposite roles depending on network topology, and a terminal at
    an intermediate voltage can go either way -- only the sign of its
    solved current says which.

    Parameters
    ----------
    terminal_node_indices : list[int], same order as terminal_voltages
    terminal_voltages : list[float]
    terminal_currents : dict node -> net current into node (solve['terminal_currents'])

    Returns
    -------
    list[dict], one entry per terminal, each with keys: node, voltage,
    current, role ("SOURCE" or "DRAIN"). Same order as terminal_node_indices.
    """
    summary = []
    for node, voltage in zip(terminal_node_indices, terminal_voltages):
        current = terminal_currents[node]
        role = "DRAIN" if current > 0 else "SOURCE"
        summary.append({"node": node, "voltage": voltage, "current": current, "role": role})
    return summary


def print_terminal_roles(terminal_node_indices, terminal_voltages, terminal_currents, title=None):
    """
    Pretty-print summarize_terminal_roles as an aligned table -- the
    quickest way to see, at a glance, which terminals are sourcing current
    and which are draining it (see summarize_terminal_roles for the role
    convention).

    Returns the same list[dict] summarize_terminal_roles does.
    """
    summary = summarize_terminal_roles(terminal_node_indices, terminal_voltages, terminal_currents)
    if title:
        print(title)
    for row in summary:
        symbol = "->" if row["role"] == "SOURCE" else "<-"
        print(f"    node {row['node']:>5}   V={row['voltage']:+.3f}   I={row['current']:+.3e}   "
              f"{symbol}  {row['role']}")
    return summary


def split_face_into_quadrants(node_indices, intersections_points, n_subdivisions=4, center=(0.5, 0.5)):
    """
    Split a face's node list (already classified as x<0.1 or x>0.9) into
    sub-groups by position in the (y, z) plane -- for building several
    independent terminals out of one face instead of merging it into one.

    n_subdivisions : 2 -> split by sign(y - center[0])            (2 groups)
                      4 -> split by (sign(y-center[0]), sign(z-center[1]))  (4 groups)

    Returns list[list[int]], length n_subdivisions (a sub-group may be
    empty if no node falls in that quadrant).
    """
    if n_subdivisions not in (2, 4):
        raise ValueError("n_subdivisions must be 2 or 4")

    groups = [[] for _ in range(n_subdivisions)]
    for n in node_indices:
        y, z = intersections_points[n][1], intersections_points[n][2]
        if n_subdivisions == 2:
            idx = 0 if y >= center[0] else 1
        else:
            idx = (0 if y >= center[0] else 1) + (0 if z >= center[1] else 2)
        groups[idx].append(n)
    return groups


def split_face_into_segments(node_indices, intersections_points, n_subdivisions=2):
    """
    2D analog of split_face_into_quadrants: split a face's node list (already
    classified as x<0.1 or x>0.9) into sub-groups by position along y --
    the only remaining axis in 2D, so subdivision is by y-bin instead of a
    (y, z) quadrant.

    n_subdivisions : number of equal-width bins to split [0, 1] (y-range)
        into, e.g. 2 -> split by sign(y - center) (2 groups), 3 -> thirds, etc.

    Returns list[list[int]], length n_subdivisions (a sub-group may be
    empty if no node falls in that bin).
    """
    if n_subdivisions < 2:
        raise ValueError("n_subdivisions must be >= 2")

    edges = np.linspace(0.0, 1.0, n_subdivisions + 1)
    groups = [[] for _ in range(n_subdivisions)]
    for n in node_indices:
        y = intersections_points[n][1]
        idx = min(np.searchsorted(edges, y, side="right") - 1, n_subdivisions - 1)
        idx = max(idx, 0)
        groups[idx].append(n)
    return groups


def assign_nodes_to_spherical_terminals(intersections_points, terminal_centers, terminal_radii):
    """
    Group contact nodes into spherical terminal regions anywhere in the
    domain. A node within radius of more than one center is assigned to the
    NEAREST center (overlapping terminal regions are a configuration choice,
    not an error, but each node belongs to exactly one terminal).

    Returns list[list[int]], one node-index group per (center, radius) pair,
    same order as terminal_centers.
    """
    centers = np.asarray(terminal_centers, dtype=float)
    radii = np.asarray(terminal_radii, dtype=float)
    inter = np.asarray(intersections_points, dtype=float)

    groups = [[] for _ in range(len(centers))]
    for n, point in enumerate(inter):
        dists = np.linalg.norm(centers - point, axis=1)
        within = np.where(dists <= radii)[0]
        if len(within) == 0:
            continue
        nearest = within[np.argmin(dists[within])]
        groups[nearest].append(n)
    return groups


def assign_terminal_voltages(n_terminals, mode="alternating", explicit_values=None, seed=None):
    """
    Generate the prescribed voltage for each of n_terminals terminals.

    mode="alternating" : [1.0, 0.0, 1.0, 0.0, ...] -- guarantees non-trivial
        current flow (the default; random voltages could by chance assign
        the same value to every terminal, giving a trivial zero-current case)
    mode="explicit"     : use explicit_values (length must equal n_terminals)
    mode="random"       : np.random.uniform(0, 1, n_terminals), seeded for
        reproducibility
    """
    if mode == "explicit":
        if explicit_values is None or len(explicit_values) != n_terminals:
            raise ValueError("explicit_values must be provided with length == n_terminals")
        return list(explicit_values)
    if mode == "alternating":
        return [1.0 if i % 2 == 0 else 0.0 for i in range(n_terminals)]
    if mode == "random":
        rng = np.random.RandomState(seed)
        return list(rng.uniform(0, 1, n_terminals))
    raise ValueError(f"unknown mode: {mode}")


def validate_multi_terminal_percolation(terminal_groups, node_to_component):
    """
    A multi-terminal network is usable iff every terminal group is
    non-empty AND has at least one node in the SAME connected component
    (otherwise that terminal is electrically isolated from the others, and
    its prescribed voltage cannot influence -- or be influenced by -- the
    rest of the network; the solve would silently force ~0 current there
    rather than raise, so this must be checked explicitly beforehand).

    Parameters
    ----------
    terminal_groups : list[list[int]] -- node indices per terminal, already
        intersected with whatever "surviving" node set is relevant (e.g.
        post-pruning)
    node_to_component : dict node_idx -> component_id (e.g. built from
        path_analysis.connected_components)

    Returns
    -------
    bool
    """
    if any(len(g) == 0 for g in terminal_groups):
        return False
    components_per_terminal = []
    for g in terminal_groups:
        comps = {node_to_component[n] for n in g if n in node_to_component}
        if not comps:
            return False
        components_per_terminal.append(comps)
    common = set.intersection(*components_per_terminal)
    return len(common) > 0
