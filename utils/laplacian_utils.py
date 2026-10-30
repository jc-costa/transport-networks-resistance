"""
Laplacian matrix utilities for electrical network analysis.
Provides functions to compute the reduced matrix (internal nodes) and print the Laplacian.
"""

import numpy as np
from impedance_calculator import build_incidence_matrix


def compute_reduced_matrix(intersections, segment_to_nodes, source_node, drain_node, k=1.0):
    """
    Build the weighted Laplacian L = A^T G A and extract the reduced matrix M
    for the internal nodes (all nodes except source and drain).

    Returns
    -------
    L         : full (n_nodes × n_nodes) Laplacian
    M         : reduced (n_internal × n_internal) matrix
    cond_M    : condition number of M (inf if empty)
    is_singular: True if cond_M > 1e10
    """
    n_nodes = len(intersections)
    n_edges = len(segment_to_nodes)

    A = build_incidence_matrix(n_nodes, n_edges, segment_to_nodes)
    G_diag = np.array([1.0 / k] * n_edges)
    # Self-loop row has net +1 at the looped node — handled transparently here
    L = A.T @ np.diag(G_diag) @ A

    boundary_nodes = {source_node, drain_node}
    internal_nodes = [i for i in range(n_nodes) if i not in boundary_nodes]

    n_int = len(internal_nodes)
    if n_int == 0:
        return L, np.array([]).reshape(0, 0), float('inf'), False

    M = np.array([[L[i, j] for j in internal_nodes] for i in internal_nodes])

    try:
        cond_M = float(np.linalg.cond(M))
        is_sing = cond_M > 1e10
    except np.linalg.LinAlgError:
        cond_M, is_sing = float('inf'), True

    return L, M, cond_M, is_sing


def _weighted_laplacian_distance_dependent(intersections, segment_to_nodes,
                                            R0=1000.0, edge_conductance=None):
    """
    Build the weighted Laplacian straight from node coordinates and edge
    lengths (R_edge = R0 * edge_length), independent of ImpedanceCalculator /
    build_incidence_matrix. Dimension-agnostic (2D or 3D `intersections`),
    since it only ever takes the norm of a coordinate difference. Shared by
    effective_resistance_pinv (single pair) and
    effective_resistance_from_reference (one reference to every other node).
    """
    inter = np.asarray(intersections, dtype=float)
    edge_conductance = edge_conductance or {}
    n = len(inter)
    L = np.zeros((n, n))
    for e, (n1, n2) in segment_to_nodes.items():
        if n1 == n2:
            continue
        length = np.linalg.norm(inter[n1] - inter[n2])
        g = edge_conductance.get(e, 1) / (R0 * length)
        L[n1, n1] += g; L[n2, n2] += g
        L[n1, n2] -= g; L[n2, n1] -= g
    return L


def effective_resistance_pinv(intersections, segment_to_nodes, source_node, drain_node,
                              R0=1000.0, edge_conductance=None):
    """
    Independent cross-check of the distance-dependent effective resistance via
    the pseudoinverse of the weighted Laplacian (R_eff = e^T L+ e). Dimension-
    agnostic (works for 2D or 3D `intersections`).

    Parameters mirror compute_reduced_matrix, plus:
    R0 : distance-dependent resistance scale (R_edge = R0 * edge_length)
    edge_conductance : dict edge_idx -> parallel-edge multiplier (default 1)
    """
    L = _weighted_laplacian_distance_dependent(intersections, segment_to_nodes,
                                               R0=R0, edge_conductance=edge_conductance)
    n = L.shape[0]
    Lp = np.linalg.pinv(L)
    e_sd = np.zeros(n)
    e_sd[source_node] = 1
    e_sd[drain_node] = -1
    return float(e_sd @ Lp @ e_sd)


def effective_resistance_merged_faces(intersections, segment_to_nodes,
                                      source_nodes, drain_nodes,
                                      R0=1000.0, edge_conductance=None):
    """
    Independent cross-check of the whole-face distance-dependent R_eff, computed
    WITHOUT moving any node.

    Merging a face into one terminal is node IDENTIFICATION: the face nodes are
    tied together by an ideal (zero resistance) conductor, and every wire keeps
    the length it physically has. That is done here by building the weighted
    Laplacian on the ORIGINAL geometry and then summing the rows and columns of
    each face into a single supernode, before the usual R_eff = e^T L+ e.

    Contrast with running effective_resistance_pinv on a MERGED network's own
    coordinates: merge_boundary_nodes parks each terminal at the average
    position of the face it replaces, so under R_edge = R0 * L_edge that
    re-measures every edge touching a terminal and changes the resistance of
    wires that never moved. This function is the reference that
    pipeline_utils.distance_dependent_impedance is expected to reproduce, and
    the two agree to ~1e-12 relative in 2D and 3D.

    Parameters
    ----------
    intersections, segment_to_nodes : the PRE-MERGE network (2D or 3D).
    source_nodes, drain_nodes : iterable of int
        The face node groups, each shorted into one terminal.
    R0 : float
        Resistance per unit length (R_edge = R0 * edge_length).
    edge_conductance : dict, optional
        edge_idx -> parallel multiplier, if the pre-merge network already has one.

    Returns
    -------
    float : R_eff between the two merged terminals.
    """
    inter = np.asarray(intersections, dtype=float)
    edge_conductance = edge_conductance or {}
    n = len(inter)

    L = np.zeros((n, n))
    for e, (n1, n2) in segment_to_nodes.items():
        if n1 == n2:
            continue
        length = float(np.linalg.norm(inter[n1] - inter[n2]))
        if length <= 0:
            continue
        g = edge_conductance.get(e, 1) / (R0 * length)
        L[n1, n1] += g; L[n2, n2] += g
        L[n1, n2] -= g; L[n2, n1] -= g

    source_set, drain_set = set(source_nodes), set(drain_nodes)
    if source_set & drain_set:
        raise ValueError("source_nodes and drain_nodes overlap; a node cannot be "
                         "shorted into both terminals")
    internal = [i for i in range(n) if i not in source_set and i not in drain_set]

    index = {old: new for new, old in enumerate(internal)}
    source_idx, drain_idx = len(internal), len(internal) + 1
    for old in source_set:
        index[old] = source_idx
    for old in drain_set:
        index[old] = drain_idx

    m = len(internal) + 2
    L_merged = np.zeros((m, m))
    for i in range(n):
        row = index[i]
        for j in range(n):
            if L[i, j]:
                L_merged[row, index[j]] += L[i, j]

    e_sd = np.zeros(m)
    e_sd[source_idx] = 1
    e_sd[drain_idx] = -1
    return float(e_sd @ np.linalg.pinv(L_merged) @ e_sd)


def effective_resistance_from_reference(intersections, segment_to_nodes, reference_node,
                                        R0=1000.0, edge_conductance=None):
    """
    R_eff from one reference node to EVERY other node, in one shot.

    The naive way to get this (measure_r_eff-style: one dense O(n^3) solve
    per target node) is wasteful when you want dozens of targets per network,
    e.g. the onion-layer protocol. Instead, compute the Laplacian
    pseudoinverse L+ once (O(n^3), the same order as a single solve), then
    every pairwise R_eff follows from the standard identity
    R_eff(i, j) = L+_ii + L+_jj - 2*L+_ij -- with i fixed at reference_node,
    that's a single vectorized O(n^2) operation for the whole node set.

    Parameters mirror effective_resistance_pinv, minus drain_node (every
    other node is a target).

    Returns
    -------
    r_eff : (n,) ndarray
        r_eff[j] is R_eff(reference_node, j) for every node j (r_eff[reference_node] == 0,
        r_eff[j] == inf for any j not reachable from reference_node -- matching
        ImpedanceCalculator's explicit "drain not reachable from source" handling; the pinv
        formula alone would silently return a finite but physically meaningless value for
        cross-component pairs, since L+ is block-diagonal across disconnected components).
    """
    n = len(intersections)
    L = _weighted_laplacian_distance_dependent(intersections, segment_to_nodes,
                                               R0=R0, edge_conductance=edge_conductance)
    Lp = np.linalg.pinv(L)
    Lp_diag = np.diag(Lp)
    r_eff = Lp_diag[reference_node] + Lp_diag - 2.0 * Lp[reference_node, :]
    r_eff[reference_node] = 0.0

    reachable = _bfs_reachable(n, segment_to_nodes, reference_node)
    r_eff[~reachable] = np.inf
    return r_eff


def _bfs_reachable(n_nodes, segment_to_nodes, start):
    """Boolean mask of nodes reachable from `start` via segment_to_nodes edges."""
    adjacency = [[] for _ in range(n_nodes)]
    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            continue
        adjacency[n1].append(n2)
        adjacency[n2].append(n1)

    reachable = np.zeros(n_nodes, dtype=bool)
    reachable[start] = True
    frontier = [start]
    while frontier:
        next_frontier = []
        for u in frontier:
            for v in adjacency[u]:
                if not reachable[v]:
                    reachable[v] = True
                    next_frontier.append(v)
        frontier = next_frontier
    return reachable


def print_laplacian(intersections, segment_to_nodes, source_node, drain_node, k=1.0, title="", em=None):
    """
    Print a Laplacian summary line and optionally persist metadata via ExperimentManager.

    Parameters
    ----------
    intersections, segment_to_nodes, source_node, drain_node, k, title : standard
    em : ExperimentManager or None
        If provided, saves laplacian_info.json and (optionally) laplacian_matrix.txt
        into the experiment directory via em.save_json().
    """
    n_nodes = len(intersections)
    n_edges = len(segment_to_nodes)
    A = build_incidence_matrix(n_nodes, n_edges, segment_to_nodes)
    G = np.diag([1.0/k] * n_edges)
    L = A.T @ G @ A

    # Compute condition number via compute_reduced_matrix
    _, M, cond_M, is_sing = compute_reduced_matrix(
        intersections, segment_to_nodes, source_node, drain_node, k=k
    )

    # Print concise summary line
    tag = f" [{title}]" if title else ""
    print(f"  Laplacian L ({n_nodes}×{n_nodes}), cond = {cond_M:.3e}, singular = {is_sing}{tag}")

    # Persist to experiment directory if ExperimentManager provided
    if em is not None:
        laplacian_info = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "condition_number": float(cond_M) if np.isfinite(cond_M) else None,
            "is_singular": bool(is_sing),
            "k_ohm": float(k),
            "title": title,
            "source_node": int(source_node),
            "drain_node": int(drain_node),
            "internal_nodes": n_nodes - 2,
        }
        em.save_json(laplacian_info, "laplacian_info.json")

        # Save full matrix to text file for debugging
        import os
        matrix_path = os.path.join(em.experiment_dir, "laplacian_matrix.txt")
        np.savetxt(matrix_path, L * k, fmt="%+.6f", header=f"Laplacian L × k ({n_nodes}×{n_nodes})")
        print(f"    (full matrix saved to laplacian_matrix.txt)")