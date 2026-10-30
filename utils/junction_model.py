"""
Junction resistance for the 2D stick network: two nodes per crossing.

The wire-limited model of the pipeline makes every crossing of two sticks a
single node, i.e. an ideal contact. Here a crossing n of sticks i and j is
split into two nodes, (n, i) on stick i and (n, j) on stick j, joined by a
JUNCTION edge of resistance R_j; the stick segments keep their length-based
resistance R0 * L and now run between the copies that lie on their own stick.
The one new parameter is the junction-to-stick ratio

    rho = R_j / (R0 * a),

the resistance of a junction over the resistance of one whole stick (Zezelj &
Stankovic 2012). rho = 0 (an ideal contact) recovers the wire-limited model
exactly, which is the validation used here: contracting the junction edges
must reproduce distance_dependent_impedance() to floating-point precision.

The graph is built from what run_full_pipeline already stores --
filtered_segment_edge_list (stick, (n1, n2)) in original crossing indices,
graph[n] = the two sticks of crossing n, intersections_points, and the
terminal strips x < 0.1 / x > 0.9 -- restricted to the source-drain component
the pipeline keeps. Terminals are ideal: every copy of a crossing inside a
strip is tied to that terminal, so the junctions inside the strips are shorted
by the electrode. Solves are sparse (the junction graph has up to 2n nodes).

Junction resistances can be a scalar (uniform), an array per crossing
(distributed), or a callable of the two sticks (e.g. a metallic/semiconducting
rule); see junction_resistances().
"""
from collections import deque

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SOURCE_X, DRAIN_X = 0.1, 0.9


def _sd_component(edges, points):
    """Restrict (stick, (n1, n2)) edges to the component joining the two strips,
    the way path_analysis.build_connected_path does."""
    adj = {}
    for _, (n1, n2) in edges:
        adj.setdefault(n1, set()).add(n2)
        adj.setdefault(n2, set()).add(n1)
    source = [n for n in adj if points[n][0] < SOURCE_X]
    drain = [n for n in adj if points[n][0] > DRAIN_X]
    seen = set(source)
    q = deque(source)
    while q:
        n = q.popleft()
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                q.append(m)
    if not any(n in seen for n in drain):
        raise ValueError("no source-drain path in the pruned edge list")
    return [e for e in edges if e[1][0] in seen and e[1][1] in seen]


def build_junction_network(res):
    """
    Junction graph of a run_full_pipeline result (2D).

    Returns a dict:
      n_cross           crossings kept (original indices in `crossings`)
      node_of[(n, s)]   index of the copy of crossing n on stick s
      seg_u, seg_v      segment edges between copies, seg_L their length
      jun_u, jun_v      junction edges (one per crossing), jun_cross the crossing
      source_nodes, drain_nodes   copies tied to each terminal
      coords            (x, y) of every node (copies share the crossing's point)
      sticks_of[n]      the two sticks of crossing n
    """
    points = res["intersections_points"]
    graph = res["graph"]
    edges = _sd_component(res["filtered_segment_edge_list"], points)

    crossings = sorted({n for _, (n1, n2) in edges for n in (n1, n2)})
    node_of, coords, sticks_of = {}, [], {}
    for n in crossings:
        si, sj = graph[n]
        sticks_of[n] = (si, sj)
        for s in (si, sj):
            node_of[(n, s)] = len(coords)
            coords.append(tuple(points[n]))

    seg_u, seg_v, seg_L = [], [], []
    for s, (n1, n2) in edges:
        if s not in sticks_of[n1] or s not in sticks_of[n2]:
            raise ValueError(f"segment on stick {s} joins crossings {n1}, {n2} not on it")
        seg_u.append(node_of[(n1, s)]); seg_v.append(node_of[(n2, s)])
        seg_L.append(float(np.hypot(points[n1][0] - points[n2][0], points[n1][1] - points[n2][1])))

    jun_u = [node_of[(n, sticks_of[n][0])] for n in crossings]
    jun_v = [node_of[(n, sticks_of[n][1])] for n in crossings]

    src = [node_of[(n, s)] for n in crossings if points[n][0] < SOURCE_X for s in sticks_of[n]]
    drn = [node_of[(n, s)] for n in crossings if points[n][0] > DRAIN_X for s in sticks_of[n]]
    return dict(
        crossings=crossings, n_cross=len(crossings), node_of=node_of, coords=np.array(coords),
        sticks_of=sticks_of,
        seg_u=np.array(seg_u), seg_v=np.array(seg_v), seg_L=np.array(seg_L),
        jun_u=np.array(jun_u), jun_v=np.array(jun_v), jun_cross=np.array(crossings),
        source_nodes=np.array(src), drain_nodes=np.array(drn),
    )


def junction_resistances(net, R_j):
    """One resistance per crossing from a scalar, an array of length n_cross,
    or a callable f(stick_i, stick_j) -> resistance."""
    if callable(R_j):
        return np.array([float(R_j(*net["sticks_of"][n])) for n in net["crossings"]])
    R = np.asarray(R_j, dtype=float)
    if R.ndim == 0:
        return np.full(net["n_cross"], float(R))
    if R.shape != (net["n_cross"],):
        raise ValueError(f"R_j has shape {R.shape}, expected ({net['n_cross']},)")
    return R


def solve_two_terminal(net, R0, R_j, contract_junctions=False):
    """
    Effective resistance between the two terminals, and the current in every
    edge, with unit potential across the terminals.

    Parameters
    ----------
    net : dict
        Junction graph from ``build_junction_network``.
    R0 : float
        Stick resistance per unit length; segment e has R0 * seg_L[e].
    R_j : float, array_like or callable
        Junction resistances, see ``junction_resistances``. They must all be
        positive, or all exactly zero (treated as contract_junctions=True).
    contract_junctions : bool, optional
        Identify the two copies of each crossing (an ideal contact), which
        is the wire-limited model.

    Returns
    -------
    dict
        R_eff, I_total (current for unit voltage), V (per node of `net`),
        I_seg and I_jun (edge currents; I_jun is None when contracted),
        seg_R, jun_R, and the internal indices merged, s, d.

    Notes
    -----
    Source copies are held at V = 1 and drain copies at V = 0; the interior
    is solved with a sparse LU factorization (``scipy.sparse.linalg.splu``).
    Every node of `net` is connected to the source by construction
    (``_sd_component``), so the interior system is non-singular. A segment of
    zero length would give an infinite conductance.
    """
    n_nodes = len(net["coords"])
    seg_R = R0 * net["seg_L"]
    jun_R = junction_resistances(net, R_j)
    if not contract_junctions and np.any(jun_R <= 0):
        if np.all(jun_R == 0):
            contract_junctions = True          # R_j = 0 everywhere is the ideal contact
        else:
            raise ValueError("junction resistances must be positive (or all zero, an ideal contact)")

    # node -> merged index: each crossing's two copies are identified when the
    # junctions are contracted; every copy inside a strip collapses to its terminal
    label = np.arange(n_nodes)
    if contract_junctions:
        label[net["jun_v"]] = net["jun_u"]
    label[net["source_nodes"]] = -1
    label[net["drain_nodes"]] = -2
    uniq = {}
    merged = np.array([uniq.setdefault(int(l), len(uniq)) for l in label])
    s, d = uniq[-1], uniq[-2]
    N = len(uniq)

    if contract_junctions:
        eu, ev, eR = net["seg_u"], net["seg_v"], seg_R
    else:
        eu = np.concatenate([net["seg_u"], net["jun_u"]])
        ev = np.concatenate([net["seg_v"], net["jun_v"]])
        eR = np.concatenate([seg_R, jun_R])
    mu, mv = merged[eu], merged[ev]
    g = 1.0 / eR
    keep = mu != mv                      # edges inside a terminal carry nothing
    rows = np.concatenate([mu[keep], mv[keep], mu[keep], mv[keep]])
    cols = np.concatenate([mu[keep], mv[keep], mv[keep], mu[keep]])
    vals = np.concatenate([g[keep], g[keep], -g[keep], -g[keep]])
    L = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))

    # V_s = 1, V_d = 0, solve the interior
    interior = np.array([i for i in range(N) if i not in (s, d)])
    V = np.zeros(N); V[s] = 1.0
    if interior.size:
        Lii = L[interior][:, interior]
        b = -L[interior][:, [s]].toarray().ravel() * V[s]
        V[interior] = spla.splu(Lii.tocsc()).solve(b)
    I_edge = g * (V[mu] - V[mv]) * keep
    I_total = float(np.sum(I_edge[mu == s]) - np.sum(I_edge[mv == s]))
    R_eff = 1.0 / I_total
    n_seg = len(net["seg_u"])
    return dict(R_eff=R_eff, I_total=I_total, V=V[merged], I_seg=I_edge[:n_seg],
                I_jun=None if contract_junctions else I_edge[n_seg:],
                seg_R=seg_R, jun_R=jun_R, merged=merged, s=s, d=d)


def red_bonds(sol, rel_tol=1e-8):
    """Segments and junctions that carry the whole terminal current (the
    percolation definition of a red bond); a topological property of the
    junction graph, independent of the value of R_j > 0."""
    I = sol["I_total"]
    red_seg = np.abs(sol["I_seg"]) >= (1 - rel_tol) * I
    red_jun = None if sol["I_jun"] is None else np.abs(sol["I_jun"]) >= (1 - rel_tol) * I
    return red_seg, red_jun
