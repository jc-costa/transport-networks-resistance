"""
Loop-topology utilities for network manipulation.
Provides functions to add and remove subgraph-based "node-loops" — small
closed subgraphs (linear cycle, branched loop, chorded loop) attached to a
single host node.  These are electrically inert (like wire self-loops) but
consist of real, newly created nodes and edges.

This module is independent of the legacy wire-loop functions in
self_loop_utils.py (add_self_loop / remove_self_loops).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math

# ──────────────────────────────────────────────────────────────────────────
# Node placement helpers
# ──────────────────────────────────────────────────────────────────────────


def _loop_node_positions(
    host_x: float,
    host_y: float,
    n_nodes: int,
    topology: str,
    ring_size: int,
    branch_from: Optional[int] = None,
    n_branches: int = 0,
    direction: float = 1.0,
) -> List[List[float]]:
    """
    Compute (x, y) positions for new loop nodes.

    The ring is NOT centered on the host itself (that made loops overlap
    the host marker and look like a single highlighted dot). Instead, the
    ring is centered at a point offset vertically from the host by
    ``gap + R``, so the loop renders as a clearly separated shape
    connected to the host by two visible "stem" edges.

    Parameters
    ----------
    host_x, host_y : float  — host node coordinates.
    n_nodes         : int   — total new nodes to place.
    topology        : str   — 'linear', 'branched', or 'chorded'.
    ring_size       : int   — how many of the first *n_nodes* belong to the ring.
    branch_from     : int or None — index (0-based within ring) of the node
                                    branches sprout from (for branched).
    n_branches      : int   — number of branch nodes (for branched).
    direction       : float — +1.0 to place the loop above the host,
                              -1.0 to place it below. Useful to avoid
                              overlapping a neighbouring node/loop.

    Returns
    -------
    positions : list of [x, y]
    """
    R = 0.11                        # ring radius (was 0.045 — too small,
                                     # caused overlap with the host marker)
    gap = 0.17                       # extra clearance between host and ring
                                     # (was 0.09 — too small, made the
                                     # host-facing ring node look like a
                                     # short stub glued to the host marker)
    center_x = host_x
    center_y = host_y + direction * (R + gap)
    positions: List[List[float]] = []

    # --- Ring nodes placed on a circle around the OFFSET center ---
    if ring_size == 1:
        # A single ring node: place it straight at the offset center.
        # The two host<->node edges (drawn as curved parallels) already
        # read clearly as a loop once they're this far from the host.
        positions.append([center_x, center_y])
    else:
        # Start the ring at the angle pointing back toward the host, so
        # the first/last ring nodes sit closest to the host and the
        # stem edges look natural.
        #
        # Special case: with exactly 2 ring nodes, "evenly spaced on the
        # circle" forces them to be 180 degrees apart -- i.e. diametrically
        # opposite -- which puts BOTH of them on the same vertical line as
        # the host (since that line already passes through the circle's
        # center). The host's two connecting edges then overlap into what
        # looks like a single straight double line instead of a loop. A
        # small tilt breaks this degeneracy without affecting ring_size=1
        # or ring_size>=3, which don't have this exact-collinearity issue.
        tilt = 0.5 if ring_size == 2 else 0.0
        start_angle = (-math.pi / 2 + tilt) * direction
        for i in range(ring_size):
            angle = start_angle + 2.0 * math.pi * i / ring_size
            x = center_x + R * math.cos(angle)
            y = center_y + R * math.sin(angle)
            positions.append([x, y])

    # --- Branch nodes (only for branched topology) ---
    if topology == "branched" and branch_from is not None and n_branches > 0:
        base_i = min(branch_from, ring_size - 1)
        bx, by = positions[base_i]
        # Place branches radially outward relative to the HOST, not the ring
        # center. Using the ring center could point a branch back toward
        # the host (when the parent node sits on the host-facing side of
        # the ring), making it fold back and visually collide with the
        # loop / host instead of extending clearly away from it.
        base_angle = math.atan2(by - host_y, bx - host_x)
        for j in range(n_branches):
            # Offset perpendicular (90 deg) to the host->parent line. A
            # diagonal offset (e.g. ~50 deg) still pointed too close to the
            # direction of the other ring nodes, making the branch look
            # like it belonged to the same rounded loop contour instead of
            # a distinct pendant sticking out sideways.
            fixed_offset = math.pi / 2
            spread = fixed_offset + (j - (n_branches - 1) / 2.0) * 0.5
            off_angle = base_angle + spread
            r_ext = R * 2.2
            x = bx + r_ext * math.cos(off_angle)
            y = by + r_ext * math.sin(off_angle)
            positions.append([x, y])

    return positions


# ──────────────────────────────────────────────────────────────────────────
# add_node_loop  (single entry point)
# ──────────────────────────────────────────────────────────────────────────


def add_node_loop(
    intersections: List[List[float]],
    segment_to_nodes: Dict[int, Tuple[int, int]],
    graph: List[List[int]],
    host_node: int,
    n_loop_nodes: int = 3,
    topology: str = "linear",
    n_branch_nodes: int = 1,
    branch_from: Optional[int] = None,
    chord_pair: Optional[Tuple[int, int]] = None,
    direction: float = 1.0,
) -> Tuple[
    List[List[float]],
    Dict[int, Tuple[int, int]],
    List[List[int]],
    List[int],
]:
    """
    Inject a small subgraph (a "node-loop") attached to a single host node.

    Three topologies are supported:

    1. **linear** (default): a simple closed ring
       ``host -> n0 -> n1 -> ... -> n_{M-1} -> host``

    2. **branched**: a closed ring with pendant dead-end branches sprouting
       from one of the ring nodes.

    3. **chorded**: a closed ring with one extra "chord" edge connecting two
       non-adjacent ring nodes (requires ``M >= 3``).

    Parameters
    ----------
    intersections   : list of [x, y]  — node positions.
    segment_to_nodes: dict {edge_idx: (n1, n2)}.
    graph           : list of lists — graph[node] = list of edge indices.
    host_node       : int — existing node to attach the loop to.
    n_loop_nodes    : int — number of *new* nodes in the ring (M).  Default 3.
    topology        : 'linear', 'branched', or 'chorded'.
    n_branch_nodes  : int — number of pendant branch nodes (branched only).
    branch_from     : int or None — 0-based index **within the ring** of the
                      node that branches sprout from (branched only).
                      Default: 0 (first ring node, i.e. the one adjacent to host).
    chord_pair      : (int, int) or None — 0-based indices **within the ring**
                      of the two non-adjacent nodes to connect (chorded only).
                      Default: (0, 2) if M >= 3.
    direction       : float — +1.0 (default) places the loop above the host,
                      -1.0 places it below. Useful when injecting several
                      loops on the same or nearby hosts, to keep them from
                      overlapping each other visually.

    Returns
    -------
    new_intersections    : list of [x, y] (appended loop-node positions).
    new_segment_to_nodes : dict {edge_idx: (n1, n2)}.
    new_graph            : list of lists.
    loop_node_ids        : list of int — all newly created node indices.
    """
    # ── Validate ────────────────────────────────────────────────────────
    M = n_loop_nodes
    if M < 1:
        raise ValueError("n_loop_nodes must be >= 1")
    if topology not in ("linear", "branched", "chorded"):
        raise ValueError(f"Unknown topology: {topology}")
    if host_node < 0 or host_node >= len(intersections):
        raise IndexError(f"host_node {host_node} out of range")

    n_ring_nodes = M
    n_branch = 0
    if topology == "branched":
        n_branch = n_branch_nodes
        if branch_from is None:
            branch_from = 0
        if branch_from < 0 or branch_from >= M:
            raise ValueError(f"branch_from ({branch_from}) out of ring [0, {M - 1}]")

    if topology == "chorded" and M < 3:
        raise ValueError("Chorded topology requires n_loop_nodes >= 3")

    total_new_nodes = n_ring_nodes + n_branch
    ring_node_start = len(intersections)       # first new node index
    branch_node_start = ring_node_start + n_ring_nodes
    all_loop_node_ids = list(range(ring_node_start, ring_node_start + total_new_nodes))

    # ── Compute positions ───────────────────────────────────────────────
    hx, hy = intersections[host_node]
    new_positions = _loop_node_positions(
        hx, hy, total_new_nodes, topology,
        ring_size=n_ring_nodes,
        branch_from=branch_from,
        n_branches=n_branch,
        direction=direction,
    )

    # ── Build new data structures (copies) ──────────────────────────────
    new_intersections = list(intersections) + new_positions
    new_graph = [list(g) for g in graph] + [[] for _ in range(total_new_nodes)]
    new_seg = dict(segment_to_nodes)

    next_edge = max(new_seg.keys(), default=-1) + 1

    # Helper to add an edge
    def _add_edge(n1: int, n2: int) -> None:
        nonlocal next_edge
        edge_id = next_edge
        next_edge += 1
        new_seg[edge_id] = (n1, n2)
        new_graph[n1].append(edge_id)
        new_graph[n2].append(edge_id)

    # ── Build ring edges ─────────────────────────────────────────────────
    # host -> n0 -> n1 -> ... -> n_{M-1} -> host
    _add_edge(host_node, ring_node_start)          # host -> n0
    for i in range(M - 1):
        _add_edge(ring_node_start + i, ring_node_start + i + 1)   # ni -> n(i+1)
    _add_edge(ring_node_start + M - 1, host_node)  # n_{M-1} -> host

    # ── Extra topology edges ─────────────────────────────────────────────

    if topology == "branched":
        # Pendant branches: branch_from is a 0-based index within the ring
        parent_idx = branch_from  # ring index
        parent_node = ring_node_start + parent_idx
        for j in range(n_branch):
            branch_node = branch_node_start + j
            _add_edge(parent_node, branch_node)

    elif topology == "chorded":
        # Chord edge connecting two non-adjacent ring nodes
        if chord_pair is None:
            chord_pair = (0, 2) if M >= 3 else (0, 1)
        i, j = chord_pair
        # Ensure the pair is non-adjacent in the ring
        if abs(i - j) % M == 1 or abs(i - j) % M == M - 1:
            # Adjacent — skip; the edge already exists in the ring
            pass
        else:
            _add_edge(ring_node_start + i, ring_node_start + j)

    return new_intersections, new_seg, new_graph, all_loop_node_ids


# ──────────────────────────────────────────────────────────────────────────
# remove_node_loops
# ──────────────────────────────────────────────────────────────────────────


def remove_node_loops(
    intersections: List[List[float]],
    segment_to_nodes: Dict[int, Tuple[int, int]],
    graph: List[List[int]],
    loop_node_ids: List[int],
) -> Tuple[
    List[List[float]],
    Dict[int, Tuple[int, int]],
    List[List[int]],
]:
    """
    Strip all injected loop / branch / chord nodes and edges, then
    re-index nodes and edges back to a clean, consecutive baseline state.

    Parameters
    ----------
    intersections    : list of [x, y]  — node positions (including loop nodes).
    segment_to_nodes : dict {edge_idx: (n1, n2)} (including loop edges).
    graph            : list of lists — graph[node] = list of edge indices.
    loop_node_ids    : list of int — the nodes to remove (returned by
                       ``add_node_loop``).

    Returns
    -------
    clean_intersections    : list of [x, y]  (loop nodes removed).
    clean_segment_to_nodes : dict {new_edge_idx: (n1, n2)} (re-indexed from 0).
    clean_graph            : list of lists (re-indexed).
    """
    loop_set = set(loop_node_ids)

    # ── 1. Identify edges to remove (any incident to a loop node) ────────
    edges_to_remove = {
        e for e, (n1, n2) in segment_to_nodes.items()
        if n1 in loop_set or n2 in loop_set
    }

    # ── 2. Build mapping from old node indices to new (excluding loop) ────
    surviving_nodes = [i for i in range(len(intersections)) if i not in loop_set]
    old_to_new_node: Dict[int, int] = {
        old: new for new, old in enumerate(surviving_nodes)
    }

    # ── 3. Filter and re-index edges ──────────────────────────────────────
    surviving_edges = sorted(
        (e, (n1, n2))
        for e, (n1, n2) in segment_to_nodes.items()
        if e not in edges_to_remove
    )
    clean_seg: Dict[int, Tuple[int, int]] = {}
    for new_idx, (_, (n1, n2)) in enumerate(surviving_edges):
        clean_seg[new_idx] = (old_to_new_node[n1], old_to_new_node[n2])

    # ── 4. Rebuild graph ──────────────────────────────────────────────────
    n_surviving = len(surviving_nodes)
    clean_graph: List[List[int]] = [[] for _ in range(n_surviving)]
    for edge_idx, (n1, n2) in clean_seg.items():
        clean_graph[n1].append(edge_idx)
        clean_graph[n2].append(edge_idx)

    # ── 5. Rebuild intersections ──────────────────────────────────────────
    clean_intersections = [intersections[i] for i in surviving_nodes]

    return clean_intersections, clean_seg, clean_graph


# ──────────────────────────────────────────────────────────────────────────
# Convenience helper: check percolation (adapted from self_loop_utils)
# ──────────────────────────────────────────────────────────────────────────


def check_percolation_loops(
    segment_to_nodes: Dict[int, Tuple[int, int]],
    source_node: int,
    drain_node: int,
) -> bool:
    """
    BFS connectivity check: is *drain_node* reachable from *source_node*?

    Ignores self-loop edges (n, n) and handles arbitrary node indices
    (some may be isolated after loop removal — the BFS still works).
    """
    adj: Dict[int, set] = {}
    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            continue
        adj.setdefault(n1, set()).add(n2)
        adj.setdefault(n2, set()).add(n1)

    visited = {source_node}
    queue = [source_node]
    while queue:
        node = queue.pop(0)
        for nb in adj.get(node, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return drain_node in visited


# ──────────────────────────────────────────────────────────────────────────
# Persistence helper  (adapted save_self_loop_experiment for node-loops)
# ──────────────────────────────────────────────────────────────────────────


def save_node_loop_experiment(
    em,
    title: str,
    intersections,
    graph,
    seg_baseline,
    seg_with_loops,
    seg_cleaned,
    res_baseline,
    res_with_loops,
    res_cleaned,
    k: float,
    loop_description: str,
    loop_node_ids: Optional[List[int]] = None,
    src_node=None,
    drn_node=None,
    extra_fields=None,
):
    """
    Save a single node-loop test as one numbered experiment.

    Mirrors ``save_self_loop_experiment`` (from ``self_loop_utils.py``) but
    counts *node-loop nodes* rather than wire self-loops, and reports both
    counts in the summary.
    """
    n_loops = sum(1 for n1, n2 in seg_with_loops.values() if n1 == n2)
    R_base = res_baseline["impedance"]
    R_loop = res_with_loops["impedance"]
    R_clean = res_cleaned["impedance"]
    error_pct = (R_loop / R_base - 1) * 100 if R_base else None
    restored_ok = abs(R_clean - R_base) < 1e-4

    summary = {
        "title": title,
        "source_notebook": "test_self_loops_multi_node.ipynb",
        "timestamp": em.timestamp,
        "experiment_number": em.experiment_number,
        "k_ohm": float(k),
        "n_nodes": len(intersections),
        "n_edges_baseline": len(seg_baseline),
        "n_edges_with_loops": len(seg_with_loops),
        "n_self_loops": n_loops,
        "n_node_loop_nodes": len(loop_node_ids) if loop_node_ids else 0,
        "loop_description": loop_description,
        "R_baseline_ohm": float(R_base),
        "R_with_loops_ohm": float(R_loop),
        "R_cleaned_ohm": float(R_clean),
        "R_loop_over_baseline": float(R_loop / R_base) if R_base else None,
        "error_percent": float(error_pct) if error_pct is not None else None,
        "restoration_ok": bool(restored_ok),
        "voltages_baseline": [float(v) for v in res_baseline.get("voltages", [])],
        "voltages_with_loops": [float(v) for v in res_with_loops.get("voltages", [])],
        "voltages_cleaned": [float(v) for v in res_cleaned.get("voltages", [])],
    }
    if extra_fields:
        summary.update(extra_fields)

    def seg_to_jsonable(seg):
        return {str(k): list(v) for k, v in seg.items()}

    # Laplacian metadata
    if src_node is None or drn_node is None:
        xs = [p[0] for p in intersections]
        _src = src_node if src_node is not None else min(range(len(xs)), key=lambda i: xs[i])
        _drn = drn_node if drn_node is not None else max(range(len(xs)), key=lambda i: xs[i])
    else:
        _src, _drn = src_node, drn_node

    from laplacian_utils import compute_reduced_matrix

    def _laplacian_meta(seg, label):
        L, M, cond, is_sing = compute_reduced_matrix(intersections, seg, _src, _drn, k=k)
        return {
            "state": label,
            "n_nodes": len(intersections),
            "n_edges": len(seg),
            "condition_number": float(cond) if cond == cond else None,
            "is_singular": bool(is_sing),
            "eigenvalue_range": None,
        }

    laplacian_info = {
        "source_node": int(_src),
        "drain_node": int(_drn),
        "states": [
            _laplacian_meta(seg_baseline, "baseline"),
            _laplacian_meta(seg_with_loops, "with_loops"),
            _laplacian_meta(seg_cleaned, "cleaned"),
        ],
    }

    em.save_json(laplacian_info, "laplacian_info.json")
    em.save_json(summary, "experiment_summary.json")
    em.save_json(intersections, "intersections.json")
    em.save_json(graph, "graph.json")
    em.save_json(seg_to_jsonable(seg_baseline), "segment_to_nodes_baseline.json")
    em.save_json(seg_to_jsonable(seg_with_loops), "segment_to_nodes_with_loops.json")
    em.save_json(seg_to_jsonable(seg_cleaned), "segment_to_nodes_cleaned.json")

    direction = "\u2192"
    check = "\u2713" if restored_ok else "\u2717"
    print(f"  R: {R_base:.2f} {direction} {R_loop:.2f} {direction} {R_clean:.2f} Ohm  "
          f"| error={error_pct:+.1f}%  | restored={check}")

    print(f"  Saving -> {em.experiment_dir}")
    print(f"  {title}")
