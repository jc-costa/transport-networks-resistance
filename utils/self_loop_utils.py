"""
Self-loop utilities for network manipulation.
Provides functions to add, remove, count, and check percolation in presence of self-loops.
"""

from typing import Dict, Tuple, List
from laplacian_utils import print_laplacian


def add_self_loop(segment_to_nodes: Dict[int, Tuple[int, int]],
                  graph: List[List[int]],
                  node_id: int) -> Tuple[Dict[int, Tuple[int, int]], List[List[int]]]:
    """
    Insert a self-loop edge (node_id, node_id) into the network.

    Parameters
    ----------
    segment_to_nodes : dict {edge_idx: (n1, n2)}
    graph : list of lists, graph[i] = list of edge indices at node i
    node_id : int, target node for the self-loop

    Returns
    -------
    new_segment_to_nodes, new_graph  (copies — originals not modified)
    """
    new_seg = dict(segment_to_nodes)
    new_graph = [list(g) for g in graph]
    new_edge_idx = max(new_seg.keys(), default=-1) + 1
    new_seg[new_edge_idx] = (node_id, node_id)
    new_graph[node_id].append(new_edge_idx)
    return new_seg, new_graph


def remove_self_loops(segment_to_nodes: Dict[int, Tuple[int, int]],
                      graph: List[List[int]]) -> Tuple[Dict[int, Tuple[int, int]], List[List[int]]]:
    """
    Remove every self-loop edge (i, i) and re-index edge keys from 0.

    Parameters
    ----------
    segment_to_nodes : dict {edge_idx: (n1, n2)}
    graph : list of lists

    Returns
    -------
    clean_seg, clean_graph  (copies — originals not modified)
    """
    # Filter out self-loops
    self_loop_indices = {e for e, (n1, n2) in segment_to_nodes.items() if n1 == n2}
    real_edges = [(e, (n1, n2)) for e, (n1, n2) in sorted(segment_to_nodes.items())
                  if e not in self_loop_indices]

    # Rebuild with consecutive indices starting from 0
    clean_seg = {new_idx: edge for new_idx, (_, edge) in enumerate(real_edges)}

    # Rebuild graph from scratch
    n_nodes = len(graph)
    clean_graph = [[] for _ in range(n_nodes)]
    for edge_idx, (n1, n2) in clean_seg.items():
        clean_graph[n1].append(edge_idx)
        clean_graph[n2].append(edge_idx)

    return clean_seg, clean_graph


def count_self_loops(segment_to_nodes: Dict[int, Tuple[int, int]]) -> int:
    """Return the number of self-loop edges in segment_to_nodes."""
    return sum(1 for n1, n2 in segment_to_nodes.values() if n1 == n2)


def check_percolation(segment_to_nodes: Dict[int, Tuple[int, int]],
                      source_node: int, drain_node: int) -> bool:
    """
    BFS connectivity check: is drain_node reachable from source_node?

    Self-loops (n, n) are silently ignored by BFS (they add n to its own
    neighbour set, which the visited-check skips on the first encounter).
    """
    adj = {}
    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            continue  # self-loop: irrelevant to connectivity
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

# ────────────────────────────────────────────────────────────────────────────
# Experiment saving for self-loop tests
# ────────────────────────────────────────────────────────────────────────────

def save_self_loop_experiment(
    em, title, intersections, graph,
    seg_baseline, seg_with_loops, seg_cleaned,
    res_baseline, res_with_loops, res_cleaned,
    k, loop_description, src_node=None, drn_node=None, extra_fields=None,
):
    """
    Save a single self-loop test as one numbered experiment.

    Parameters
    ----------
    em               : ExperimentManager instance
    title            : human-readable experiment name
    intersections    : list of [x, y] node positions
    graph            : adjacency list
    seg_baseline     : segment_to_nodes dict
    seg_with_loops   : segment_to_nodes dict with self-loops injected
    seg_cleaned      : segment_to_nodes dict after remove_self_loops
    res_baseline     : ImpedanceCalculator result dict (baseline state)
    res_with_loops   : ImpedanceCalculator result dict (looped state)
    res_cleaned      : ImpedanceCalculator result dict (cleaned state)
    k                : resistance per segment [Ohm]
    loop_description : short string describing which loops were added
    extra_fields     : optional dict merged into experiment_summary.json
    """
    n_loops = sum(1 for n1, n2 in seg_with_loops.values() if n1 == n2)
    R_base  = res_baseline['impedance']
    R_loop  = res_with_loops['impedance']
    R_clean = res_cleaned['impedance']
    error_pct = (R_loop / R_base - 1) * 100 if R_base else None
    restored_ok = abs(R_clean - R_base) < 1e-4

    summary = {
        "title":               title,
        "source_notebook":     "test_self_loops.ipynb",
        "timestamp":           em.timestamp,
        "experiment_number":   em.experiment_number,
        "k_ohm":               float(k),
        "n_nodes":             len(intersections),
        "n_edges_baseline":    len(seg_baseline),
        "n_edges_with_loops":  len(seg_with_loops),
        "n_self_loops":        n_loops,
        "loop_description":    loop_description,
        "R_baseline_ohm":      float(R_base),
        "R_with_loops_ohm":    float(R_loop),
        "R_cleaned_ohm":       float(R_clean),
        "R_loop_over_baseline": float(R_loop / R_base) if R_base else None,
        "error_percent":       float(error_pct) if error_pct is not None else None,
        "restoration_ok":      bool(restored_ok),
        "voltages_baseline":   [float(v) for v in res_baseline.get("voltages", [])],
        "voltages_with_loops": [float(v) for v in res_with_loops.get("voltages", [])],
        "voltages_cleaned":    [float(v) for v in res_cleaned.get("voltages", [])],
    }
    if extra_fields:
        summary.update(extra_fields)

    def seg_to_jsonable(seg):
        return {str(k): list(v) for k, v in seg.items()}

    
    # ── Laplacian metadata ─────────────────────────────────────────────────
    # Use min-x / max-x as source/drain if not explicitly provided
    if src_node is None or drn_node is None:
        xs = [p[0] for p in intersections]
        _src = src_node if src_node is not None else min(range(len(xs)), key=lambda i: xs[i])
        _drn = drn_node if drn_node is not None else max(range(len(xs)), key=lambda i: xs[i])
    else:
        _src, _drn = src_node, drn_node

    from laplacian_utils import compute_reduced_matrix

    def _laplacian_meta(seg, label):
        """Compute Laplacian metadata dict for a given state."""
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
    em.save_json(summary,                         "experiment_summary.json")
    em.save_json(intersections,                   "intersections.json")
    em.save_json(graph,                           "graph.json")
    em.save_json(seg_to_jsonable(seg_baseline),   "segment_to_nodes_baseline.json")
    em.save_json(seg_to_jsonable(seg_with_loops), "segment_to_nodes_with_loops.json")
    em.save_json(seg_to_jsonable(seg_cleaned),    "segment_to_nodes_cleaned.json")

    direction = "\u2192"
    check = "\u2713" if restored_ok else "\u2717"
    print(f"  R: {R_base:.2f} {direction} {R_loop:.2f} {direction} {R_clean:.2f} Ohm  "
          f"| error={error_pct:+.1f}%  | restored={check}")

    print(f"  Saving -> {em.experiment_dir}")
    print(f"  {title}")


# ── Self-loop test runner ──────────────────────────────────────────────

def run_self_loop_test(intersections, graph, seg, src, drn, k,
                       loop_nodes, label="", verbose=True):
    """
    Run a complete self-loop test: baseline -> inject loops -> cleanup.

    Parameters
    ----------
    intersections : list of [x, y] node positions
    graph : list of lists, adjacency
    seg : dict, segment_to_nodes (baseline)
    src, drn : int, source and drain node indices
    k : float, resistance per segment
    loop_nodes : list of int, node IDs to inject one self-loop each
    label : str, test label for output

    Returns
    -------
    dict with keys: 'baseline', 'looped', 'cleaned'
      each containing: 'result' (ImpedanceCalculator result),
                       'seg', 'graph', 'impedance', 'voltages'
    """
    from pipeline_utils import run_impedance
    from io import StringIO
    from contextlib import redirect_stdout

    # Baseline
    with redirect_stdout(StringIO()):
        print_laplacian(intersections, seg, src, drn, k=k)
    res_base = run_impedance(intersections, graph, seg, k=k,
                             label=f"{label} baseline" if verbose else "")

    # Inject loops
    seg_loop, graph_loop = seg, graph
    for node in loop_nodes:
        seg_loop, graph_loop = add_self_loop(seg_loop, graph_loop, node)

    with redirect_stdout(StringIO()):
        print_laplacian(intersections, seg_loop, src, drn, k=k)
    res_loop = run_impedance(intersections, graph_loop, seg_loop, k=k,
                             label=f"{label} looped" if verbose else "")

    # Cleanup
    seg_clean, graph_clean = remove_self_loops(seg_loop, graph_loop)
    res_clean = run_impedance(intersections, graph_clean, seg_clean, k=k,
                              label=f"{label} cleaned" if verbose else "")

    return {
        'baseline': {'result': res_base, 'seg': seg, 'graph': graph,
                     'impedance': res_base['impedance'],
                     'voltages': res_base['voltages']},
        'looped': {'result': res_loop, 'seg': seg_loop, 'graph': graph_loop,
                   'impedance': res_loop['impedance'],
                   'voltages': res_loop['voltages']},
        'cleaned': {'result': res_clean, 'seg': seg_clean, 'graph': graph_clean,
                    'impedance': res_clean['impedance'],
                    'voltages': res_clean['voltages']},
    }

