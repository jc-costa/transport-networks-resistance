"""
self_loops.py

Utilities to detect and remove self-loops from the intersection graph
built by `identify_intersections` (intersection_utils.py).

Context
-------
`graph` is a list of edges `(i, j)` where `i` and `j` are stick (segment)
indices in `range(N)`. Each entry of `graph` is index-aligned with the
corresponding entry of `intersections_points` (both are appended together,
once per valid intersection, inside `identify_intersections`).

A "self-loop" here means an edge `(i, i)`: a stick recorded as intersecting
itself. By construction, `identify_intersections` iterates with `j` in
`range(i + 1, N)`, so `i != j` always holds and self-loops never occur
naturally. Still, this module treats them as data-integrity issues worth
guarding against (e.g. manually edited graphs, merged/concatenated runs,
or any future change to the generation code), and provides:

- `detect_self_loops`: find self-loop entries without modifying anything.
- `remove_self_loops`: strip them out, keeping `graph` and
  `intersections_points` index-aligned.
- `inject_synthetic_self_loops`: TESTING ONLY helper to create self-loops
  on top of an otherwise clean, real network, so the two functions above
  can be exercised against something realistic.

Nothing in the existing pipeline (`point_generator.py`, `intersection_utils.py`,
`path_analysis.py`, `pruning.py`, `plotting.py`) is modified or depended upon
in a way that changes its behavior; this module only reads/copies the
`graph` / `intersections_points` structures it already produces.
"""

import random


def detect_self_loops(graph, verbose=True):
    """
    Find self-loop edges in `graph`.

    Parameters
    ----------
    graph : list of (i, j)
    verbose : bool
        Print a short report.

    Returns
    -------
    self_loop_indices : list of int
        Positions `k` in `graph` such that `graph[k][0] == graph[k][1]`.
    """
    self_loop_indices = [k for k, edge in enumerate(graph) if edge[0] == edge[1]]

    if verbose:
        if self_loop_indices:
            print(f"  Self-loops found: {len(self_loop_indices)} at indices {self_loop_indices}")
            for k in self_loop_indices:
                print(f"    graph[{k}] = {graph[k]}")
        else:
            print("  No self-loops found.")

    return self_loop_indices


def remove_self_loops(graph, intersections_points, verbose=True):
    """
    Remove self-loop edges from `graph`, dropping the aligned entry in
    `intersections_points` as well so both lists stay index-consistent.

    Parameters
    ----------
    graph : list of (i, j)
    intersections_points : list of (x, y), index-aligned with `graph`
    verbose : bool

    Returns
    -------
    filtered_graph : list
    filtered_intersections_points : list
    removed_indices : list of int
        Original indices (in the input `graph`) that were removed.
    """
    self_loop_indices = detect_self_loops(graph, verbose=verbose)

    if not self_loop_indices:
        return list(graph), list(intersections_points), []

    removed_set = set(self_loop_indices)
    filtered_graph = [edge for k, edge in enumerate(graph) if k not in removed_set]
    filtered_intersections_points = [
        pt for k, pt in enumerate(intersections_points) if k not in removed_set
    ]

    if verbose:
        print(f"  Removed {len(self_loop_indices)} self-loop(s). "
              f"graph: {len(graph)} -> {len(filtered_graph)} entries "
              f"| intersections_points: {len(intersections_points)} -> {len(filtered_intersections_points)} entries")

    return filtered_graph, filtered_intersections_points, self_loop_indices


def inject_synthetic_self_loops(graph, intersections_points, num_loops=2, seed=None):
    """
    TESTING UTILITY ONLY.

    Real networks built by `identify_intersections` never contain
    self-loops (see module docstring), so to validate `detect_self_loops`
    and `remove_self_loops` we need networks that *do* contain them.
    This function returns copies of `graph` / `intersections_points`
    with `num_loops` synthetic self-loop edges appended at the end.

    Parameters
    ----------
    graph : list of (i, j)
    intersections_points : list of (x, y), index-aligned with `graph`
    num_loops : int
        Number of synthetic self-loops to inject.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    graph_with_loops : list
    intersections_points_with_loops : list
    injected_indices : list of int
        Indices (in the returned lists) of the injected self-loops.
    """
    rng = random.Random(seed)
    graph_out = list(graph)
    points_out = list(intersections_points)
    injected_indices = []

    n_segments = max((max(edge) for edge in graph), default=-1) + 1
    if n_segments == 0:
        n_segments = 1  # fall back so we can still inject something on empty graphs

    for _ in range(num_loops):
        seg = rng.randrange(n_segments)
        graph_out.append((seg, seg))

        base_pt = rng.choice(points_out) if points_out else (0.5, 0.5)
        # Fabricate a coordinate close to an existing one purely so the
        # list has *some* point to keep index-alignment with graph_out;
        # this is synthetic test data, not a real intersection.
        points_out.append((base_pt[0], base_pt[1]))

        injected_indices.append(len(graph_out) - 1)

    return graph_out, points_out, injected_indices
