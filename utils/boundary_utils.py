"""Utilities for strict source/drain boundary handling."""


def get_boundary_nodes(intersections_points, source_threshold=0.1, drain_threshold=0.9):
    """Return source/drain node indices using strict x-threshold regions."""
    source_nodes = []
    drain_nodes = []

    for idx, point in enumerate(intersections_points):
        x = point[0]
        if x < source_threshold:
            source_nodes.append(idx)
        elif x > drain_threshold:
            drain_nodes.append(idx)

    return source_nodes, drain_nodes


def validate_strict_boundary_nodes(source_nodes, drain_nodes, context=""):
    """Raise ValueError when strict boundary conditions are not satisfied."""
    if source_nodes and drain_nodes:
        return

    ctx = f" [{context}]" if context else ""
    raise ValueError(
        "Strict boundary conditions not satisfied"
        f"{ctx}: source_nodes={len(source_nodes)}, drain_nodes={len(drain_nodes)}. "
        "No nearest-node fallback is allowed."
    )


def get_boundary_segments(graph, source_nodes, drain_nodes):
    """Collect segment IDs touching source and drain boundary nodes."""
    source_segments = set()
    drain_segments = set()

    for node_idx in source_nodes:
        if node_idx < 0 or node_idx >= len(graph):
            raise IndexError(f"Source boundary node {node_idx} out of graph bounds")
        source_segments.update(graph[node_idx])

    for node_idx in drain_nodes:
        if node_idx < 0 or node_idx >= len(graph):
            raise IndexError(f"Drain boundary node {node_idx} out of graph bounds")
        drain_segments.update(graph[node_idx])

    return source_segments, drain_segments
