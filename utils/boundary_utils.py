"""Utilities for strict source/drain boundary handling."""


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

