"""
Minimal 3D counterpart to network_visualization.py's
plot_network_colored_by_resistance. Only the plain "nodes + straight edges,
colored by a scalar" pattern is ported: network_visualization.py's curved
parallel-edge Bezier bowing (built from a 2D perpendicular vector) and its
self-loop arcs (matplotlib FancyArrowPatch/connectionstyle, a 2D-canvas-only
API) have no unique 3D generalization and are intentionally not attempted
here -- edges that overlap in 3D are simply drawn as straight, overlapping
lines for now.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection


def plot_network_3d_colored_by_resistance(intersections, segment_to_nodes, edge_resistances,
                                          title=None, figsize=(8, 7), cmap_name='viridis'):
    """
    Plot a 3D network with each edge colored by its resistance value and a
    colorbar -- 3D analogue of plot_network_colored_by_resistance.

    Args:
        intersections: List of [x, y, z] node coordinates
        segment_to_nodes: dict edge_idx -> (n1, n2)
        edge_resistances: array of per-edge resistance values (Ohm)
        title: Optional custom title
        figsize: Figure size tuple
        cmap_name: Matplotlib colormap name

    Returns:
        Matplotlib figure object, or None if there are no finite resistances to plot
    """
    intersections = np.asarray(intersections, dtype=float)
    edge_resistances = np.asarray(edge_resistances, dtype=float)
    finite = edge_resistances[np.isfinite(edge_resistances)]
    if len(finite) == 0:
        print("[WARNING] No finite edge resistances to plot")
        return None

    norm = plt.Normalize(vmin=finite.min(), vmax=finite.max())
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    for e, (n1, n2) in segment_to_nodes.items():
        if n1 == n2:
            continue
        x = [intersections[n1][0], intersections[n2][0]]
        y = [intersections[n1][1], intersections[n2][1]]
        z = [intersections[n1][2], intersections[n2][2]]
        ax.plot(x, y, z, color=cmap(norm(edge_resistances[e])), linewidth=2)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Edge resistance (Ω)')

    ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2],
              c='black', s=12)
    ax.set_title(title or 'Network colored by edge resistance', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.tight_layout()

    return fig


def plot_pre_merge_geometry_3d(res, R0=1000.0, title="", figsize=(6.5, 5.5)):
    """
    Plot the real pre-merge, post-pruning geometry (res["intersections_points"]
    + res["filtered_segment_edge_list"]), colored by distance-dependent
    resistance -- unlike plot_network_3d_colored_by_resistance (which plots
    the *merged* network used for the circuit solve), this preserves the
    actual spatial layout even when several edges connect source directly to
    drain and would otherwise collapse into one merged edge with a
    conductance multiplier (e.g. parallel-rail networks).
    """
    inter = np.array(res["intersections_points"])
    edges = {i: e for i, (_, e) in enumerate(res["filtered_segment_edge_list"])}
    resistances = np.array([R0 * np.linalg.norm(inter[n1] - inter[n2]) for n1, n2 in edges.values()])

    if resistances.max() > resistances.min():
        norm = plt.Normalize(vmin=resistances.min(), vmax=resistances.max())
    else:
        norm = plt.Normalize(vmin=0, vmax=resistances.max() + 1e-9)
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    for (n1, n2), r in zip(edges.values(), resistances):
        ax.plot([inter[n1, 0], inter[n2, 0]], [inter[n1, 1], inter[n2, 1]], [inter[n1, 2], inter[n2, 2]],
                color=cmap(norm(r)), linewidth=2)
    ax.scatter(inter[:, 0], inter[:, 1], inter[:, 2], c='black', s=20)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label='Edge resistance (Ω)')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _blank_3d_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def plot_pruning_comparison_grid_3d(results, titles=None, suptitle="Before pruning (blue=kept, red=pruned dead end)",
                                    panel_size=(3.0, 3.2)):
    """
    Small-multiples version of plot_pruning_comparison_3d: one panel per
    network in `results`, each colored the same way (kept vs pruned dead-end
    edges) -- for comparing several independent random realizations at once.
    """
    n = len(results)
    titles = titles or [""] * n
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]),
                             subplot_kw={"projection": "3d"})
    axes = np.atleast_1d(axes)
    for ax, res, title in zip(axes, results, titles):
        inter = np.array(res["intersections_points"])
        edges_all = {i: e for i, (_, e) in enumerate(res["segment_edge_list"])}
        edges_kept = {tuple(sorted(e)) for _, e in res["filtered_segment_edge_list"]}
        for n1, n2 in edges_all.values():
            kept = tuple(sorted((n1, n2))) in edges_kept
            ax.plot([inter[n1, 0], inter[n2, 0]], [inter[n1, 1], inter[n2, 1]], [inter[n1, 2], inter[n2, 2]],
                    color="tab:blue" if kept else "tab:red", linewidth=1.3)
        ax.scatter(inter[:, 0], inter[:, 1], inter[:, 2], c="black", s=6)
        ax.set_title(title, fontsize=9)
        _blank_3d_axes(ax)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_self_loop_grid_3d(results, titles=None,
                           suptitle="Pre-merge self-loop candidates (orange) -- both endpoints in the same boundary region",
                           panel_size=(3.0, 3.2)):
    """
    Small-multiples plot highlighting, for each network in `results`, the
    pre-merge edges (res["pre_merge_segment_to_nodes"]) that will become
    self-loops once merge_boundary_nodes collapses the source/drain regions
    -- see pipeline_utils_3d.identify_self_loop_edges.
    """
    from pipeline_utils_3d import identify_self_loop_edges

    n = len(results)
    titles = titles or [""] * n
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]),
                             subplot_kw={"projection": "3d"})
    axes = np.atleast_1d(axes)
    for ax, res, title in zip(axes, results, titles):
        inter = np.array(res["pre_merge_intersections"])
        loop_edges = {tuple(sorted(e)) for _, e in identify_self_loop_edges(res)}
        for n1, n2 in res["pre_merge_segment_to_nodes"].values():
            is_loop = tuple(sorted((n1, n2))) in loop_edges
            ax.plot([inter[n1, 0], inter[n2, 0]], [inter[n1, 1], inter[n2, 1]], [inter[n1, 2], inter[n2, 2]],
                    color="tab:orange" if is_loop else "tab:blue", linewidth=1.3)
        ax.scatter(inter[:, 0], inter[:, 1], inter[:, 2], c="black", s=6)
        ax.set_title(title, fontsize=9)
        _blank_3d_axes(ax)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_merged_network_grid_3d(results, titles=None, suptitle="After (final circuit)", panel_size=(3.0, 3.2)):
    """
    Small-multiples plot of the final merged/pruned network for several
    results at once (single color per edge -- no per-edge resistance needed),
    for a compact "after" comparison alongside plot_pruning_comparison_grid_3d
    / plot_self_loop_grid_3d.
    """
    n = len(results)
    titles = titles or [""] * n
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]),
                             subplot_kw={"projection": "3d"})
    axes = np.atleast_1d(axes)
    for ax, res, title in zip(axes, results, titles):
        inter = np.array(res["merged_intersections"])
        for n1, n2 in res["merged_segment_to_nodes"].values():
            if n1 == n2:
                continue
            ax.plot([inter[n1, 0], inter[n2, 0]], [inter[n1, 1], inter[n2, 1]], [inter[n1, 2], inter[n2, 2]],
                    color="tab:blue", linewidth=1.3)
        ax.scatter(inter[:, 0], inter[:, 1], inter[:, 2], c="black", s=6)
        ax.set_title(title, fontsize=9)
        _blank_3d_axes(ax)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_pruning_comparison_3d(res, title="Before pruning", figsize=(6, 5)):
    """
    Plot every raw (pre-pruning) contact/edge, coloring the edges that survive
    pruning (blue) differently from the ones removed as dead ends (red) --
    illustrates prune_dead_end_nodes' effect for a hand-built network.
    """
    inter = np.array(res["intersections_points"])
    edges_all = {i: e for i, (_, e) in enumerate(res["segment_edge_list"])}
    edges_kept = {tuple(sorted(e)) for _, e in res["filtered_segment_edge_list"]}

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    for n1, n2 in edges_all.values():
        kept = tuple(sorted((n1, n2))) in edges_kept
        ax.plot([inter[n1, 0], inter[n2, 0]], [inter[n1, 1], inter[n2, 1]], [inter[n1, 2], inter[n2, 2]],
                color="tab:blue" if kept else "tab:red", linewidth=2,
                label="kept" if kept else "pruned (dead end)")
    ax.scatter(inter[:, 0], inter[:, 1], inter[:, 2], c="black", s=25)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_network_3d_colored_by_voltage(intersections, segment_to_nodes, node_voltages,
                                       terminal_nodes=None, terminal_currents=None, title=None,
                                       figsize=(8, 7), cmap_name='coolwarm'):
    """
    Plot a multi-terminal 3D network with each NODE colored by its solved
    voltage (edges drawn in a neutral color) -- companion to
    plot_network_3d_colored_by_resistance, which instead colors edges by a
    per-edge scalar. Terminal nodes (if given) are drawn larger with a black
    outline so the prescribed-voltage nodes are visually identifiable among
    the internal (solved) ones.

    Voltage color alone can't show terminal ROLE: a terminal's voltage is
    prescribed, but whether it ends up sourcing or draining current is a
    computed consequence of the whole network's topology (see
    multi_terminal_solver.summarize_terminal_roles). When terminal_currents
    is given, terminal markers are additionally split by shape -- up-triangle
    for SOURCE (current flows out into the network), down-triangle for DRAIN
    (current flows in from the network) -- so role is visible on the plot
    itself, not just in solve['terminal_currents'].

    Args:
        intersections: List of [x, y, z] node coordinates
        segment_to_nodes: dict edge_idx -> (n1, n2)
        node_voltages: array of per-node voltage values (Volt), length n_nodes
        terminal_nodes: optional list of node indices to highlight
        terminal_currents: optional {node: net_current_into_node}, i.e.
            solve_multi_terminal's 'terminal_currents' -- enables the
            source/drain marker-shape split described above
        title: Optional custom title
        figsize: Figure size tuple
        cmap_name: Matplotlib colormap name (diverging by default, since
            voltage differences -- not just magnitude -- are the point)

    Returns:
        Matplotlib figure object
    """
    intersections = np.asarray(intersections, dtype=float)
    node_voltages = np.asarray(node_voltages, dtype=float)
    terminal_set = set(terminal_nodes or [])

    vmax = np.max(np.abs(node_voltages)) if len(node_voltages) else 1.0
    norm = plt.Normalize(vmin=-vmax, vmax=vmax) if vmax > 0 else plt.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            continue
        ax.plot([intersections[n1][0], intersections[n2][0]],
                [intersections[n1][1], intersections[n2][1]],
                [intersections[n1][2], intersections[n2][2]],
                color='gray', alpha=0.5, linewidth=1)

    internal_idx = [i for i in range(len(intersections)) if i not in terminal_set]
    if internal_idx:
        ax.scatter(intersections[internal_idx, 0], intersections[internal_idx, 1], intersections[internal_idx, 2],
                  c=node_voltages[internal_idx], cmap=cmap, norm=norm, s=20)
    if terminal_set:
        term_idx = sorted(terminal_set)
        if terminal_currents:
            source_idx = [n for n in term_idx if terminal_currents.get(n, 0) < 0]
            drain_idx = [n for n in term_idx if terminal_currents.get(n, 0) >= 0]
            if source_idx:
                ax.scatter(intersections[source_idx, 0], intersections[source_idx, 1], intersections[source_idx, 2],
                          c=node_voltages[source_idx], cmap=cmap, norm=norm, s=150, marker='^',
                          edgecolors='black', linewidths=1.5, depthshade=False, label='Source (current out)')
            if drain_idx:
                ax.scatter(intersections[drain_idx, 0], intersections[drain_idx, 1], intersections[drain_idx, 2],
                          c=node_voltages[drain_idx], cmap=cmap, norm=norm, s=150, marker='v',
                          edgecolors='black', linewidths=1.5, depthshade=False, label='Drain (current in)')
            ax.legend(loc='upper left', fontsize=9)
        else:
            ax.scatter(intersections[term_idx, 0], intersections[term_idx, 1], intersections[term_idx, 2],
                      c=node_voltages[term_idx], cmap=cmap, norm=norm, s=90,
                      edgecolors='black', linewidths=1.5, depthshade=False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Voltage (V)')

    subtitle = 'large ▲/▼ = source/drain terminals' if terminal_currents else 'large markers = terminals'
    ax.set_title(title or f'Network colored by voltage ({subtitle})', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.tight_layout()

    return fig
