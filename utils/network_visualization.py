# ============================================================================
# NETWORK VISUALIZATION
# Functions for visualizing random networks, intersections, paths, and pruning
# ============================================================================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Arc, Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
import numpy as np

def plot_random_lines(points_1, points_2, N, title=None, figsize=(8, 6)):
    """
    Plot all randomly generated lines.
    
    Args:
        points_1: List of start points [(x, y), ...]
        points_2: List of end points [(x, y), ...]
        N: Number of lines
        title: Optional custom title
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    for i in range(N):
        plt.plot([points_1[i][0], points_2[i][0]], 
                 [points_1[i][1], points_2[i][1]], 
                 linestyle='-', color='steelblue')
    
    plt.title(title or f'N = {N} - Random Lines', fontsize=22)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle='dotted', linewidth=0.5, color='gray')
    
    return fig


def plot_intersections(points_1, points_2, N, intersections_points, 
                      intersected_line_1, intersected_line_2,
                      title=None, figsize=(8, 6)):
    """
    Plot lines with intersection points highlighted.
    
    Args:
        points_1, points_2: Start/end points of all lines
        N: Number of lines
        intersections_points: List of intersection points [(x, y), ...]
        intersected_line_1, intersected_line_2: Intersected line endpoints
        title: Optional custom title
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object or None if no intersections
    """
    if not intersections_points:
        print("No intersection points to plot")
        return None
    
    fig = plt.figure(figsize=figsize)
    
    # Plot all lines
    for i in range(N):
        plt.plot([points_1[i][0], points_2[i][0]], 
                 [points_1[i][1], points_2[i][1]], 
                 linestyle='-', color='steelblue', zorder=0)
    
    # Plot intersected lines in red
    for i in range(len(intersected_line_1)):
        plt.plot([intersected_line_1[i][0], intersected_line_2[i][0]], 
                 [intersected_line_1[i][1], intersected_line_2[i][1]], 
                 linestyle='-', color='red', zorder=1)
    
    # Plot intersection points
    x_values, y_values = zip(*intersections_points)
    plt.scatter(x_values, y_values, c='black', marker='o', s=6, zorder=2)
    
    plt.title(title or f'N = {N} - Intersections', fontsize=22)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])
    
    return fig


def plot_percolating_paths_before_pruning(intersections_points, graph,
                                          all_path_perc, path_nodes,
                                          points_1, points_2,
                                          N, title=None, figsize=(8, 6)):
    """
    Plot percolating paths before pruning using consecutive-only edge discovery.

    Args:
        intersections_points: List of all intersection points
        graph: graph[i] = list of stick indices at node i
        all_path_perc: List of percolation status for each path (1=percolating)
        path_nodes: List of node lists for each path
        points_1, points_2: Stick endpoints
        N: Number of lines (for title)
        title: Optional custom title
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object or None if no percolating paths
    """
    from edge_discovery import discover_consecutive_edges

    if not intersections_points or not any(p == 1 for p in all_path_perc):
        print("No percolating paths to plot")
        return None

    fig = plt.figure(figsize=figsize)

    x_values, y_values = zip(*intersections_points)

    all_perc_nodes = set()
    for k, perc in enumerate(all_path_perc):
        if perc == 1:
            all_perc_nodes.update(path_nodes[k])

    segment_edge_list = discover_consecutive_edges(
        graph, intersections_points, all_perc_nodes, points_1, points_2
    )

    all_edges = set((min(n1, n2), max(n1, n2)) for seg, (n1, n2) in segment_edge_list)

    for n1, n2 in all_edges:
        plt.plot([x_values[n1], x_values[n2]], [y_values[n1], y_values[n2]],
                 linestyle='-', color='green', zorder=1)

    plt.scatter(x_values, y_values, c='black', marker='o', s=14, zorder=3)

    plt.title(title or f'N = {N} - Percolating Paths (Before Pruning)', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    return fig


def plot_before_after_pruning(segment_edge_list, filtered_segment_edge_list, 
                             intersections_points, N, figsize=(16, 8)):
    """
    Plot side-by-side visualization of network before and after pruning.
    Clean style: no per-node text annotations.

    Returns (fig, stats) where stats contains before/after counts.
    """
    if not segment_edge_list or not filtered_segment_edge_list:
        print("[WARNING] Nothing to visualize")
        return None, {}

    edges_before_set = set((min(n1, n2), max(n1, n2)) for seg, (n1, n2) in segment_edge_list)
    edges_after_set = set((min(n1, n2), max(n1, n2)) for seg, (n1, n2) in filtered_segment_edge_list)
    removed_edges = edges_before_set - edges_after_set

    nodes_before = set()
    for seg, (n1, n2) in segment_edge_list:
        nodes_before.add(n1)
        nodes_before.add(n2)
    nodes_after = set()
    for seg, (n1, n2) in filtered_segment_edge_list:
        nodes_after.add(n1)
        nodes_after.add(n2)
    removed_nodes = nodes_before - nodes_after

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- LEFT: BEFORE PRUNING ---
    ax1 = axes[0]
    ax1.set_title(f'BEFORE Pruning\n{len(edges_before_set)} edges, {len(nodes_before)} nodes',
                  fontsize=12, fontweight='bold')

    for seg, (n1, n2) in segment_edge_list:
        x1, y1 = intersections_points[n1]
        x2, y2 = intersections_points[n2]
        edge_key = (min(n1, n2), max(n1, n2))
        if edge_key in removed_edges:
            ax1.plot([x1, x2], [y1, y2], color='red', linewidth=1.5, linestyle='--', alpha=0.6, zorder=1)
        else:
            ax1.plot([x1, x2], [y1, y2], color='steelblue', linewidth=1.5, alpha=0.8, zorder=1)

    for node in nodes_before:
        x, y = intersections_points[node]
        is_source = x < 0.1
        is_drain = x > 0.9
        will_be_removed = node in removed_nodes
        if is_source:
            ax1.scatter(x, y, c='green', s=200, marker='s', zorder=4,
                       edgecolors='darkgreen', linewidths=1.5)
        elif is_drain:
            ax1.scatter(x, y, c='red', s=200, marker='^', zorder=4,
                       edgecolors='darkred', linewidths=1.5)
        elif will_be_removed:
            ax1.scatter(x, y, c='orange', s=80, marker='x', zorder=3, linewidths=2)
        else:
            ax1.scatter(x, y, c='steelblue', s=60, marker='o', zorder=3,
                       edgecolors='navy', linewidths=0.8)

    n_src = sum(1 for n in nodes_before if intersections_points[n][0] < 0.1)
    n_drn = sum(1 for n in nodes_before if intersections_points[n][0] > 0.9)
    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=2, label=f'Kept edges ({len(edges_after_set)})'),
        Line2D([0], [0], color='red', lw=1.5, ls='--', label=f'Removed edges ({len(removed_edges)})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10,
               markeredgecolor='darkgreen', markeredgewidth=1.5, label=f'Source nodes ({n_src})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='darkred', markeredgewidth=1.5, label=f'Drain nodes ({n_drn})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8,
               markeredgecolor='navy', label='Internal nodes'),
        Line2D([0], [0], marker='x', color='orange', markersize=10, lw=0,
               label=f'Removed nodes ({len(removed_nodes)})'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)
    ax1.axvline(x=0.1, color='green', linestyle=':', alpha=0.4, linewidth=1.5)
    ax1.axvline(x=0.9, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # --- RIGHT: AFTER PRUNING ---
    ax2 = axes[1]
    ax2.set_title(f'AFTER Pruning\n{len(edges_after_set)} edges, {len(nodes_after)} nodes',
                  fontsize=12, fontweight='bold')

    for seg, (n1, n2) in filtered_segment_edge_list:
        x1, y1 = intersections_points[n1]
        x2, y2 = intersections_points[n2]
        ax2.plot([x1, x2], [y1, y2], color='steelblue', linewidth=1.5, alpha=0.8, zorder=1)

    for node in nodes_after:
        x, y = intersections_points[node]
        if x < 0.1:
            ax2.scatter(x, y, c='green', s=200, marker='s', zorder=4,
                       edgecolors='darkgreen', linewidths=1.5)
        elif x > 0.9:
            ax2.scatter(x, y, c='red', s=200, marker='^', zorder=4,
                       edgecolors='darkred', linewidths=1.5)
        else:
            ax2.scatter(x, y, c='steelblue', s=60, marker='o', zorder=3,
                       edgecolors='navy', linewidths=0.8)

    ax2.axvline(x=0.1, color='green', linestyle=':', alpha=0.4, linewidth=1.5)
    ax2.axvline(x=0.9, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    stats = {
        'edges_before': len(edges_before_set),
        'edges_after': len(edges_after_set),
        'nodes_before': len(nodes_before),
        'nodes_after': len(nodes_after),
        'removed_edges': removed_edges,
        'removed_nodes': removed_nodes,
    }

    return fig, stats


def plot_network_with_impedance(merged_intersections, merged_graph, result, 
                                merged_source_node, merged_drain_node,
                                nodes_merged_to_source, nodes_merged_to_drain,
                                figsize=(14, 10)):
    """
    Plot the merged network with impedance calculation results (voltages and currents).
    
    Args:
        merged_intersections: List of node positions after merge
        merged_graph: List of edges after merge
        result: Dictionary with 'voltages', 'currents', 'impedance'
        merged_source_node: Index of merged source node
        merged_drain_node: Index of merged drain node
        nodes_merged_to_source: List of original nodes merged to source
        nodes_merged_to_drain: List of original nodes merged to drain
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object or None if no result
    """
    if not result or not merged_intersections:
        print("[WARNING] No impedance result to visualize")
        return None
    
    voltages = result.get('voltages', [])
    currents = result.get('currents', [])
    impedance = result.get('impedance', 0)
    
    fig = plt.figure(figsize=figsize)
    
    # Normalize currents for line width
    if len(currents) > 0:
        max_current = np.max(np.abs(currents))
        if max_current > 0:
            normalized_currents = np.abs(currents) / max_current
        else:
            normalized_currents = np.zeros_like(currents)
    else:
        normalized_currents = []
    
    # Plot edges with width proportional to current
    for idx, edge in enumerate(merged_graph):
        n1, n2 = edge[0], edge[1]
        x1, y1 = merged_intersections[n1]
        x2, y2 = merged_intersections[n2]
        
        if idx < len(normalized_currents):
            lw = 1 + 5 * normalized_currents[idx]
        else:
            lw = 1
        
        plt.plot([x1, x2], [y1, y2], color='gray', linewidth=lw, alpha=0.7, zorder=1)
    
    # Plot nodes with color based on voltage
    for i, (x, y) in enumerate(merged_intersections):
        if i == merged_source_node:
            color = 'green'
            marker = 's'
            size = 200
            label = 'Source (V=1)'
        elif i == merged_drain_node:
            color = 'red'
            marker = '^'
            size = 200
            label = 'Drain (V=0)'
        else:
            # Color by voltage
            v = voltages[i] if i < len(voltages) else 0.5
            color = plt.cm.coolwarm(v)
            marker = 'o'
            size = 100
            label = None
        
        plt.scatter(x, y, c=[color], s=size, marker=marker, zorder=3, 
                   edgecolors='black', linewidths=0.5)
    
    plt.title(f'Network with Impedance\nR = {impedance:.4f} Ohm', fontsize=16, fontweight='bold')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axvline(x=0.1, color='green', linestyle=':', alpha=0.5)
    plt.axvline(x=0.9, color='red', linestyle=':', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar for voltage
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Voltage (V)', fontsize=10)
    
    return fig


def plot_before_after_merge(filtered_segment_edge_list, filtered_nodes,
                            intersections_points, merged_intersections,
                            merged_segment_to_nodes, merged_source_node,
                            merged_drain_node, edge_conductance, result,
                            k, figsize=(18, 8)):
    """
    Plot BEFORE vs AFTER merge with current/voltage colormaps.

    Returns the matplotlib figure.
    """
    import matplotlib.colors as mcolors

    if not result or not merged_intersections:
        print("No impedance result to visualize")
        return None

    source_global = [n for n in filtered_nodes if intersections_points[n][0] < 0.1]
    drain_global = [n for n in filtered_nodes if intersections_points[n][0] > 0.9]
    internal_global = [n for n in filtered_nodes if n not in source_global and n not in drain_global]

    voltages = result['voltages']
    currents = result['currents']
    max_current = max(abs(c) for c in currents) if len(currents) > 0 else 1

    # Fix positions for after-merge plot
    viz_positions = list(merged_intersections)
    source_ys = [intersections_points[n][1] for n in source_global]
    drain_ys = [intersections_points[n][1] for n in drain_global]
    viz_positions[merged_source_node] = (
        0.1, sum(source_ys) / len(source_ys) if source_ys else 0.5)
    viz_positions[merged_drain_node] = (
        0.9, sum(drain_ys) / len(drain_ys) if drain_ys else 0.5)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.03, 0.03], wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax_current = fig.add_subplot(gs[0, 2])
    cax_voltage = fig.add_subplot(gs[0, 3])

    # --- LEFT: BEFORE MERGE ---
    ax1.set_title(f'BEFORE MERGE\n{len(set(e for s, e in filtered_segment_edge_list))} edges, '
                  f'{len(filtered_nodes)} nodes', fontsize=11, fontweight='bold')

    for seg, (n1, n2) in filtered_segment_edge_list:
        x1, y1 = intersections_points[n1]
        x2, y2 = intersections_points[n2]
        ax1.plot([x1, x2], [y1, y2], color='steelblue', linewidth=1.5, alpha=0.7, zorder=1)

    for node in filtered_nodes:
        x, y = intersections_points[node]
        if node in source_global:
            ax1.scatter(x, y, c='green', s=200, marker='s', zorder=4,
                       edgecolors='darkgreen', linewidths=1.5)
        elif node in drain_global:
            ax1.scatter(x, y, c='red', s=200, marker='^', zorder=4,
                       edgecolors='darkred', linewidths=1.5)
        else:
            ax1.scatter(x, y, c='steelblue', s=50, marker='o', zorder=3,
                       edgecolors='navy', linewidths=0.8)

    legend1 = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10,
               markeredgecolor='darkgreen', markeredgewidth=1.5,
               label=f'Source ({len(source_global)} nodes)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='darkred', markeredgewidth=1.5,
               label=f'Drain ({len(drain_global)} nodes)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8,
               markeredgecolor='navy',
               label=f'Internal ({len(internal_global)} nodes)'),
    ]
    ax1.legend(handles=legend1, loc='upper left', fontsize=8)
    ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
    ax1.axvline(x=0.9, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # --- RIGHT: AFTER MERGE ---
    ax2.set_title(f'AFTER MERGE\n{len(merged_segment_to_nodes)} edges, '
                  f'{len(merged_intersections)} nodes\n'
                  f'R_eq = {result["impedance"]:.2f} ohm',
                  fontsize=11, fontweight='bold')

    current_cmap = plt.cm.plasma
    current_norm = mcolors.Normalize(vmin=0, vmax=max_current)

    for seg_idx, (n1, n2) in merged_segment_to_nodes.items():
        x1, y1 = viz_positions[n1]
        x2, y2 = viz_positions[n2]
        current_mag = abs(currents[seg_idx])
        color = current_cmap(current_norm(current_mag))
        lw = 1.5 + 3.0 * (current_mag / max_current)
        ax2.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.85, zorder=1)

    voltage_cmap = plt.cm.coolwarm
    voltage_norm = mcolors.Normalize(vmin=0, vmax=1)

    for i in range(len(merged_intersections)):
        x, y = viz_positions[i]
        v = voltages[i]
        v_color = voltage_cmap(voltage_norm(v))
        if i == merged_source_node:
            ax2.scatter(x, y, c=[v_color], s=350, marker='s', zorder=4,
                       edgecolors='darkgreen', linewidths=2.5)
            ax2.annotate('S', (x, y), textcoords="offset points",
                        xytext=(-12, 0), fontsize=11, fontweight='bold', color='darkgreen',
                        ha='center', va='center')
        elif i == merged_drain_node:
            ax2.scatter(x, y, c=[v_color], s=350, marker='^', zorder=4,
                       edgecolors='darkred', linewidths=2.5)
            ax2.annotate('D', (x, y), textcoords="offset points",
                        xytext=(12, 0), fontsize=11, fontweight='bold', color='darkred',
                        ha='center', va='center')
        else:
            ax2.scatter(x, y, c=[v_color], s=60, marker='o', zorder=3,
                       edgecolors='black', linewidths=0.8)

    legend2 = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10,
               markeredgecolor='darkgreen', markeredgewidth=2, label='Source (merged)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='darkred', markeredgewidth=2, label='Drain (merged)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', label='Internal node'),
    ]
    ax2.legend(handles=legend2, loc='upper left', fontsize=8)
    ax2.axvline(x=0.1, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
    ax2.axvline(x=0.9, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Colorbars
    sm_current = plt.cm.ScalarMappable(cmap=current_cmap, norm=current_norm)
    sm_current.set_array([])
    cbar_c = fig.colorbar(sm_current, cax=cax_current)
    cbar_c.set_label('|I| (A)', fontsize=9)

    sm_voltage = plt.cm.ScalarMappable(cmap=voltage_cmap, norm=voltage_norm)
    sm_voltage.set_array([])
    cbar_v = fig.colorbar(sm_voltage, cax=cax_voltage)
    cbar_v.set_label('V (V)', fontsize=9)

    return fig



# def draw_self_loop(ax, x, y, count, xlim=None, ylim=None):
#     """
#     Draw one or more self-loop arcs attached to a node at (x, y)
#     using FancyArrowPatch with arc3 connection style.

#     Parameters:
#     - ax: matplotlib axes
#     - x, y: node coordinates
#     - count: number of self-loops (>=1)
#     - xlim, ylim: unused (kept for backwards compatibility)

#     Returns (label_x, label_y) for placing the label.
#     """
#     if count <= 0:
#         return x, y

#     base_rad = 0.2
#     for i in range(count):
#         rad = base_rad + i * 0.05
#         ls = ['-', '--', ':'][min(i, 2)]
#         lw = max(0.8, 2.0 - i * 0.3)
#         loop = FancyArrowPatch(
#             (x, y), (x, y),
#             arrowstyle='-',
#             connectionstyle=f"arc3,rad={rad}",
#             color='orange', lw=lw, zorder=6, linestyle=ls
#         )
#         ax.add_patch(loop)

#     label_x = x
#     label_y = y + 0.08 + count * 0.03
#     return label_x, label_y

# def draw_self_loop(ax, x, y, count, xlim=None, ylim=None):
#     """
#     Draw one or more self-loop arcs above the node using Arc patches.
#     Returns (label_x, label_y) for placing the label.
#     """
#     if count <= 0:
#         return x, y

#     # Ajuste a altura do arco conforme o número de loops
#     height_step = 0.12   # altura total do arco (será dobrada internamente)
#     for i in range(count):
#         h = height_step * (i + 1)          # altura do arco
#         # Arco com centro no nó, largura fixa 0.12, altura h
#         arc = Arc((x, y), width=0.12, height=h,
#                   angle=0, theta1=0, theta2=180,
#                   color='orange', lw=2.0 - i*0.3, zorder=6,
#                   linestyle=['-', '--', ':'][min(i,2)])
#         ax.add_patch(arc)

#     # Posição do label: acima do nó, com deslocamento vertical acumulado
#     label_y = y + 0.08 + count * 0.03
#     return x, label_y


def draw_self_loop(ax, x, y, count, x_pos=None):
    """Draw self‑loops as closed teardrop shapes attached to the node.

    Uses FancyArrowPatch with arc3 connection style and a small
    perturbation between start/end so matplotlib renders the patch
    (identical start/end produces a degenerate patch that is dropped).

    Direction:
      - Interior nodes: loop bulges upward.
      - Source nodes (x < 0.1): loop bulges left (outside the graph).
      - Drain nodes (x > 0.9): loop bulges right (outside the graph).
    Multiple loops on the same node are stacked with increasing rad.

    Returns (label_x, label_y) for the label text.
    """
    if count <= 0:
        return x, y

    eps = 0.04           # vertical anchor separation = 0.08
    d = 2 * eps          # distance between perturbed points

    is_source = x_pos is not None and x_pos < 0.1
    is_drain = x_pos is not None and x_pos > 0.9

    # Perturbation direction & rad sign to produce the desired bulge
    if is_source:
        # Vertical perturbation → arc bulges horizontally.
        # Negative rad bulges LEFT (outside the graph).
        start, end = (x, y - eps), (x, y + eps)
        rad_sign = -1          # → rad = -4.0
        base_rad = 4.0
        step_rad = 1.7
        label_x = x - 0.20
        label_y = y
    elif is_drain:
        # Vertical perturbation → arc bulges horizontally.
        # Positive rad bulges RIGHT (outside the graph).
        start, end = (x, y - eps), (x, y + eps)
        rad_sign = 1           # → rad = +4.0
        base_rad = 4.0
        step_rad = 1.7
        label_x = x + 0.20
        label_y = y
    else:
        # Interior: horizontal perturbation → arc bulges upward.
        # Negative rad bulges UP.
        start, end = (x - eps, y), (x + eps, y)
        rad_sign = -1
        base_rad = 3.3
        step_rad = 1.7
        # Label above the outermost loop
        max_bulge = (base_rad + (count - 1) * step_rad) * d
        label_x = x
        label_y = y + max_bulge + 0.04

    for i in range(count):
        rad = base_rad + i * step_rad   # 4.0/3.3, 5.7/5.0, 7.4/6.7, …
        loop = FancyArrowPatch(
            start, end,
            arrowstyle='-',
            connectionstyle=f"arc3,rad={rad_sign * rad}",
            color='orange', lw=2.5, zorder=6,
            linestyle=['-', '--', ':'][min(i, 2)],
        )
        ax.add_patch(loop)

    return label_x, label_y


def draw_chain_panel(ax, intersections, segment_to_nodes, voltages,
                     source_node, drain_node, title,
                     node_labels=None, y_mid=0.50):
    """
    Draw a horizontal‑chain network with nodes coloured by voltage.
    Self‑loops are rendered as arcs above the node (stacked if multiple).
    """
    nx = [p[0] for p in intersections]
    ny = [y_mid] * len(intersections)

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.05, 0.95)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=7)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)
    ax.axvline(0.1, ls='--', lw=0.7, color='steelblue', alpha=0.4)
    ax.axvline(0.9, ls='--', lw=0.7, color='tomato', alpha=0.4)

    # Count self-loops per node
    loop_count = {}
    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            loop_count[n1] = loop_count.get(n1, 0) + 1
    
    print(f"[DEBUG] self‑loops encontrados: {loop_count}")
    for node, cnt in loop_count.items():
        print(f"   nó {node} tem {cnt} self‑loop(s) in chain panel")

    # Adjust y‑limits to make room for self‑loop arcs above the chain
    if loop_count:
        max_loops = max(loop_count.values(), default=0)
        ax.set_ylim(0.05, 0.95 + 0.1 * max_loops)   # extra space for the arcs

    # Draw real edges
    for n1, n2 in segment_to_nodes.values():
        if n1 != n2:
            ax.plot([nx[n1], nx[n2]], [y_mid, y_mid], 'k-', lw=2, zorder=2)

    # Draw self-loop arcs (directional concentric arcs)
    # for node, count in loop_count.items():
    #     lx, ly = draw_self_loop(ax, nx[node], ny[node], count)
    #     lbl = f"⚡" + (f"×{count}" if count > 1 else "")
    #     ax.text(lx, ly, lbl, ha='center', fontsize=7, color='orange')
    for node, count in loop_count.items():
        lx, ly = draw_self_loop(ax, nx[node], y_mid, count, x_pos=nx[node])
        lbl = f"⚡" + (f"×{count}" if count > 1 else "")
        ax.text(lx, ly, lbl, ha='center', fontsize=7, color='orange')

    # Draw nodes coloured by voltage
    cmap_v = plt.get_cmap("RdYlGn")
    for i, (vx, v) in enumerate(zip(nx, voltages)):
        color = cmap_v(v)
        ax.scatter(vx, y_mid, s=260, c=[color], zorder=5,
                   edgecolors='k', lw=1.2)
        ax.text(vx, y_mid + 0.22, f"V={v:.3f}",
                ha='center', fontsize=7.5,
                bbox=dict(fc='white', alpha=0.85, pad=1, edgecolor='none'))
        lbl = (node_labels or {}).get(i, f"N{i}")
        ax.text(vx, y_mid - 0.18, lbl, ha='center', fontsize=8,
                color='navy', fontweight='bold')


# def draw_ladder_panel(ax, intersections, segment_to_nodes, voltages,
#                       source_node, drain_node, title, node_labels=None):
#     """
#     Draw a 2‑D network (ladder) with nodes at their actual (x, y) positions.
#     Self‑loops are rendered as small circles near the node.
#     """
#     nx = [p[0] for p in intersections]
#     ny = [p[1] for p in intersections]

#     ax.set_xlim(-0.08, 1.08)
#     ax.set_ylim(-0.05, 1.05)
#     ax.set_xticks([0.0, 0.5, 1.0])
#     ax.set_xticklabels(["0", "0.5", "1"], fontsize=7)
#     ax.set_yticks([])
#     ax.set_title(title, fontsize=9, pad=3)
#     ax.axvline(0.1, ls='--', lw=0.7, color='steelblue', alpha=0.4)
#     ax.axvline(0.9, ls='--', lw=0.7, color='tomato', alpha=0.4)

#     loop_count = {}
#     for n1, n2 in segment_to_nodes.values():
#         if n1 == n2:
#             loop_count[n1] = loop_count.get(n1, 0) + 1

#     # Draw real edges
#     for n1, n2 in segment_to_nodes.values():
#         if n1 != n2:
#             ax.plot([nx[n1], nx[n2]], [ny[n1], ny[n2]], 'k-', lw=1.8, zorder=2)

#     # Draw self-loop arcs (directional concentric arcs)
#     for node, count in loop_count.items():
#         lx, ly = draw_self_loop(ax, nx[node], ny[node], count)
#         lbl = f"⚡" + (f"×{count}" if count > 1 else "")
#         ax.text(lx, ly, lbl, ha='center', fontsize=7, color='orange')

#     cmap_v = plt.get_cmap("RdYlGn")
#     for i, (vx, vy, v) in enumerate(zip(nx, ny, voltages)):
#         color = cmap_v(v)
#         ax.scatter(vx, vy, s=240, c=[color], zorder=5, edgecolors='k', lw=1.2)
#         ax.text(vx, vy + 0.10, f"V={v:.3f}", ha='center', fontsize=7.5,
#                 bbox=dict(fc='white', alpha=0.85, pad=1, edgecolor='none'))
#         lbl = (node_labels or {}).get(i, f"N{i}")
#         ax.text(vx, vy - 0.12, lbl, ha='center', fontsize=8,
#                 color='navy', fontweight='bold')


def draw_self_loop_ladder(ax, x, y, count, x_pos=None):
    """Draw self‑loop teardrops attached to a node (ladder topology).

    Same approach as draw_self_loop: FancyArrowPatch + arc3 +
    perturbation to force rendering.  Direction choice:
      - Interior: upward.
      - Source (x < 0.1): leftward (outside the graph).
      - Drain (x > 0.9): rightward (outside the graph).
    Multiple loops are stacked with increasing rad.

    Returns (label_x, label_y) for the label text.
    """
    if count <= 0:
        return x, y

    eps = 0.04
    d = 2 * eps

    is_source = x_pos is not None and x_pos < 0.1
    is_drain = x_pos is not None and x_pos > 0.9

    if is_source:
        start, end = (x, y - eps), (x, y + eps)
        rad_sign = -1             # → rad = -4.0, bulges LEFT
        base_rad = 4.0
        step_rad = 1.7
        label_x = x - 0.20
        label_y = y
    elif is_drain:
        start, end = (x, y - eps), (x, y + eps)
        rad_sign = 1              # → rad = +4.0, bulges RIGHT
        base_rad = 4.0
        step_rad = 1.7
        label_x = x + 0.20
        label_y = y
    else:
        start, end = (x - eps, y), (x + eps, y)
        rad_sign = -1             # bulges UP
        base_rad = 3.3
        step_rad = 1.7
        max_bulge = (base_rad + (count - 1) * step_rad) * d
        label_x = x
        label_y = y + max_bulge + 0.04

    for i in range(count):
        rad = base_rad + i * step_rad
        loop = FancyArrowPatch(
            start, end,
            arrowstyle='-',
            connectionstyle=f"arc3,rad={rad_sign * rad}",
            color='orange', lw=2.5, zorder=6,
            linestyle=['-', '--', ':'][min(i, 2)],
        )
        ax.add_patch(loop)

    return label_x, label_y


def draw_ladder_panel(ax, intersections, segment_to_nodes, voltages,
                      source_node, drain_node, title, node_labels=None):
    """
    Draw a 2‑D network (ladder) with nodes at their actual (x, y) positions.
    Self‑loops are rendered as small arcs offset to the right of the node
    (stacked with increasing radius for multiple loops on the same node).
    """
    nx = [p[0] for p in intersections]
    ny = [p[1] for p in intersections]

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=7)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)
    ax.axvline(0.1, ls='--', lw=0.7, color='steelblue', alpha=0.4)
    ax.axvline(0.9, ls='--', lw=0.7, color='tomato', alpha=0.4)

    # Count self-loops per node
    loop_count = {}
    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            loop_count[n1] = loop_count.get(n1, 0) + 1

    print(f"[DEBUG] self‑loops encontrados: {loop_count}")
    for node, cnt in loop_count.items():
        print(f"   nó {node} tem {cnt} self‑loop(s) in ladder panel")
    # Draw real edges
    for n1, n2 in segment_to_nodes.values():
        if n1 != n2:
            ax.plot([nx[n1], nx[n2]], [ny[n1], ny[n2]], 'k-', lw=1.8, zorder=2)

    # Draw self-loop arcs (directional concentric arcs offset to the right)
    for node, count in loop_count.items():
        lx, ly = draw_self_loop_ladder(ax, nx[node], ny[node], count, x_pos=nx[node])
        lbl = f"⚡" + (f"×{count}" if count > 1 else "")
        ax.text(lx, ly, lbl, ha='center', fontsize=7, color='orange')

    # Draw nodes coloured by voltage
    cmap_v = plt.get_cmap("RdYlGn")
    for i, (vx, vy, v) in enumerate(zip(nx, ny, voltages)):
        color = cmap_v(v)
        ax.scatter(vx, vy, s=240, c=[color], zorder=5, edgecolors='k', lw=1.2)
        ax.text(vx, vy + 0.10, f"V={v:.3f}", ha='center', fontsize=7.5,
                bbox=dict(fc='white', alpha=0.85, pad=1, edgecolor='none'))
        lbl = (node_labels or {}).get(i, f"N{i}")
        ax.text(vx, vy - 0.12, lbl, ha='center', fontsize=8,
                color='navy', fontweight='bold')


def plot_small_network_with_impedance(intersections, segment_to_nodes, result,
                                      source_node, drain_node, title=None,
                                      figsize=None, show_resistance_labels=True):
    """
    Visualize a small (few-node) test network together with its impedance
    result: edge width ~ |current|, edge label = resistance (Ohm), node color
    = voltage, source/drain marked distinctly. Unlike plot_network_with_impedance,
    this is not tied to the [0,1] percolation domain, so it works for the
    hand-built series/parallel/triangle style test cases.

    Args:
        intersections: List of [x, y] node coordinates
        segment_to_nodes: dict edge_idx -> (n1, n2)
        result: dict from ImpedanceCalculator.calculate() (voltages, currents,
            edge_resistances, impedance)
        source_node, drain_node: node indices
        title: Optional custom title (defaults to "R_total = ... Ohm")
        figsize: Figure size tuple. If None, sized automatically from the
            network's own aspect ratio (a flat line gets a short figure
            instead of a mostly-empty square one).
        show_resistance_labels: If True, annotate each edge with its resistance

    Returns:
        Matplotlib figure object
    """
    intersections = np.asarray(intersections, dtype=float)
    voltages = result.get('voltages', np.zeros(len(intersections)))
    currents = result.get('currents', np.zeros(len(segment_to_nodes)))
    edge_resistances = result.get('edge_resistances', None)
    impedance = result.get('impedance', float('nan'))

    xs, ys = intersections[:, 0], intersections[:, 1]
    diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())) or 1.0
    label_offset = 0.07 * diag

    if figsize is None:
        width = 7.0
        data_aspect = (ys.max() - ys.min() + 1e-9) / (xs.max() - xs.min() + 1e-9)
        figsize = (width, float(np.clip(width * data_aspect + 2.0, 3.2, 7.0)))

    fig, ax = plt.subplots(figsize=figsize)

    # Group edges by node pair so parallel (multi-)edges can be curved apart
    # instead of being drawn on top of each other.
    pair_edges = {}
    for e, (n1, n2) in segment_to_nodes.items():
        if n1 == n2:
            continue
        key = (min(n1, n2), max(n1, n2))
        pair_edges.setdefault(key, []).append(e)

    max_current = np.max(np.abs(currents)) if len(currents) else 0.0
    max_reach = 0.0  # tracks how far curves/labels stray from the node bounding box

    for (n1, n2), edges in pair_edges.items():
        x1, y1 = intersections[n1]
        x2, y2 = intersections[n2]
        seg_len = np.hypot(x2 - x1, y2 - y1) or 1.0
        perp = np.array([-(y2 - y1), x2 - x1]) / seg_len
        midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        multi = len(edges) > 1

        for j, e in enumerate(edges):
            lw = 1.5 + 4 * (abs(currents[e]) / max_current) if max_current > 0 else 1.5
            # bulge = how far the control point sits off the straight line; the
            # actual curve (quadratic Bezier) only reaches half that at its peak.
            bulge = 0.45 * seg_len * (j - (len(edges) - 1) / 2.0) if multi else 0.0
            curve_peak = midpoint + perp * (bulge * 0.5)

            if multi:
                control = midpoint + perp * bulge
                path = Path([(x1, y1), tuple(control), (x2, y2)],
                           [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                ax.add_patch(PathPatch(path, facecolor='none', edgecolor='gray',
                                     linewidth=lw, zorder=1))
            else:
                ax.plot([x1, x2], [y1, y2], color='gray', linewidth=lw, zorder=1)

            if show_resistance_labels and edge_resistances is not None:
                sign = 1.0 if bulge >= 0 else -1.0
                label_pos = curve_peak + perp * (sign * label_offset)
                max_reach = max(max_reach, float(np.max(np.abs(label_pos - midpoint))))
                r_e = edge_resistances[e]
                label = f"{r_e:.0f} Ω" if np.isfinite(r_e) else "0 Ω"
                ax.annotate(label, tuple(label_pos), fontsize=8.5, color='darkred', ha='center',
                           va='center', zorder=4,
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    # Node markers with the source/drain identity and voltage captioned inline,
    # instead of a floating legend that can land on top of the network.
    for i, (x, y) in enumerate(intersections):
        if i == source_node:
            color, marker, size = 'green', 's', 220
            label = f"Source\nV={voltages[i]:.2f}"
        elif i == drain_node:
            color, marker, size = 'red', '^', 220
            label = f"Drain\nV={voltages[i]:.2f}"
        else:
            v = voltages[i] if i < len(voltages) else 0.5
            color, marker, size = plt.cm.coolwarm(v), 'o', 140
            label = f"N{i}\nV={v:.2f}"
        ax.scatter(x, y, c=[color], marker=marker, s=size, zorder=5,
                  edgecolors='black', linewidths=0.6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 15),
                   fontsize=7.5, ha='center', color='black', zorder=6,
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    ax.set_title(title or f"R_total = {impedance:.2f} Ω", fontsize=13, pad=12)

    pad_x = max(0.35, 0.3 * (xs.max() - xs.min() + 1e-9), max_reach + 0.15)
    pad_y = max(0.4, 0.35 * (ys.max() - ys.min() + 1e-9), max_reach + 0.15)
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    fig.tight_layout()

    return fig


def plot_network_colored_by_resistance(intersections, segment_to_nodes, edge_resistances,
                                       title=None, figsize=(8, 6), cmap_name='viridis'):
    """
    Plot a (real, many-edge) network with each edge colored by its resistance
    value and a colorbar -- for inspecting where a distance-dependent model
    puts its large/small resistors across a whole random stick network.

    Args:
        intersections: List of [x, y] node coordinates
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

    fig, ax = plt.subplots(figsize=figsize)
    for e, (n1, n2) in segment_to_nodes.items():
        if n1 == n2:
            continue
        x = [intersections[n1][0], intersections[n2][0]]
        y = [intersections[n1][1], intersections[n2][1]]
        ax.plot(x, y, color=cmap(norm(edge_resistances[e])), linewidth=2)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Edge resistance (Ω)')

    ax.scatter(intersections[:, 0], intersections[:, 1], c='black', s=12, zorder=3)
    ax.set_title(title or 'Network colored by edge resistance', fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    return fig


def plot_network_colored_by_voltage(intersections, segment_to_nodes, node_voltages,
                                    terminal_nodes=None, terminal_currents=None, title=None,
                                    figsize=(8, 6), cmap_name='coolwarm', show_node_ids=False):
    """
    2D sibling of network_visualization_3d.py's
    plot_network_3d_colored_by_voltage: plot a multi-terminal network with
    each NODE colored by its solved voltage (edges drawn in a neutral
    color). Terminal nodes (if given) are drawn larger with a black outline
    so the prescribed-voltage nodes are visually identifiable among the
    internal (solved) ones.

    Voltage color alone can't show terminal ROLE: a terminal's voltage is
    prescribed, but whether it ends up sourcing or draining current is a
    computed consequence of the whole network's topology (see
    multi_terminal_solver.summarize_terminal_roles). When terminal_currents
    is given, terminal markers are additionally split by shape -- up-triangle
    for SOURCE (current flows out into the network), down-triangle for DRAIN
    (current flows in from the network) -- so role is visible on the plot
    itself, not just in solve['terminal_currents'].

    Args:
        intersections: List of [x, y] node coordinates
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
        show_node_ids: if True, annotate every node with its index, so a
            probe result plot doubles as the node-ID reference for picking
            the next pair (off by default: other callers of this function
            don't expect labels, and a dense network gets cluttered fast)

    Returns:
        Matplotlib figure object
    """
    intersections = np.asarray(intersections, dtype=float)
    node_voltages = np.asarray(node_voltages, dtype=float)
    terminal_set = set(terminal_nodes or [])

    vmax = np.max(np.abs(node_voltages)) if len(node_voltages) else 1.0
    norm = plt.Normalize(vmin=-vmax, vmax=vmax) if vmax > 0 else plt.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            continue
        ax.plot([intersections[n1][0], intersections[n2][0]],
                [intersections[n1][1], intersections[n2][1]],
                color='gray', alpha=0.5, linewidth=1)

    internal_idx = [i for i in range(len(intersections)) if i not in terminal_set]
    if internal_idx:
        ax.scatter(intersections[internal_idx, 0], intersections[internal_idx, 1],
                  c=node_voltages[internal_idx], cmap=cmap, norm=norm, s=20, zorder=3)
    if terminal_set:
        term_idx = sorted(terminal_set)
        if terminal_currents:
            source_idx = [n for n in term_idx if terminal_currents.get(n, 0) < 0]
            drain_idx = [n for n in term_idx if terminal_currents.get(n, 0) >= 0]
            if source_idx:
                ax.scatter(intersections[source_idx, 0], intersections[source_idx, 1],
                          c=node_voltages[source_idx], cmap=cmap, norm=norm, s=160, marker='^',
                          edgecolors='black', linewidths=1.5, zorder=4, label='Source (current out)')
            if drain_idx:
                ax.scatter(intersections[drain_idx, 0], intersections[drain_idx, 1],
                          c=node_voltages[drain_idx], cmap=cmap, norm=norm, s=160, marker='v',
                          edgecolors='black', linewidths=1.5, zorder=4, label='Drain (current in)')
            ax.legend(loc='best', fontsize=9)
        else:
            ax.scatter(intersections[term_idx, 0], intersections[term_idx, 1],
                      c=node_voltages[term_idx], cmap=cmap, norm=norm, s=90,
                      edgecolors='black', linewidths=1.5, zorder=4)

    if show_node_ids:
        for i, (x, y) in enumerate(intersections):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(3, 3),
                       fontsize=7, zorder=5)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Voltage (V)')

    subtitle = 'large ▲/▼ = source/drain terminals' if terminal_currents else 'large markers = terminals'
    ax.set_title(title or f'Network colored by voltage ({subtitle})', fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    return fig


def plot_r_eff_vs_distance(records, reference_node=None, title=None, figsize=(8, 6),
                           log_y=True):
    """
    Plot the R_eff(reference_node, j) vs. separation-distance curve produced
    by multimeter.run_onion_protocol(): one point per probed node j, grouped
    by onion layer, plus the per-layer mean +/- std as a guide since several
    j nodes usually qualify for the same shell.

    Args:
        records: list of dict from run_onion_protocol(), each with
            'layer_distance', 'probe_distance', 'r_eff'.
        reference_node: optional node index, used only for the title.
        title: optional custom title.
        figsize: figure size tuple.
        log_y: plot R_eff on a log scale (resistance in percolating networks
            typically spans orders of magnitude across layers).

    Returns:
        Matplotlib figure object, or None if there are no finite R_eff
        records to plot. Each layer-mean marker is annotated with "n=" the
        count of node pairs (probes) behind that point, and the title
        states the total probed / excluded (disconnected, R_eff=inf) counts,
        so the sample size behind the curve is visible on the plot itself.
    """
    n_total = len(records)
    finite = [r for r in records if np.isfinite(r['r_eff'])]
    n_excluded = n_total - len(finite)
    if not finite:
        print("[WARNING] No finite R_eff records to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    probe_d = [r['probe_distance'] for r in finite]
    r_eff = [r['r_eff'] for r in finite]
    ax.scatter(probe_d, r_eff, c='steelblue', s=30, alpha=0.6, zorder=3,
              label='individual probes')

    layers = sorted(set(r['layer_distance'] for r in finite))
    layer_values = [[r['r_eff'] for r in finite if r['layer_distance'] == d] for d in layers]
    layer_means = [np.mean(v) for v in layer_values]
    layer_stds = [np.std(v) for v in layer_values]
    layer_counts = [len(v) for v in layer_values]
    ax.errorbar(layers, layer_means, yerr=layer_stds, color='darkred', marker='o',
               markersize=6, linewidth=1.5, capsize=4, zorder=4,
               label='layer mean ± std')
    for d, mean_val, n in zip(layers, layer_means, layer_counts):
        ax.annotate(f'n={n}', (d, mean_val), textcoords="offset points", xytext=(0, 9),
                   ha='center', fontsize=7.5, color='darkred', zorder=5)

    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel('Distance from reference node')
    ax.set_ylabel('R_eff (Ω)')
    ref_str = f' from node {reference_node}' if reference_node is not None else ''
    count_str = f'{len(finite)} node pairs probed'
    if n_excluded:
        count_str += f', {n_excluded} excluded (disconnected)'
    ax.set_title(title or f'Effective resistance vs. distance{ref_str}\n'
                          f'(onion-layer protocol, {count_str})',
                fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()

    return fig
