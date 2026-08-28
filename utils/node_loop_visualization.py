"""
Node-loop visualization utilities.

Drawing functions for the multi-node self-loop tests
(test_self_loops_multi_node.ipynb): chain/ladder panels with voltage
coloring plus node-loop highlighting (ring nodes vs. dead-end branch
nodes get distinct styling), and the 3-panel baseline/looped/cleaned
plot used to report each test.
"""
import os
import matplotlib.pyplot as plt


def _edge_curve_points(x1, y1, x2, y2, bend, n=30):
    'Return (xs, ys) for a quadratic-bezier arc between two points.'
    import numpy as np
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        t = np.linspace(0, 2 * np.pi, n)
        r = bend if bend != 0 else 0.03
        return x1 + r * np.cos(t), y1 + r * np.sin(t)
    px, py = -dy / length, dx / length
    cx, cy = mx + px * bend, my + py * bend
    t = np.linspace(0, 1, n)
    xs = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
    ys = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
    return xs, ys


def _draw_edges_with_parallels(ax, segment_to_nodes, nx, ny, hl_set, branch_set=None, base_lw=2.0):
    'Draw all edges, curving apart parallel/duplicate edges.'
    if branch_set is None:
        branch_set = set()
    from collections import defaultdict
    pair_seen = defaultdict(int)
    pair_total = defaultdict(int)
    for n1, n2 in segment_to_nodes.values():
        key = (min(n1, n2), max(n1, n2))
        pair_total[key] += 1

    for n1, n2 in segment_to_nodes.values():
        is_branch = n1 in branch_set or n2 in branch_set
        is_loop = (n1 in hl_set or n2 in hl_set) and not is_branch
        if is_branch:
            color = '#9B30FF'
            ls = '--'
        else:
            color = 'orange' if is_loop else 'k'
            ls = '-'
        lw = 1.5 if (is_loop or is_branch) else base_lw
        alpha = 0.85 if (is_loop or is_branch) else 1.0

        key = (min(n1, n2), max(n1, n2))
        total = pair_total[key]
        idx = pair_seen[key]
        pair_seen[key] += 1

        if total == 1:
            ax.plot([nx[n1], nx[n2]], [ny[n1], ny[n2]],
                    color=color, lw=lw, zorder=2, alpha=alpha, linestyle=ls)
        elif total == 2:
            dist = ((nx[n1] - nx[n2]) ** 2 + (ny[n1] - ny[n2]) ** 2) ** 0.5
            minor_bend = 0.05 * dist
            major_bend = 1.7 * dist
            bend = minor_bend if idx == 0 else major_bend
            xs, ys = _edge_curve_points(nx[n1], ny[n1], nx[n2], ny[n2], bend)
            ax.plot(xs, ys, color=color, lw=lw, zorder=2, alpha=alpha, linestyle=ls)
        else:
            dist = ((nx[n1] - nx[n2]) ** 2 + (ny[n1] - ny[n2]) ** 2) ** 0.5
            span = max(1.0 * dist, 0.08)
            bend = span * (idx - (total - 1) / 2.0) or span * 0.5
            xs, ys = _edge_curve_points(nx[n1], ny[n1], nx[n2], ny[n2], bend)
            ax.plot(xs, ys, color=color, lw=lw, zorder=2, alpha=alpha, linestyle=ls)


def _marker_label_offset(ax, marker_s=260, safety=2.0):
    'Return y-offset (data units) that clears a scatter marker.'
    import math
    fig = ax.figure
    dpi = fig.dpi
    marker_radius_pts = math.sqrt(marker_s / math.pi)
    marker_radius_px = marker_radius_pts * dpi / 72.0
    inv = ax.transData.inverted()
    p0 = inv.transform((0, 0))
    p1 = inv.transform((0, marker_radius_px))
    marker_radius_data_y = abs(p1[1] - p0[1])
    return marker_radius_data_y * safety


def draw_chain_panel_hl(ax, intersections, segment_to_nodes, voltages,
                         source_node, drain_node, title,
                         node_labels=None, y_mid=0.50,
                         highlight_nodes=None, branch_nodes=None):
    'Chain panel with voltage colours and orange/purple highlighting.'
    if highlight_nodes is None:
        highlight_nodes = []
    if branch_nodes is None:
        branch_nodes = []
    hl_set = set(highlight_nodes) - set(branch_nodes)
    branch_set = set(branch_nodes)

    nx = [p[0] for p in intersections]
    ny = [p[1] for p in intersections]

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.05, 0.95)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=7)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)
    ax.axvline(0.1, ls='--', lw=0.7, color='steelblue', alpha=0.4)
    ax.axvline(0.9, ls='--', lw=0.7, color='tomato', alpha=0.4)

    if hl_set:
        max_y = max(ny) if ny else 0.5
        min_y = min(ny) if ny else 0.5
        y_range = max_y - min_y if max_y != min_y else 0.3
        ax.set_ylim(min_y - 0.15, max_y + y_range * 1.5 + 0.1)

    _draw_edges_with_parallels(ax, segment_to_nodes, nx, ny, hl_set,
                                branch_set=branch_set, base_lw=2)

    cmap_v = plt.get_cmap("RdYlGn")
    marker_s = 260
    lbl_off = _marker_label_offset(ax, marker_s=marker_s)
    for i, (vx, vy, v) in enumerate(zip(nx, ny, voltages)):
        color = cmap_v(v)
        is_hl = i in hl_set
        is_branch = i in branch_set
        if is_branch:
            edge_color = '#9B30FF'
        elif is_hl:
            edge_color = '#FF6B00'
        else:
            edge_color = 'k'
        edge_width = 3.5 if (is_hl or is_branch) else 1.2
        ax.scatter(vx, vy, s=marker_s, c=[color], zorder=5,
                   edgecolors=edge_color, lw=edge_width)
        ax.text(vx, vy + lbl_off, "V={:.3f}".format(v),
                ha='center', va='bottom', fontsize=7.5,
                bbox=dict(fc='white', alpha=0.85, pad=1, edgecolor='none'))
        lbl = (node_labels or {}).get(i, "N{}".format(i))
        ax.text(vx, vy - lbl_off, lbl, ha='center', va='top', fontsize=8,
                color='navy', fontweight='bold')

    if highlight_nodes or branch_nodes:
        from matplotlib.lines import Line2D
        legend_elem = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markersize=10,
                   markeredgecolor='#FF6B00', markeredgewidth=3.5,
                   label='Ring node'),
            Line2D([0], [0], color='orange', lw=1.5,
                   label='Ring edges'),
        ]
        if branch_nodes:
            legend_elem += [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='gray', markersize=10,
                       markeredgecolor='#9B30FF', markeredgewidth=3.5,
                       label='Branch node'),
                Line2D([0], [0], color='#9B30FF', lw=1.5, linestyle='--',
                       label='Branch edge'),
            ]
        ax.legend(handles=legend_elem, loc='upper right', fontsize=6.5,
                 framealpha=0.85)


def draw_ladder_panel_hl(ax, intersections, segment_to_nodes, voltages,
                          source_node, drain_node, title,
                          node_labels=None,
                          highlight_nodes=None, branch_nodes=None):
    'Ladder panel with voltage colours and orange/purple highlighting.'
    if highlight_nodes is None:
        highlight_nodes = []
    if branch_nodes is None:
        branch_nodes = []
    hl_set = set(highlight_nodes) - set(branch_nodes)
    branch_set = set(branch_nodes)

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

    _draw_edges_with_parallels(ax, segment_to_nodes, nx, ny, hl_set,
                                branch_set=branch_set, base_lw=1.8)

    cmap_v = plt.get_cmap("RdYlGn")
    marker_s = 240
    lbl_off = _marker_label_offset(ax, marker_s=marker_s)
    for i, (vx, vy, v) in enumerate(zip(nx, ny, voltages)):
        color = cmap_v(v)
        is_hl = i in hl_set
        is_branch = i in branch_set
        if is_branch:
            edge_color = '#9B30FF'
        elif is_hl:
            edge_color = '#FF6B00'
        else:
            edge_color = 'k'
        edge_width = 3.5 if (is_hl or is_branch) else 1.2
        ax.scatter(vx, vy, s=marker_s, c=[color], zorder=5,
                   edgecolors=edge_color, lw=edge_width)
        ax.text(vx, vy + lbl_off, "V={:.3f}".format(v), ha='center', va='bottom',
                fontsize=7.5,
                bbox=dict(fc='white', alpha=0.85, pad=1, edgecolor='none'))
        lbl = (node_labels or {}).get(i, "N{}".format(i))
        ax.text(vx, vy - lbl_off, lbl, ha='center', va='top', fontsize=8,
                color='navy', fontweight='bold')

    if highlight_nodes or branch_nodes:
        from matplotlib.lines import Line2D
        legend_elem = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markersize=10,
                   markeredgecolor='#FF6B00', markeredgewidth=3.5,
                   label='Ring node'),
            Line2D([0], [0], color='orange', lw=1.5,
                   label='Ring edges'),
        ]
        if branch_nodes:
            legend_elem += [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='gray', markersize=10,
                       markeredgecolor='#9B30FF', markeredgewidth=3.5,
                       label='Branch node'),
                Line2D([0], [0], color='#9B30FF', lw=1.5, linestyle='--',
                       label='Branch edge'),
            ]
        ax.legend(handles=legend_elem, loc='upper right', fontsize=6.5,
                 framealpha=0.85)


def display_and_save_plot(exp_dir, intersections, seg_baseline, seg_looped, seg_cleaned,
                           res_baseline, res_looped, res_cleaned, src_node, drn_node,
                           topology='chain', title='', highlight_nodes=None, branch_nodes=None):
    '3-panel voltage profile plot with node-loop highlighting.'
    if highlight_nodes is None:
        highlight_nodes = []
    if branch_nodes is None:
        branch_nodes = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cases = [
        (seg_baseline, res_baseline['voltages'], 'Baseline',           [], []),
        (seg_looped,   res_looped['voltages'],   'With Loops',         highlight_nodes, branch_nodes),
        (seg_cleaned,  res_cleaned['voltages'],  'After Removal',      [], []),
    ]
    for ax, (seg, volts, label, hl_nodes, br_nodes) in zip(axes, cases):
        if topology == 'ladder':
            draw_ladder_panel_hl(ax, intersections, seg, volts,
                                 src_node, drn_node, label,
                                 highlight_nodes=hl_nodes, branch_nodes=br_nodes)
        else:
            draw_chain_panel_hl(ax, intersections, seg, volts,
                                src_node, drn_node, label,
                                highlight_nodes=hl_nodes, branch_nodes=br_nodes)
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()
    if exp_dir:
        path = os.path.join(exp_dir, 'voltage_profiles.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print('    plot -> {}'.format(path))
    plt.close(fig)
