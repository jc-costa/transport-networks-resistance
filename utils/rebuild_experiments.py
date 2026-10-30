#!/usr/bin/env python3
# ============================================================================
# REBUILD EXPERIMENTS
# Retroactively regenerates all missing plots and PDFs for existing
# exp_* folders using the data stored in experiment_summary.json.
#
# Run from the utils/ folder:
#   python3 rebuild_experiments.py
# ============================================================================

import os, sys, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless - no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -- make sure utils/ is importable ------------------------------------------
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.normpath(os.path.join(UTILS_DIR, "..", "experiments"))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from pdf_generator import generate_experiment_pdf, generate_general_pdf


# -- helpers ------------------------------------------------------------------

def _pts(lst):
    """Convert [[x,y], ...] -> list of (x, y) tuples."""
    return [tuple(p) for p in lst]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK]  {os.path.basename(path)}")


# -- plot generators ----------------------------------------------------------

def make_01_random_lines(summary, N):
    lines = summary["network_generation"]["lines"]
    fig = plt.figure(figsize=(8, 6))
    for L in lines:
        x = [L["start_point"][0], L["end_point"][0]]
        y = [L["start_point"][1], L["end_point"][1]]
        plt.plot(x, y, linestyle="-", color="steelblue")
    plt.title(f"N = {N} - Random Lines", fontsize=22)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xticks([]); plt.yticks([])
    plt.grid(True, linestyle="dotted", linewidth=0.5, color="gray")
    return fig


def make_02_intersections(summary, N):
    lines  = summary["network_generation"]["lines"]
    with open(os.path.join(_exp_dir, "all_intersections.json")) as f:
        pts = _pts(json.load(f))
    with open(os.path.join(_exp_dir, "segment_edge_list_original.json")) as f:
        seg_orig = json.load(f)

    # Which stick indices have at least one intersection
    intersected_sticks = set(e[0] for e in seg_orig)

    fig = plt.figure(figsize=(8, 6))
    # Draw all sticks in steelblue first
    for L in lines:
        x = [L["start_point"][0], L["end_point"][0]]
        y = [L["start_point"][1], L["end_point"][1]]
        plt.plot(x, y, linestyle="-", color="steelblue", zorder=0)
    # Overlay intersected sticks in red (matching original notebook)
    for L in lines:
        if L["line_index"] in intersected_sticks:
            x = [L["start_point"][0], L["end_point"][0]]
            y = [L["start_point"][1], L["end_point"][1]]
            plt.plot(x, y, linestyle="-", color="red", zorder=1)
    # Intersection points in black
    if pts:
        xs, ys = zip(*pts)
        plt.scatter(xs, ys, c="black", marker="o", s=6, zorder=2)
    plt.title(f"N = {N} - Intersections", fontsize=22)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xticks([]); plt.yticks([])
    return fig


def make_03_percolating_paths(summary, N):
    with open(os.path.join(_exp_dir, "all_intersections.json")) as f:
        pts = _pts(json.load(f))
    perc_paths = summary["percolation_analysis"]["percolating_paths"]

    fig = plt.figure(figsize=(8, 6))
    if not pts or not perc_paths:
        plt.title(f"N = {N} - No percolating paths", fontsize=16)
        return fig

    xs_all = [p[0] for p in pts]
    ys_all = [p[1] for p in pts]
    colors = ["green", "red", "blue", "yellow", "orange", "black"]
    for ci, path in enumerate(perc_paths):
        col = colors[ci % len(colors)]
        for ni in path["nodes"]:
            plt.scatter(xs_all[ni], ys_all[ni], c=col, marker="*", s=6, zorder=2)
        for e in path["edges"]:
            i, j = e[0], e[1]
            plt.plot([xs_all[i], xs_all[j]], [ys_all[i], ys_all[j]],
                     linestyle="-", color=col, zorder=0)
    plt.title(f"N = {N} - Percolating Paths (Before Pruning)", fontsize=16)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xticks([]); plt.yticks([])
    return fig


def make_04_before_after_pruning(summary, N):
    with open(os.path.join(_exp_dir, "all_intersections.json")) as f:
        pts = _pts(json.load(f))
    with open(os.path.join(_exp_dir, "segment_edge_list_original.json")) as f:
        seg_orig = json.load(f)         # [[seg, [n1, n2]], ...]
    with open(os.path.join(_exp_dir, "segment_edge_list_pruned.json")) as f:
        seg_prun = json.load(f)

    def to_edge_set(seg_list):
        return set((min(e[1][0], e[1][1]), max(e[1][0], e[1][1])) for e in seg_list)

    edges_before = to_edge_set(seg_orig)
    edges_after  = to_edge_set(seg_prun)
    removed_edges = edges_before - edges_after

    nodes_before = {n for e in seg_orig for n in e[1]}
    nodes_after  = {n for e in seg_prun for n in e[1]}
    removed_nodes = nodes_before - nodes_after

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, seg_list, title_sfx, show_removed in [
        (axes[0], seg_orig, "Before Pruning", True),
        (axes[1], seg_prun, "After Pruning",  False),
    ]:
        for e in seg_list:
            n1, n2 = e[1][0], e[1][1]
            x1, y1 = pts[n1]; x2, y2 = pts[n2]
            key = (min(n1, n2), max(n1, n2))
            if show_removed and key in removed_edges:
                ax.plot([x1, x2], [y1, y2], color="red", lw=2, ls="--", alpha=0.7, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], color="blue", lw=3, alpha=0.9, zorder=1)

        node_set = nodes_before if show_removed else nodes_after
        for node in node_set:
            x, y = pts[node]
            if x < 0.1:
                c, mk, sz = "green",  "s", 150
            elif x > 0.9:
                c, mk, sz = "red",    "^", 150
            elif show_removed and node in removed_nodes:
                c, mk, sz = "orange", "x", 100
            else:
                c, mk, sz = "blue",   "o",  80
            ax.scatter(x, y, c=c, s=sz, marker=mk, zorder=3,
                       edgecolors="black", linewidths=0.5)
            ax.annotate(f"{node}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, fontweight="bold")

        n_e = len(edges_before) if show_removed else len(edges_after)
        n_n = len(nodes_before) if show_removed else len(nodes_after)
        ax.set_title(f"{title_sfx}\n{n_e} edges, {n_n} nodes", fontsize=12, fontweight="bold")
        ax.axvline(x=0.1, color="green", ls=":", alpha=0.5, lw=2)
        ax.axvline(x=0.9, color="red",   ls=":", alpha=0.5, lw=2)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def make_05_before_after_merge(summary, N):
    """Reconstruct the before/after merge plot from experiment_summary.json."""
    merge  = summary.get("merge_operation", {})
    merged_pts_raw    = merge.get("merged_intersections", [])
    merged_graph_raw  = merge.get("merged_graph", [])
    merged_src        = merge.get("merged_source_node")
    merged_drn        = merge.get("merged_drain_node")
    nodes_to_src      = merge.get("nodes_merged_to_source", [])
    nodes_to_drn      = merge.get("nodes_merged_to_drain",  [])

    elec = summary.get("electrical_solution", {})
    voltages_raw = elec.get("voltages", [])
    currents_raw = elec.get("currents", [])

    imp_calc = summary.get("impedance_calculation", {})
    R_val = imp_calc.get("calculated_impedance_ohms", None)
    k_val = imp_calc.get("k_used_ohms", "?")

    # -- pruned network (before merge) --------------------------------
    with open(os.path.join(_exp_dir, "all_intersections.json")) as f:
        all_pts = _pts(json.load(f))
    with open(os.path.join(_exp_dir, "segment_edge_list_pruned.json")) as f:
        seg_prun = json.load(f)

    nodes_after  = {n for e in seg_prun for n in e[1]}
    edges_pruned = [(e[1][0], e[1][1]) for e in seg_prun]

    # Validate node indices before attempting to draw
    merged_pts_tmp = [tuple(p) for p in merged_pts_raw] if merged_pts_raw else []
    max_node_ref = max(
        (max(eg["node1"], eg["node2"]) for eg in merged_graph_raw),
        default=-1
    ) if merged_graph_raw else -1
    data_ok = (bool(merged_pts_raw) and merged_src is not None
               and max_node_ref < len(merged_pts_tmp))
    has_merge = data_ok

    # -- figure layout ------------------------------------------------
    if has_merge:
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 8))
        ax2 = None; cax = None

    # -- LEFT: before merge (pruned network) --------------------------
    for n1, n2 in edges_pruned:
        x1, y1 = all_pts[n1]; x2, y2 = all_pts[n2]
        ax1.plot([x1, x2], [y1, y2], color="steelblue", lw=2.5, alpha=0.8, zorder=1)

    src_nodes_before = [n for n in nodes_after if all_pts[n][0] < 0.1]
    drn_nodes_before = [n for n in nodes_after if all_pts[n][0] > 0.9]
    int_nodes_before = [n for n in nodes_after
                        if all_pts[n][0] >= 0.1 and all_pts[n][0] <= 0.9]

    for ni in src_nodes_before:
        x, y = all_pts[ni]
        ax1.scatter(x, y, c="green", s=150, marker="s", zorder=3,
                    edgecolors="darkgreen", linewidths=2)
        ax1.annotate(f"S{ni}", (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, fontweight="bold", color="darkgreen")
    for ni in drn_nodes_before:
        x, y = all_pts[ni]
        ax1.scatter(x, y, c="red", s=150, marker="^", zorder=3,
                    edgecolors="darkred", linewidths=2)
        ax1.annotate(f"D{ni}", (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, fontweight="bold", color="darkred")
    for ni in int_nodes_before:
        x, y = all_pts[ni]
        ax1.scatter(x, y, c="blue", s=80, marker="o", zorder=3,
                    edgecolors="black", linewidths=0.5)
        ax1.annotate(f"{ni}", (x, y), textcoords="offset points",
                     xytext=(4, 4), fontsize=7)

    legend1 = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="green",
               markersize=10, markeredgecolor="darkgreen", markeredgewidth=2,
               label=f"SOURCE nodes ({len(src_nodes_before)})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markersize=10, markeredgecolor="darkred", markeredgewidth=2,
               label=f"DRAIN nodes ({len(drn_nodes_before)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
               markersize=8, markeredgecolor="black",
               label=f"INTERNAL nodes ({len(int_nodes_before)})"),
        Line2D([0], [0], color="steelblue", lw=2.5,
               label=f"Edges ({len(edges_pruned)})"),
    ]
    ax1.legend(handles=legend1, loc="upper left", fontsize=8, title="BEFORE MERGE")
    ax1.axvline(x=0.1, color="green", ls="--", alpha=0.5, lw=2)
    ax1.axvline(x=0.9, color="red",   ls="--", alpha=0.5, lw=2)
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
    ax1.set_title(f"BEFORE Merge\n{len(nodes_after)} nodes, {len(edges_pruned)} edges",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    # -- RIGHT: after merge --------------------------------------------
    if has_merge:
        merged_pts = _pts(merged_pts_raw)
        voltages = np.array(voltages_raw) if voltages_raw else None
        currents = np.array(currents_raw) if currents_raw else None

        max_curr = float(np.max(np.abs(currents))) if currents is not None and len(currents) else 1.0

        # Build a per-edge-pair -> current mapping using merged_segment_to_nodes
        # (currents are indexed by segment, not by merged_graph order)
        cond_info = summary.get("conductance_info", {})
        seg_to_nodes = cond_info.get("merged_segment_to_nodes", {})

        # Map (n1,n2) key -> sum of |current| for all segments on that pair
        edge_current_map: dict = {}
        for seg_str, nodes_pair in seg_to_nodes.items():
            seg_idx = int(seg_str)
            n1_, n2_ = nodes_pair[0], nodes_pair[1]
            key = (min(n1_, n2_), max(n1_, n2_))
            c_val = (abs(float(currents[seg_idx]))
                     if currents is not None and seg_idx < len(currents) else 0.0)
            edge_current_map[key] = edge_current_map.get(key, 0.0) + c_val

        # edges coloured by current magnitude
        heat = plt.cm.YlOrRd
        for eg in merged_graph_raw:
            n1, n2 = eg["node1"], eg["node2"]
            x1, y1 = merged_pts[n1]; x2, y2 = merged_pts[n2]
            key = (min(n1, n2), max(n1, n2))
            c_val = edge_current_map.get(key, 0.0)
            norm_c = c_val / max_curr if max_curr > 0 else 0.5
            lw = 1.5 + 6 * norm_c
            ax2.plot([x1, x2], [y1, y2], color=heat(norm_c), lw=lw, zorder=1)

        # nodes coloured by voltage
        for i, (x, y) in enumerate(merged_pts):
            v = float(voltages[i]) if voltages is not None and i < len(voltages) else 0.5
            vcol = plt.cm.coolwarm(v)
            if i == merged_src:
                mk, sz, ec, lbl = "s", 500, "darkgreen", f"SOURCE\nV={v:.2f}"
            elif i == merged_drn:
                mk, sz, ec, lbl = "^", 500, "darkred",   f"DRAIN\nV={v:.2f}"
            else:
                mk, sz, ec, lbl = "o", 200, "black",     f"I{i}\nV={v:.2f}"
            ax2.scatter(x, y, c=[vcol], s=sz, marker=mk, zorder=3,
                        edgecolors=ec, linewidths=3)
            ax2.annotate(lbl, (x, y), textcoords="offset points",
                         xytext=(8, 8), fontsize=8, fontweight="bold")

        legend2 = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
                   markersize=12, markeredgecolor="darkgreen", markeredgewidth=3,
                   label="SOURCE merged (V=1)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="blue",
                   markersize=12, markeredgecolor="darkred", markeredgewidth=3,
                   label="DRAIN merged (V=0)"),
            Line2D([0], [0], color=heat(0.1), lw=3, label="Edge: low current"),
            Line2D([0], [0], color=heat(0.5), lw=5, label="Edge: medium current"),
            Line2D([0], [0], color=heat(0.9), lw=7, label="Edge: high current"),
        ]
        ax2.legend(handles=legend2, loc="lower right", fontsize=8, title="AFTER MERGE")
        ax2.axvline(x=0.1, color="green", ls="--", alpha=0.5, lw=2)
        ax2.axvline(x=0.9, color="red",   ls="--", alpha=0.5, lw=2)
        ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.05)
        ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
        r_str = f"{R_val:.4f} Ohm" if isinstance(R_val, (int, float)) else "--"
        ax2.set_title(
            f"AFTER Merge\n{len(merged_pts)} nodes, {len(merged_graph_raw)} edges"
            f"  |  R_eq = {r_str}  (k={k_val} Ohm)",
            fontsize=12, fontweight="bold")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, cax=cax).set_label("Voltage (V)", fontsize=10)
    else:
        # -- Fallback: reconstruct after-merge from nodes_detailed + edges_detailed --
        # Used when merged_intersections has index inconsistency (old export format)
        with open(os.path.join(_exp_dir, "nodes_detailed.json")) as f:
            nodes_det = json.load(f)   # [{local_index, x, y, type, voltage_V}, ...]
        with open(os.path.join(_exp_dir, "edges_detailed.json")) as f:
            edges_det = json.load(f)   # [{edge_index, node1, node2, current_A, ...}]

        # Build position and voltage maps keyed by global node index
        pos_map  = {nd["global_index"]: (nd["x"], nd["y"]) for nd in nodes_det}
        volt_map = {nd["global_index"]: nd.get("voltage_V") for nd in nodes_det}
        type_map = {nd["global_index"]: nd["type"] for nd in nodes_det}

        # Identify source/drain global indices from merge info
        src_globals = {d["global"] for d in merge.get("nodes_merged_to_source", [])}
        drn_globals = {d["global"] for d in merge.get("nodes_merged_to_drain",  [])}

        # Compute virtual merged positions: centroid of all source/drain nodes
        def centroid(g_set):
            xs = [pos_map[g][0] for g in g_set if g in pos_map]
            ys = [pos_map[g][1] for g in g_set if g in pos_map]
            return (sum(xs)/len(xs), sum(ys)/len(ys)) if xs else (0.05, 0.5)

        src_pos = centroid(src_globals)
        drn_pos = centroid(drn_globals)

        # Voltages: use max voltage for source group, 0 for drain group
        src_volts = [volt_map[g] for g in src_globals if volt_map.get(g) is not None]
        drn_volts = [volt_map[g] for g in drn_globals if volt_map.get(g) is not None]
        v_src = max(src_volts) if src_volts else 1.0
        v_drn = min(drn_volts) if drn_volts else 0.0

        # Internal nodes: not source or drain
        int_nodes = [nd for nd in nodes_det
                     if nd["global_index"] not in src_globals
                     and nd["global_index"] not in drn_globals]

        # Currents for heat map
        all_currents = [abs(e.get("current_A") or 0) for e in edges_det]
        max_curr_fb = max(all_currents) if all_currents else 1.0
        heat = plt.cm.YlOrRd

        # Draw merged figure on the right axes of the existing figure
        # (we need to create ax2 here since has_merge was False)
        plt.close(fig)  # discard single-axis figure
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
        ax1_new = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])

        # Re-draw left panel on ax1_new
        for n1, n2 in edges_pruned:
            x1, y1 = all_pts[n1]; x2, y2 = all_pts[n2]
            ax1_new.plot([x1, x2], [y1, y2], color="steelblue", lw=2.5, alpha=0.8, zorder=1)
        for ni in src_nodes_before:
            x, y = all_pts[ni]
            ax1_new.scatter(x, y, c="green", s=150, marker="s", zorder=3,
                            edgecolors="darkgreen", linewidths=2)
            ax1_new.annotate(f"S{ni}", (x, y), textcoords="offset points",
                             xytext=(5, 5), fontsize=8, fontweight="bold", color="darkgreen")
        for ni in drn_nodes_before:
            x, y = all_pts[ni]
            ax1_new.scatter(x, y, c="red", s=150, marker="^", zorder=3,
                            edgecolors="darkred", linewidths=2)
            ax1_new.annotate(f"D{ni}", (x, y), textcoords="offset points",
                             xytext=(5, 5), fontsize=8, fontweight="bold", color="darkred")
        for ni in int_nodes_before:
            x, y = all_pts[ni]
            ax1_new.scatter(x, y, c="blue", s=80, marker="o", zorder=3,
                            edgecolors="black", linewidths=0.5)
            ax1_new.annotate(f"{ni}", (x, y), textcoords="offset points",
                             xytext=(4, 4), fontsize=7)
        ax1_new.legend(handles=legend1, loc="upper left", fontsize=8, title="BEFORE MERGE")
        ax1_new.axvline(x=0.1, color="green", ls="--", alpha=0.5, lw=2)
        ax1_new.axvline(x=0.9, color="red",   ls="--", alpha=0.5, lw=2)
        ax1_new.set_xlim(-0.05, 1.05); ax1_new.set_ylim(-0.05, 1.05)
        ax1_new.set_aspect("equal"); ax1_new.grid(True, alpha=0.3)
        ax1_new.set_title(f"BEFORE Merge\n{len(nodes_after)} nodes, {len(edges_pruned)} edges",
                          fontsize=12, fontweight="bold")
        ax1_new.set_xlabel("x"); ax1_new.set_ylabel("y")

        # Draw right panel (after merge) using nodes_detailed / edges_detailed
        for e in edges_det:
            g1, g2 = e["node1"], e["node2"]
            # Use virtual merged position for source/drain groups
            p1 = src_pos if g1 in src_globals else (drn_pos if g1 in drn_globals else pos_map.get(g1))
            p2 = src_pos if g2 in src_globals else (drn_pos if g2 in drn_globals else pos_map.get(g2))
            if p1 is None or p2 is None:
                continue
            curr = abs(e.get("current_A") or 0)
            norm_c = curr / max_curr_fb if max_curr_fb > 0 else 0.5
            lw = 1.5 + 6 * norm_c
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color=heat(norm_c), lw=lw, zorder=1)

        # Source merged node
        ax2.scatter(*src_pos, c=[plt.cm.coolwarm(v_src)], s=500, marker="s", zorder=3,
                    edgecolors="darkgreen", linewidths=3)
        ax2.annotate(f"SOURCE\nV={v_src:.2f}", src_pos, textcoords="offset points",
                     xytext=(8, 8), fontsize=8, fontweight="bold")
        # Drain merged node
        ax2.scatter(*drn_pos, c=[plt.cm.coolwarm(v_drn)], s=500, marker="^", zorder=3,
                    edgecolors="darkred", linewidths=3)
        ax2.annotate(f"DRAIN\nV={v_drn:.2f}", drn_pos, textcoords="offset points",
                     xytext=(8, 8), fontsize=8, fontweight="bold")
        # Internal nodes
        for nd in int_nodes:
            x, y = nd["x"], nd["y"]
            v = nd.get("voltage_V") or 0.5
            ax2.scatter(x, y, c=[plt.cm.coolwarm(v)], s=200, marker="o", zorder=3,
                        edgecolors="black", linewidths=1)
            ax2.annotate(f"V={v:.2f}", (x, y), textcoords="offset points",
                         xytext=(5, 5), fontsize=7)

        legend2 = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
                   markersize=12, markeredgecolor="darkgreen", markeredgewidth=3,
                   label=f"SOURCE merged (V={v_src:.2f})"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="blue",
                   markersize=12, markeredgecolor="darkred", markeredgewidth=3,
                   label=f"DRAIN merged (V={v_drn:.2f})"),
            Line2D([0], [0], color=heat(0.1), lw=3, label="Edge: low current"),
            Line2D([0], [0], color=heat(0.9), lw=7, label="Edge: high current"),
        ]
        ax2.legend(handles=legend2, loc="lower right", fontsize=8, title="AFTER MERGE")
        ax2.axvline(x=0.1, color="green", ls="--", alpha=0.5, lw=2)
        ax2.axvline(x=0.9, color="red",   ls="--", alpha=0.5, lw=2)
        ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.05)
        ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
        r_str = f"{R_val:.4f} Ohm" if isinstance(R_val, (int, float)) else "--"
        ax2.set_title(
            f"AFTER Merge (reconstructed)\nR_eq = {r_str}  (k={k_val} Ohm)",
            fontsize=12, fontweight="bold")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, cax=cax).set_label("Voltage (V)", fontsize=10)

    plt.tight_layout()
    return fig


# -- main rebuild loop ---------------------------------------------------------

def rebuild_experiment(exp_dir):
    global _exp_dir
    _exp_dir = exp_dir           # module-level so helper fns can read JSONs

    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [WARNING]  No experiment_summary.json in {exp_dir} -- skipping.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    N = summary.get("parameters", {}).get("N", "?")
    print(f"  -> Rebuilding plots for N={N} ...")

    # Map of filename -> generator function
    plots = {
        f"01_random_lines_N{N}.png":                  make_01_random_lines,
        f"02_intersections_N{N}.png":                 make_02_intersections,
        f"03_percolating_paths_before_pruning_N{N}.png": make_03_percolating_paths,
        f"04_before_after_pruning_N{N}.png":          make_04_before_after_pruning,
        f"05_before_after_merge_N{N}.png":            make_05_before_after_merge,
    }

    for fname, gen_fn in plots.items():
        fpath = os.path.join(exp_dir, fname)
        try:
            fig = gen_fn(summary, N)
            _save(fig, fpath)
        except Exception as e:
            print(f"    [FAIL]  {fname}: {e}")

    # Regenerate individual PDF
    try:
        generate_experiment_pdf(exp_dir, summary=summary)
    except Exception as e:
        print(f"    [FAIL]  PDF individual: {e}")


def main():
    pattern = re.compile(r"^exp_(\d+)$")
    entries = []
    for name in os.listdir(EXPERIMENTS_DIR):
        m = pattern.match(name)
        if m:
            entries.append((int(m.group(1)), name))
    entries.sort()

    if not entries:
        print("No experiment folders found.")
        return

    print(f"Found {len(entries)} experiment folder(s): {[e[1] for e in entries]}\n")

    for num, folder in entries:
        exp_dir = os.path.join(EXPERIMENTS_DIR, folder)
        print(f"[exp_{num}]")
        rebuild_experiment(exp_dir)
        print()

    # Regenerate general PDF
    print("[General PDF]")
    generate_general_pdf(EXPERIMENTS_DIR)
    print("\nDone [OK]")


if __name__ == "__main__":
    main()
