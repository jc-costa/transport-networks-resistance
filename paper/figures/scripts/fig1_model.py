"""Fig. 1 (single column): (a) schematic of a stick film between two strip
electrodes and the three measurement protocols compared in the paper: one
movable tip with the fixed source electrode, two movable tips, and a four-point
probe; no data.
(b) the near-threshold example network (N = 30, a = 0.4), rebuilt from its
stored sticks by regen_example_network.py, with its red bonds marked.

Input: data/example_network.json (written by regen_example_network.py).
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from style import ONE_COL, CAT, INK, INK2, RED, GRAY, DATADIR, panel, save


def seg_intersect(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
    u = ((q1[0] - p1[0]) * d1[1] - (q1[1] - p1[1]) * d1[0]) / den
    return p1 + t * d1 if 0 <= t <= 1 and 0 <= u <= 1 else None


fig, (axa, axb) = plt.subplots(1, 2, figsize=(ONE_COL, 1.85))
for ax in (axa, axb):
    ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=0.6, ec=INK2))
    ax.add_patch(Rectangle((0, 0), 0.1, 1, color="#d8d7d2", lw=0, zorder=0))
    ax.add_patch(Rectangle((0.9, 0), 0.1, 1, color="#d8d7d2", lw=0, zorder=0))

# (a) a hand-placed film: sticks, and the three protocols placed on wires
rng = np.random.default_rng(7)
sticks = []
while len(sticks) < 22:
    c = rng.uniform(0.05, 0.95, 2); th = rng.uniform(0, np.pi)
    d = 0.21 * np.array([np.cos(th), np.sin(th)])
    sticks.append((np.clip(c - d, 0, 1), np.clip(c + d, 0, 1)))
for p, q in sticks:
    axa.plot([p[0], q[0]], [p[1], q[1]], color="#b9b8b3", lw=0.8, solid_capstyle="round")


def on_wire(target):
    """The nearest point of any stick to a target position."""
    best, bd = None, 9.0
    for p, q in sticks:
        v = q - p; t = np.clip(np.dot(target - p, v) / np.dot(v, v), 0, 1); pt = p + t * v
        d = np.hypot(*(pt - target))
        if d < bd:
            best, bd = pt, d
    return best


one = on_wire(np.array([0.50, 0.57]))
two = [on_wire(np.array([0.50, 0.90])), on_wire(np.array([0.78, 0.62]))]
four = [on_wire(np.array([0.20 + 0.10 * k, 0.10 + 0.12 * k])) for k in range(4)]
# Filled marker: the contact carries the current (and, in a two-point reading, the voltage is read across the same
# two contacts). Open marker: the contact only reads the voltage (the inner tips of the four-point probe).
axa.plot([0.1, 0.1], [0, 1], color=CAT[0], lw=1.1, zorder=3)                                   # the fixed electrode: the source strip
axa.plot([0.1, one[0]], [one[1], one[1]], color=CAT[0], lw=0.8, ls=(0, (2, 1.5)), zorder=4)   # reading between electrode and tip
axa.scatter(*one, s=18, color=CAT[0], zorder=6, lw=0, marker="v")
axa.plot([p[0] for p in two], [p[1] for p in two], color=CAT[1], lw=0.8, ls=(0, (2, 1.5)), zorder=4)
axa.scatter([p[0] for p in two], [p[1] for p in two], s=18, color=CAT[1], zorder=6, lw=0, marker="s")
axa.plot([four[0][0], four[3][0]], [four[0][1], four[3][1]], color=CAT[2], lw=0.6, ls=(0, (2, 1.5)), zorder=4)
axa.scatter([four[k][0] for k in (0, 3)], [four[k][1] for k in (0, 3)], s=16, color=CAT[2], zorder=6, lw=0, marker="D")
axa.scatter([four[k][0] for k in (1, 2)], [four[k][1] for k in (1, 2)], s=14, facecolor="white", edgecolor=CAT[2], zorder=6, lw=0.8, marker="D")
axa.text(0.05, -0.04, "source", ha="center", va="top", fontsize=6.5, color=INK2)
axa.text(0.95, -0.04, "drain", ha="center", va="top", fontsize=6.5, color=INK2)
from matplotlib.lines import Line2D  # noqa: E402
axa.legend(handles=[Line2D([], [], color=CAT[0], marker="v", ls="", ms=3.5, label="one tip"),
                    Line2D([], [], color=CAT[1], marker="s", ls="", ms=3.5, label="two tips"),
                    Line2D([], [], color=CAT[2], marker="D", ls="", ms=3.2, label="four points"),
                    Line2D([], [], color=INK2, marker="o", ls="", ms=3.2, label="current"),
                    Line2D([], [], color=INK2, marker="o", ls="", ms=3.2, mfc="white", mew=0.8, label="voltage only")],
           loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=6.3, handletextpad=0.1, labelspacing=0.15,
           columnspacing=0.5, borderaxespad=0.0)
panel(axa, "a", x=-0.02, y=1.0)

# (b) the example network with its red bonds
net = json.load(open(os.path.join(DATADIR, "example_network.json")))
for p, q in net["sticks"]:
    axb.plot([p[0], q[0]], [p[1], q[1]], color="#dcdbd6", lw=0.6, zorder=1)
X = np.array(net["pre_merge_nodes"])
used = set()
for p_, q_, red, alive in net["pre_merge_edges_red_alive"]:
    col, lw, z = (RED, 1.4, 4) if red else ((INK2, 0.8, 3) if alive else (GRAY, 0.5, 2))
    axb.plot([X[p_, 0], X[q_, 0]], [X[p_, 1], X[q_, 1]], color=col, lw=lw, zorder=z, solid_capstyle="round")
    used.update((p_, q_))
used = sorted(used)
axb.scatter(X[used, 0], X[used, 1], s=3, color=INK, zorder=5, lw=0)
axb.plot([], [], color=RED, lw=1.4, label="red bond")
axb.plot([], [], color=INK2, lw=0.8, label="current")
axb.plot([], [], color=GRAY, lw=0.5, label="no current")
axb.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.5, handlelength=1.0,
           columnspacing=0.6, handletextpad=0.4)
panel(axb, "b", x=-0.02, y=1.0)
fig.subplots_adjust(wspace=0.08)
save(fig, "fig1_model.pdf")
print(f"example: R_eff = {net['R_eff_ohm']:.1f} Ohm = {net['R_eff_ohm'] / 1000:.3f} R0 L, "
      f"{net['n_red']} red bonds, {100 * net['red_R_fraction']:.1f}% of R_eff")
