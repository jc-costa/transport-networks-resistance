"""TOC graphical abstract for IEEE Trans. Nanotechnology (author information: at most 3.25 x 1.75 in,
TIFF 300 dpi in color, sans-serif 8 pt and no smaller than 6 pt, no specific results, mostly free of text).
It shows the question of the paper and no data or numbers: all pairs of interior points of a stick film
(the upper bound) against the few readings that a probe protocol can take (one tip with the source electrode,
two tips, a four-point probe), to estimate the junction resistance.
Writes figures/toc_graphic.pdf, .png and .tif.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "text.color": "#0b0b0b"})
BLUE, ORANGE, GREEN, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#0b0b0b"
FIGDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

rng = np.random.default_rng(7)
sticks = []
while len(sticks) < 24:
    c = rng.uniform(0.05, 0.95, 2); th = rng.uniform(0, np.pi)
    d = 0.2 * np.array([np.cos(th), np.sin(th)])
    sticks.append((np.clip(c - d, 0, 1), np.clip(c + d, 0, 1)))


def cross(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
    u = ((q1[0] - p1[0]) * d1[1] - (q1[1] - p1[1]) * d1[0]) / den
    return p1 + t * d1 if 0 <= t <= 1 and 0 <= u <= 1 else None


nodes = np.array([x for i in range(len(sticks)) for j in range(i + 1, len(sticks))
                  if (x := cross(*sticks[i], *sticks[j])) is not None])


def film(ax):
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 0.1, 1, color="#d8d7d2", lw=0))
    ax.add_patch(Rectangle((0.9, 0), 0.1, 1, color="#d8d7d2", lw=0))
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=0.5, ec=GRAY))
    for p, q in sticks:
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#b9b8b3", lw=0.8, solid_capstyle="round", zorder=1)


def on_wire(target):
    best, bd = None, 9.0
    for p, q in sticks:
        v = q - p; t = np.clip(np.dot(target - p, v) / np.dot(v, v), 0, 1); pt = p + t * v
        if np.hypot(*(pt - target)) < bd:
            best, bd = pt, np.hypot(*(pt - target))
    return best


fig = plt.figure(figsize=(3.25, 1.7))
axl = fig.add_axes([0.01, 0.17, 0.455, 0.64]); film(axl)
axr = fig.add_axes([0.535, 0.17, 0.455, 0.64]); film(axr)

# left: all pairs of interior points, drawn as a web of thin lines between the crossings
inner = nodes[(nodes[:, 0] > 0.12) & (nodes[:, 0] < 0.88)]
# farthest-point sampling, so that the nodes spread over the film
chosen = [int(np.argmin(inner[:, 0] + inner[:, 1]))]
while len(chosen) < min(22, len(inner)):
    d = np.min(np.linalg.norm(inner[:, None, :] - inner[chosen][None, :, :], axis=2), axis=1)
    chosen.append(int(np.argmax(d)))
sel = inner[chosen]
for i in range(len(sel)):
    for j in range(i + 1, len(sel)):
        axl.plot([sel[i, 0], sel[j, 0]], [sel[i, 1], sel[j, 1]], color=BLUE, lw=0.35, alpha=0.5, zorder=2)
axl.scatter(sel[:, 0], sel[:, 1], s=5, color=BLUE, zorder=3, lw=0)

# right: a few readings
one = on_wire(np.array([0.55, 0.74]))
two = [on_wire(np.array([0.34, 0.3])), on_wire(np.array([0.64, 0.4]))]
four = [on_wire(np.array([0.2 + 0.1 * k, 0.1 + 0.12 * k])) for k in range(4)]
axr.plot([0.1, one[0]], [one[1], one[1]], color=BLUE, lw=0.8, ls=(0, (2, 1.5)), zorder=4)
axr.scatter(*one, s=22, color=BLUE, zorder=6, lw=0, marker="v")
axr.scatter([p[0] for p in two], [p[1] for p in two], s=20, color=ORANGE, zorder=6, lw=0, marker="s")
axr.scatter([p[0] for p in four], [p[1] for p in four], s=18, color=GREEN, zorder=6, lw=0, marker="D")

fig.text(0.5, 0.945, "junction resistance?", ha="center", va="top", fontsize=8.5, fontweight="bold")
fig.text(0.237, 0.075, "all pairs", ha="center", va="center", fontsize=8)
fig.text(0.763, 0.075, "a few readings", ha="center", va="center", fontsize=8)
fig.text(0.5, 0.49, "vs", ha="center", va="center", fontsize=8, color=GRAY)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, f"toc_graphic.{ext}"), dpi=300)
Image.open(os.path.join(FIGDIR, "toc_graphic.png")).convert("RGB").save(os.path.join(FIGDIR, "toc_graphic.tif"), dpi=(300, 300), compression="tiff_lzw")
im = Image.open(os.path.join(FIGDIR, "toc_graphic.tif")); print("TOC graphic:", im.size, "px at 300 dpi =", round(im.size[0] / 300, 2), "x", round(im.size[1] / 300, 2), "in")
