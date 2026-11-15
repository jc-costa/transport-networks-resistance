"""Supplementary figure: score for log10 d against the relative noise added to
every shell value and terminal resistance, from data/noise_robustness.json
(written by noise_robustness.py). New analysis, not in the thesis."""
import os, json
import matplotlib.pyplot as plt
from style import ONE_COL, CAT, MARK, GRAY, INK2, DATADIR, grid, save

r = json.load(open(os.path.join(DATADIR, "noise_robustness.json")))
sig = r["sigmas"]
fig, ax = plt.subplots(figsize=(ONE_COL, 2.7))
for (key, lab), col, m in zip((("slab_3d", "slab, 3D shells"), ("flat_inplane", "flat sticks, in-plane")), CAT, MARK):
    for k, ls in (("1", "-"), ("24", "--")):
        ax.plot([100 * s for s in sig], [r[key][str(s)]["curve+R"][k] for s in sig], color=col, ls=ls,
                marker=m, ms=3, lw=1.0, mec="white", mew=0.4, label=f"{lab}, $k={k}$")
ax.plot([100 * s for s in sig], [r["slab_3d"][str(s)]["R"]["1"] for s in sig], color=GRAY, ls=":", lw=1.0,
        label=r"$R$ alone (slab)")
ax.axhline(0, color=INK2, lw=0.5)
ax.set_xlabel("relative noise on each measured value (%)")
ax.set_ylabel(r"out-of-sample $R^2$ for $\log_{10} d$")
ax.set_ylim(-0.1, 1.0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6, handlelength=1.8, labelspacing=0.2, columnspacing=1.0)
grid(ax)
os.makedirs(os.path.join(os.path.dirname(DATADIR), "supp"), exist_ok=True)
save(fig, os.path.join("supp", "noise_robustness.pdf"))
