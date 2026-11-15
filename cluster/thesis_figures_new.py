"""
Figures for the dissertation from the simulations made after it was first
written (large-film sweep, Apuana job 17598) and the noise analysis of the
paper. Same style as cluster/critical_sweep_figure.py (default matplotlib,
font 10.5, 200 dpi PNG). Writes to <thesis>/images/resultados/:

  large_film_resistance.png  R_eff/(R0 a) against N a^2, films 5 to 80 stick lengths
  large_film_redbonds.png    mean number of red bonds at N a^2 = 5.5 against Lambda
  noise_robustness.png       R^2 for log10 d against the relative noise added

Usage: python thesis_figures_new.py <ColabProject dir> <paper repo dir> <thesis dir>
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

COLAB, PAPER, THESIS = sys.argv[1:4]
OUT = os.path.join(THESIS, "images", "resultados")
plt.rcParams.update({"font.size": 10.5})
COLS = ['#8fb8ea', '#2a78d6', '#1c5aa8', '#0d3b75', '#06203f']
MARKERS = ['o', 's', '^', 'v', 'P']

pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(COLAB, "cluster", "large_film_results", "point_*.json")))]
assert len(pts) == 70
Ls = sorted({p["Lambda"] for p in pts})

# 1. resistance per stick
fig, ax = plt.subplots(figsize=(5.6, 4.3))
for L, c, m in zip(Ls, COLS, MARKERS):
    ps = sorted([p for p in pts if p["Lambda"] == L and p["n"] >= 10], key=lambda p: p["x"])
    y = [np.array(p["R_eff_values"]) / (p["R0"] * p["a"]) for p in ps]
    ax.errorbar([p["x"] for p in ps], [v.mean() for v in y], yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in y],
                color=c, marker=m, ms=4.5, lw=1.3, capsize=2, label=f"$\\Lambda = {L:.0f}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3.7, 21.5)
ax.xaxis.set_major_locator(FixedLocator([4, 5, 6, 8, 10, 12, 16, 20]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_locator(NullLocator())
ax.axvline(5.64, color='k', lw=0.8, ls=':'); ax.text(5.64 * 1.02, 0.11, "5.64", fontsize=8)
ax.grid(alpha=.3, which='major')
ax.set_xlabel("Excluded-area parameter $Na^2$"); ax.set_ylabel("$R_{\\mathrm{eff}}/(R_0 a)$  (units of one stick)")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "large_film_resistance.png"), dpi=200); plt.close(fig)

# 2. red bonds against film size at N a^2 = 5.5
ps = sorted([p for p in pts if p["x_target"] == 5.5], key=lambda p: p["Lambda"])
L = np.array([p["Lambda"] for p in ps]); red = [np.array([m["n_red"] for m in p["networks"]]) for p in ps]
mean = np.array([r.mean() for r in red]); se = np.array([r.std(ddof=1) / np.sqrt(len(r)) for r in red])
b, a0 = np.polyfit(np.log(L), np.log(mean), 1)
fig, ax = plt.subplots(figsize=(5.6, 4.3))
ax.errorbar(L, mean, yerr=se, fmt='o', color=COLS[3], ms=5, capsize=2, label="measured, $Na^2 = 5.5$")
Lg = np.geomspace(4, 100, 50)
ax.plot(Lg, np.exp(a0) * Lg ** b, color=COLS[1], lw=1.2, label=f"fit, slope {b:.2f}")
ax.plot(Lg, mean[0] * (Lg / L[0]) ** 0.75, color='0.4', lw=1, ls='--', label="slope $1/\\nu = 3/4$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.3, which='major')
ax.set_xticks(L); ax.set_xticklabels([f"{v:.0f}" for v in L]); ax.minorticks_off()
ax.set_xlabel("Film size $\\Lambda$ (stick lengths)"); ax.set_ylabel("Mean number of red bonds")
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "large_film_redbonds.png"), dpi=200); plt.close(fig)
print(f"red bonds at N a^2 = 5.5: {np.round(mean, 1).tolist()}, slope {b:.2f}")

# 3. noise robustness
r = json.load(open(os.path.join(PAPER, "figures", "data", "noise_robustness.json")))
sig = np.array(r["sigmas"]) * 100
fig, ax = plt.subplots(figsize=(5.6, 4.9))
for key, lab, c, m in (("slab_3d", "slab, 3D shells", '#2a78d6', 'o'), ("flat_inplane", "flat sticks, in-plane shells", '#eb6834', 's')):
    for k, ls in (("1", "-"), ("24", "--")):
        ax.plot(sig, [r[key][str(s)]["curve+R"][k] for s in r["sigmas"]], color=c, ls=ls, marker=m, ms=4, lw=1.3,
                label=f"{lab}, $k = {k}$")
ax.plot(sig, [r["slab_3d"][str(s)]["R"]["1"] for s in r["sigmas"]], color='0.5', ls=':', lw=1.2, label="$R_{\\mathrm{eff}}$ alone (slab)")
ax.axhline(0, color='0.3', lw=0.8); ax.set_ylim(-0.1, 1.0); ax.grid(alpha=.3)
ax.set_xlabel("Relative noise on each measured value (%)"); ax.set_ylabel("Out-of-sample $R^2$ for $\\log_{10} d$")
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "noise_robustness.png"), dpi=200); plt.close(fig)
print("wrote three figures to", OUT)
