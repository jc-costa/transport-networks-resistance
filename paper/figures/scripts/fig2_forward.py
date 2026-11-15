"""Fig. 2 (double column) and its supplementary companion: where the wire-dominated
scaling R = R0 l g(N a^2) holds.
fig2_forward.pdf (main text):
(a) Large-film sweep (new, sparse solver, cluster/run_large_film_task.py in the
    paper repository): mean R/(R0 l) against N a^2 in networks 5 to 80 stick lengths
    wide; points with fewer than 10 percolating networks are left out.
(b) Same sweep: share of R carried in series by red bonds.
figS_forward.pdf (supplement, Fig. S2):
(a) The 2919 pooled realizations, R/(R0 L a^{q_c}) against N a^2 - c, with the
    percolation-form fit (c free); colored by stick length a.
(b) Junction resistance (thesis data): R(rho)/R(0) for films 5, 6.7 and 10 wide.
Color encodes the film size Lambda = L/l, light to dark.

Inputs (read-only, ColabProject):
  experiments/exp_*/effective_resistance_sweep_summary*.json via cluster/fit_figures.py (pool, fits)
  cluster/large_film_results/point_*.json (Apuana job 17598)
  cluster/junction_results/point_*.json via cluster/junction_analysis.py
"""
import os, sys, glob, json, io, contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from style import COLAB, TWO_COL, CAT, MARK, INK2, R0, grid, panel, save

sys.path.insert(0, os.path.join(COLAB, "cluster"))
import fit_figures as ff            # noqa: E402

with contextlib.redirect_stdout(io.StringIO()) as log:
    N, a, R = ff.pool()
    f = ff.fits(N, a, R)
print(log.getvalue().strip())
k = f["critical"]
assert len(R) == 2919, len(R)

fig, (axb, axc) = plt.subplots(1, 2, figsize=(TWO_COL, 1.75), gridspec_kw=dict(wspace=0.30))
figS, (axs_a, axs_d) = plt.subplots(1, 2, figsize=(TWO_COL, 1.95), gridspec_kw=dict(wspace=0.30))

# supplement (a): percolation-form collapse
ax = axs_a
blues = LinearSegmentedColormap.from_list("b", ["#b9d3f2", "#2a78d6", "#0d3b75"])
norm = Normalize(a.min(), a.max())
xs = f["x"] - k["c"]
ys = R / R0 / a ** k["q"]
sc = ax.scatter(xs, ys, c=a, cmap=blues, norm=norm, s=2.5, alpha=0.55, lw=0, rasterized=True)
xl = np.geomspace(xs.min(), xs.max(), 50)
ax.plot(xl, k["A"] / R0 * xl ** (-k["t"]), color="#0b0b0b", lw=0.9, ls="--",
        label=f"slope $-t$, $t={k['t']:.2f}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$Na^2 - c$")
ax.set_ylabel(r"$R/(R_0 L\, a^{q_c})$")
ax.legend(loc="lower left", handlelength=1.6)
cb = figS.colorbar(sc, ax=ax, pad=0.02, fraction=0.06)
cb.ax.set_title("$a$", fontsize=7.5, pad=2); cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.outline.set_linewidth(0.4); cb.solids.set_alpha(1)
grid(ax); panel(ax, "a", x=-0.14, y=1.05)

# main (a), (b): large-film sweep; color by film size Lambda (log scale), one marker each
LF = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(COLAB, "cluster", "large_film_results", "point_*.json")))]
assert len(LF) == 70, len(LF)
ramp = LinearSegmentedColormap.from_list("L", ["#8fb8ea", "#2a78d6", "#0d3b75", "#06203f"])
LMARK = {5: "o", 6.7: "D", 10: "s", 20: "^", 40: "v", 80: "P"}


def lcol(L):
    return ramp((np.log(L) - np.log(5)) / (np.log(80) - np.log(5)))


XT = [4, 5, 6, 8, 10, 12, 16, 20]


def xaxis(ax):
    ax.set_xscale("log"); ax.set_xlim(3.7, 21.5)
    ax.xaxis.set_major_locator(FixedLocator(XT))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(5.64, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel(r"$Na^2$")


for ax in (axb, axc):
    xaxis(ax)
for L in sorted({p["Lambda"] for p in LF}):
    ps = sorted([p for p in LF if p["Lambda"] == L and p["n"] >= 10], key=lambda p: p["x"])
    x = np.array([p["x"] for p in ps])
    y = [np.array(p["R_eff_values"]) / (p["R0"] * p["a"]) for p in ps]
    sh = [np.array([m["red_R_fraction"] for m in p["networks"]]) for p in ps]
    kw = dict(color=lcol(L), marker=LMARK[round(L, 1)], ms=2.8, lw=0.9, capsize=1.0, elinewidth=0.5,
              mec="white", mew=0.3, label=f"$L = {L:.0f}\,l$")
    axb.errorbar(x, [v.mean() for v in y], yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in y], **kw)
    axc.errorbar(x, [100 * v.mean() for v in sh], yerr=[100 * v.std(ddof=1) / np.sqrt(len(v)) for v in sh], **kw)
axb.set_yscale("log")
axb.set_ylabel(r"$R/(R_0 l)$", labelpad=1)
axb.legend(loc="upper right", handlelength=1.4, fontsize=6.5, labelspacing=0.2)
grid(axb, "y"); panel(axb, "a", x=-0.14)
axc.set_ylim(-3, 80)
axc.set_ylabel(r"red-bond share of $R$ (%)", labelpad=1)
grid(axc, "y"); panel(axc, "b", x=-0.16)

# supplement (b): junction resistance, R(rho)/R(0) in units of one stick, same networks
import junction_analysis as ja  # noqa: E402
rows = [ja.per_point(p) for p in ja.load()]
ax = axs_d
for aa in sorted({r["a"] for r in rows}, reverse=True):
    col, m = lcol(1 / aa), LMARK[round(1 / aa, 1)]
    for xt, ls in ((4, "-"), (20, "--")):
        r = [r for r in rows if r["a"] == aa and r["x_target"] == xt]
        if not r:
            continue
        r = r[0]; rho = np.array(r["rhos"][1:]); y = np.array(r["ratio_mean"][1:])
        ax.plot(rho, y, color=col, ls=ls, lw=0.9, marker=m, ms=2.5, mec="white", mew=0.3)
        k1 = r["rhos"].index(1.0)
        print(f"junction a={aa} Na2={xt}: R(1)/R(0)={r['ratio_mean'][k1]:.2f}, rho*={r['rho_star']:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\rho = R_j/(R_0 l)$")
ax.set_ylabel(r"$R(\rho)/R(0)$", labelpad=1)
ax.text(0.80, 0.80, r"$Na^2=4$", transform=ax.transAxes, ha="right", fontsize=6.5)
ax.text(0.97, 0.42, r"$Na^2=20$", transform=ax.transAxes, ha="right", fontsize=6.5)
ax.minorticks_off()
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], color=lcol(1 / aa), marker=LMARK[round(1 / aa, 1)], ms=2.5, lw=0.9,
                          mec="white", mew=0.3, label=f"$L = {1 / aa:.2g}\\,l$")
                   for aa in sorted({r["a"] for r in rows}, reverse=True)],
          loc="upper left", handlelength=1.4, fontsize=6.5, labelspacing=0.2)
grid(ax); panel(ax, "b", x=-0.14)

save(fig, "fig2_forward.pdf")
save(figS, "figS_forward.pdf")
