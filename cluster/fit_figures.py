"""
Figures and numbers for the thesis section on the scaling fits (Tables "fit-results" and
"fit-critical", Figs. "fit-scaling" and "fit-critical-pair", two panels each).

The pooling and the three fits are those of notebooks/inverse_N_a_from_R_eff.ipynb
(sections 2-4): clipped-rule sweep summaries from exp_843 onward, each distinct cell
pooled once, log10 least squares over individual percolating trials. They are repeated
here so the four panels can be regenerated in the thesis' figure style without running
the notebook, and every number shown in a legend is printed so it can be checked
against the tables.

Panels written to <thesis>/images/resultados/:
  scaling_fit_collapse-2.png           (a) of fit-scaling: R_eff against N a^2, collapse fit
  scaling_fit_comparison-2.png         (b) of fit-scaling: predicted vs measured, both power laws
  scaling_fit_critical_collapse-2.png  (a) of fit-critical-pair: R_eff / a^q_c against N a^2 - c
  scaling_fit_comparison_critical.png  (b) of fit-critical-pair: predicted vs measured, critical vs general

Usage:
  python fit_figures.py [--thesis PATH]   default PATH: ~/Desktop/MSc---Thesis
"""
import os
import re
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(HERE, "..", "experiments")
MERGE_FIX_FIRST_EXPERIMENT = 843   # first sweep after the merge-geometry fix
BOUNDARY_RULE = "clip"
C_INF = 5.64
COLS = ['#2a78d6', '#eb6834']


# ---------------------------------------------------------------- pooling (notebook section 2)
def pool():
    paths = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "exp_*", "effective_resistance_sweep_summary*.json")))
    merged, seen = defaultdict(list), set()
    used = 0
    for p in paths:
        if int(re.search(r"exp_(\d+)", p).group(1)) < MERGE_FIX_FIRST_EXPERIMENT:
            continue
        summary = json.load(open(p))
        if summary.get("config", {}).get("boundary") != BOUNDARY_RULE:
            continue
        used += 1
        for rec in summary.get("sweep", []):
            vals = rec.get("R_eff_values") or []
            if not vals:
                continue
            key = (rec["N"], round(rec["a"], 3))
            # a seeded re-run reproduces the trials, but a run on another machine only to
            # ~1e-12 (exp_883 on the laptop vs exp_870 on the cluster), so compare at 9 digits
            cell = (key, tuple(float(f"{v:.9g}") for v in vals))
            if cell in seen:          # seeded re-run of a cell already pooled
                continue
            seen.add(cell)
            merged[key].extend(vals)
    N = np.concatenate([[k[0]] * len(v) for k, v in merged.items()]).astype(float)
    a = np.concatenate([[k[1]] * len(v) for k, v in merged.items()])
    R = np.concatenate([v for v in merged.values()])
    print(f"pooled {used} summaries, {len(merged)} (N, a) cells, {len(R)} trials")
    return N, a, R


def r2(y, pred):
    return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------- fits (notebook sections 3-4)
def fits(N, a, R):
    x, lR = N * a ** 2, np.log10(R)
    s, b = np.polyfit(np.log10(x), lR, 1)
    pred_c = s * np.log10(x) + b
    X = np.column_stack([np.ones_like(lR), np.log10(N), np.log10(a)])
    coef = np.linalg.lstsq(X, lR, rcond=None)[0]
    pred_g = X @ coef

    def crit_pred(pr):
        lA, q, t, c = pr
        return lA + q * np.log10(a) - t * np.log10(np.clip(x - c, 1e-6, None))
    res = least_squares(lambda pr: crit_pred(pr) - lR, [3.0, 0.0, 1.0, 2.0],
                        bounds=([-10, -10, 0, 0], [10, 10, 10, x.min() - 1e-3]))
    crit = res.x
    pred_k = crit_pred(crit)

    out = dict(x=x, lR=lR,
               collapse=dict(A=10 ** b, s=s, r2=r2(lR, pred_c), pred=pred_c),
               general=dict(A=10 ** coef[0], p=coef[1], q=coef[2], r2=r2(lR, pred_g), pred=pred_g),
               critical=dict(A=10 ** crit[0], q=crit[1], t=crit[2], c=crit[3], r2=r2(lR, pred_k), pred=pred_k))
    c, g, k = out["collapse"], out["general"], out["critical"]
    print(f"collapse:  R = {c['A']:.0f} (N a^2)^{c['s']:.3f}            R2 = {c['r2']:.3f}")
    print(f"general:   R = {g['A']:.0f} N^{g['p']:.3f} a^{g['q']:.3f}      R2 = {g['r2']:.3f}")
    print(f"critical:  R = {k['A']:.0f} a^{k['q']:.3f} (N a^2 - {k['c']:.2f})^-{k['t']:.3f}   R2 = {k['r2']:.3f}")
    return out


# ---------------------------------------------------------------- figure
def scatter_by_a(ax, fig, xs, ys, a):
    sc = ax.scatter(xs, ys, c=a, cmap='plasma', s=12, alpha=0.6, edgecolor='none')
    fig.colorbar(sc, ax=ax, label="Stick length $a$")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(alpha=.3, which='both')


def pred_vs_measured(ax, lR, series):
    for col, (pred, lab) in zip(COLS, series):
        ax.scatter(lR, pred, s=7, alpha=0.4, color=col, edgecolor='none', label=lab)
    lims = [lR.min() - 0.1, lR.max() + 0.1]
    ax.plot(lims, lims, 'k--', lw=1, alpha=.6, label="perfect prediction")
    ax.set_aspect('equal')
    ax.grid(alpha=.3)
    ax.set_xlabel("Measured $\\log_{10} R_{\\mathrm{eff}}$ ($\\Omega$)")
    ax.set_ylabel("Predicted $\\log_{10} R_{\\mathrm{eff}}$ ($\\Omega$)")
    leg = ax.legend(fontsize=8, frameon=False, loc='upper left', markerscale=2)
    for h in leg.legend_handles:
        h.set_alpha(1)


def make_panels(N, a, R, f, outdir):
    plt.rcParams.update({'font.size': 10.5})
    x, lR = f["x"], f["lR"]
    c, g, k = f["collapse"], f["general"], f["critical"]
    xl = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter_by_a(ax, fig, x, R, a)
    ax.plot(xl, c["A"] * xl ** c["s"], 'k--', lw=1.8,
            label=f"$R_{{\\mathrm{{eff}}}} = {c['A']:.0f}\\,(Na^2)^{{{c['s']:.3f}}}$, $R^2={c['r2']:.3f}$")
    ax.set_xlabel("Excluded-area parameter $Na^2$")
    ax.set_ylabel("$R_{\\mathrm{eff}}$ ($\\Omega$)")
    ax.legend(fontsize=8.5, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "scaling_fit_collapse-2.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    pred_vs_measured(ax, lR, [(c["pred"], f"$Na^2$ collapse, $R^2={c['r2']:.3f}$"),
                              (g["pred"], f"general $N^p a^q$, $R^2={g['r2']:.3f}$")])
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "scaling_fit_comparison-2.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    xs = x - k["c"]
    scatter_by_a(ax, fig, xs, R / a ** k["q"], a)
    xsl = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 100)
    ax.plot(xsl, k["A"] * xsl ** (-k["t"]), 'k--', lw=1.8,
            label=f"slope $-t={-k['t']:.2f}$, $R^2={k['r2']:.3f}$")
    ax.set_xlabel(f"$Na^2 - c$  ($c={k['c']:.2f}$)")
    ax.set_ylabel(f"$R_{{\\mathrm{{eff}}}}/a^{{q_c}}$  ($q_c={k['q']:.2f}$)")
    ax.legend(fontsize=8.5, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "scaling_fit_critical_collapse-2.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    pred_vs_measured(ax, lR, [(g["pred"], f"general $N^p a^q$, $R^2={g['r2']:.3f}$"),
                              (k["pred"], f"critical form, $c$ free, $R^2={k['r2']:.3f}$")])
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "scaling_fit_comparison_critical.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thesis", default=os.path.expanduser("~/Desktop/MSc---Thesis"),
                    help="thesis repository: images/resultados/ is written")
    args = ap.parse_args()
    outdir = os.path.join(args.thesis, "images", "resultados")
    if not os.path.isdir(outdir):
        sys.exit(f"{outdir} does not exist; pass --thesis")
    N, a, R = pool()
    f = fits(N, a, R)
    make_panels(N, a, R, f, outdir)
    print(f"\nwrote four panels to {outdir}")
