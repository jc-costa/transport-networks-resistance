"""
Figure and numbers for the thesis subsection "The near-threshold sweep"
(Fig. "critical-sweep", two subfigures), from the 33 points of
critical_sweep_results/point_NN.json written by run_critical_sweep_task.py.

  (a) mean R_eff / (R0 a), the resistance in units of one stick's resistance,
      against N a^2 for the three box sizes, with one-standard-error bars;
  (b) the running exponent -d ln R_eff / d ln(N a^2 - c) between successive
      points, with c at each box's half-percolation density.

Both panels shade N a^2 < 8, the box-dependent (non-universal) region: below it
the three boxes give different curves and different exponents, above it the
resistance per stick is one function of N a^2 and the exponent of every box sits
near the two-dimensional conductivity exponent.

The half-percolation density is interpolated linearly in logit(rate) between the
two points that bracket a rate of one half. The fits of the same sweep are in
critical_sweep_fit.py.

Usage:
  python critical_sweep_figure.py [--thesis PATH]   default PATH: ~/Desktop/MSc---Thesis
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "critical_sweep_results")
R0 = 1000.0
T_2D = 1.31
X_UNIVERSAL = 8.0
COLS = ['#2a78d6', '#eb6834', '#1baf7a']
MARKERS = ['o', 's', '^']
SHADE = '#d9d9d9'


def load():
    pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "point_*.json")))]
    by_a = {}
    for p in pts:
        by_a.setdefault(p["a"], []).append(p)
    out = {}
    for a, ps in sorted(by_a.items(), reverse=True):
        ps.sort(key=lambda p: p["x"])
        x = np.array([p["x"] for p in ps], float)        # actual N a^2 (N is an integer)
        xt = np.array([p["x_target"] for p in ps], float)
        y = np.array([np.mean(p["R_eff_values"]) for p in ps]) / (R0 * a)
        se = np.array([np.std(p["R_eff_values"], ddof=1) / np.sqrt(p["n"]) for p in ps]) / (R0 * a)
        rate = np.array([p["perc_rate"] for p in ps])
        out[a] = dict(x=x, xt=xt, y=y, se=se, rate=rate, n=sum(p["n"] for p in ps))
    return out


def half_percolation(x, rate):
    lg = np.log(rate / (1 - np.clip(rate, None, 1 - 1e-9)))
    k = np.where((rate[:-1] < 0.5) & (rate[1:] >= 0.5))[0][0]
    return x[k] + (0 - lg[k]) / (lg[k + 1] - lg[k]) * (x[k + 1] - x[k])


def running_exponent(x, y, c):
    ok = x > c
    xs, ys = x[ok], y[ok]
    t = -np.diff(np.log(ys)) / np.diff(np.log(xs - c))
    return np.sqrt(xs[:-1] * xs[1:]), t


def report(data):
    print(f"{sum(len(d['x']) for d in data.values())} points, {sum(d['n'] for d in data.values())} realizations")
    for a, d in data.items():
        c = half_percolation(d["x"], d["rate"])
        d["c"] = c
        xm, t = running_exponent(d["x"], d["y"], c)
        d["xm"], d["t"] = xm, t
        print(f"a={a:.2f} (box {1 / a:.1f} stick lengths): half-percolation c = {c:.2f}")
        print(f"   rate at N a^2 = 4, 5: {d['rate'][0]:.1%}, {d['rate'][1]:.1%}")
        print(f"   running exponent at {np.round(xm, 1).tolist()}: {np.round(t, 2).tolist()}")
    common = sorted(set.intersection(*[set(d["xt"]) for d in data.values()]))
    print("spread of R_eff/(R0 a) between the three boxes, (max - min) / min, at common N a^2:")
    for x in common:
        ys = [d["y"][d["xt"] == x][0] for d in data.values()]
        print(f"   N a^2 = {x:4.0f}: {100 * (max(ys) - min(ys)) / min(ys):5.1f}%   "
              f"(a = {', '.join(f'{v:.3f}' for v in ys)})")


def shade(ax, ymin, ymax, ytext):
    ax.axvspan(ax.get_xlim()[0], X_UNIVERSAL, color=SHADE, alpha=.45, lw=0, zorder=0)
    ax.axvline(X_UNIVERSAL, color='0.45', lw=0.8, ls='-', zorder=0)
    ax.text(np.sqrt(ax.get_xlim()[0] * X_UNIVERSAL), ytext, "non-universal", ha='center', va='center',
            fontsize=8.5, color='0.3', style='italic')
    ax.text(np.sqrt(X_UNIVERSAL * ax.get_xlim()[1]), ytext, "universal", ha='center', va='center',
            fontsize=8.5, color='0.3', style='italic')


def ticks(ax, values):
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def make_panels(data, outdir):
    plt.rcParams.update({'font.size': 10.5})
    xlim = (3.7, 21.5)

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(*xlim); ax.set_ylim(0.09, 16)
    shade(ax, 0.09, 16, 12)
    for (a, d), col, m in zip(data.items(), COLS, MARKERS):
        ax.errorbar(d["x"], d["y"], yerr=d["se"], color=col, marker=m, ms=4.5, lw=1.3, capsize=2,
                    label=f"$a={a:.2f}$ (box {1 / a:.1f} stick lengths)")
    ax.axvline(5.64, color='k', lw=0.8, ls=':')
    ax.text(5.64 * 1.02, 0.11, "5.64", fontsize=8)
    ticks(ax, [4, 5, 6, 8, 10, 12, 16, 20])
    ax.grid(alpha=.3, which='major')
    ax.set_xlabel("Excluded-area parameter $Na^2$")
    ax.set_ylabel("$R_{\\mathrm{eff}}/(R_0 a)$  (units of one stick)")
    ax.legend(fontsize=8, frameon=False, loc='upper right', bbox_to_anchor=(1, 0.92))
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "critical_sweep_resistance.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.set_xscale('log'); ax.set_xlim(*xlim); ax.set_ylim(0, 2.6)
    shade(ax, 0, 2.6, 2.45)
    for (a, d), col, m in zip(data.items(), COLS, MARKERS):
        ax.plot(d["xm"], d["t"], color=col, marker=m, ms=4.5, lw=1.3, label=f"$a={a:.2f}$, $c={d['c']:.2f}$")
    ax.axhline(T_2D, color='k', lw=1, ls='--', label=f"$t={T_2D}$ (2D percolation)")
    ax.axhline(1.0, color='0.4', lw=1, ls=':', label="$t=1$ (dense wire limit)")
    ticks(ax, [4, 5, 6, 8, 10, 12, 16, 20])
    ax.grid(alpha=.3, which='major')
    ax.set_xlabel("Excluded-area parameter $Na^2$")
    ax.set_ylabel("Running exponent $-\\mathrm{d}\\ln R_{\\mathrm{eff}}/\\mathrm{d}\\ln(Na^2-c)$")
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "critical_sweep_exponent.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thesis", default=os.path.expanduser("~/Desktop/MSc---Thesis"),
                    help="thesis repository: images/resultados/ is written")
    args = ap.parse_args()
    outdir = os.path.join(args.thesis, "images", "resultados")
    if not os.path.isdir(outdir):
        sys.exit(f"{outdir} does not exist; pass --thesis")
    data = load()
    report(data)
    make_panels(data, outdir)
    print(f"\nwrote two panels to {outdir}")
