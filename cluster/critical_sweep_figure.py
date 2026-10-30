"""
Figures and numbers for the thesis subsection "The near-threshold sweep", from
the 33 points of critical_sweep_results/point_NN.json written by
run_critical_sweep_task.py, and the 70 points of large_film_results/point_NN.json
written by run_large_film_task.py.

  critical_sweep_resistance.png  mean R_eff / (R0 a), the resistance in units of
      one stick's resistance, against N a^2 for the three film sizes of the
      near-threshold sweep, with one-standard-error bars (Fig. "critical-sweep" (a));
  large_film_resistance.png  the same for the five film sizes of the large-film
      sweep, points with fewer than 10 networks left out (Fig. "critical-sweep" (b));
  critical_sweep_exponent.png  the running exponent -d ln R_eff / d ln(N a^2 - c)
      between successive points of the near-threshold sweep, with c at each film's
      half-percolation density (Appendix C).

Each film size Lambda = 1/a keeps one color and marker in every panel, steps of
one blue ramp from light (small films) to dark (large films). The near-threshold
panels shade N a^2 < 8, where the three film sizes of that sweep disagree.

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
LARGE_DIR = os.path.join(HERE, "large_film_results")
MIN_N = 10
R0 = 1000.0
T_2D = 1.31
X_UNIVERSAL = 8.0
# film size Lambda -> (color, marker); six even steps of one blue ramp
STYLE = {5: ('#86b6ef', 'o'), 6.7: ('#5a9be8', 'D'), 10: ('#307edc', 's'),
         20: ('#2164b7', '^'), 40: ('#164c91', 'v'), 80: ('#0d366b', 'P')}
YLIM = (0.09, 80)
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


def style(a):
    lam = round(1 / a, 1)
    return lam, *STYLE[int(lam) if lam.is_integer() else lam]


def load_large_film():
    by_a = {}
    for p in (json.load(open(f)) for f in sorted(glob.glob(os.path.join(LARGE_DIR, "point_*.json")))):
        if p["n"] >= MIN_N:
            by_a.setdefault(p["a"], []).append(p)
    out = {}
    for a, ps in sorted(by_a.items(), reverse=True):
        ps.sort(key=lambda p: p["x"])
        y = [np.array(p["R_eff_values"]) / (p["R0"] * a) for p in ps]
        out[a] = dict(x=np.array([p["x"] for p in ps]), y=np.array([v.mean() for v in y]),
                      se=np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in y]))
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
    ax.text(np.sqrt(ax.get_xlim()[0] * X_UNIVERSAL), ytext, "$Na^2 < 8$", ha='center', va='center',
            fontsize=8.5, color='0.3', style='italic')
    ax.text(np.sqrt(X_UNIVERSAL * ax.get_xlim()[1]), ytext, "$Na^2 \\geq 8$", ha='center', va='center',
            fontsize=8.5, color='0.3', style='italic')


def ticks(ax, values):
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def resistance_panel(data, outdir, name, shaded):
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(3.7, 21.5); ax.set_ylim(*YLIM)
    if shaded:
        shade(ax, *YLIM, 50)
    for a, d in data.items():
        lam, col, m = style(a)
        ax.errorbar(d["x"], d["y"], yerr=d["se"], color=col, marker=m, ms=4.5, lw=1.3, capsize=2,
                    label=f"$\\Lambda = {lam:g}$")
    ax.axvline(5.64, color='k', lw=0.8, ls=':')
    ax.text(5.64 * 1.02, 0.11, "5.64", fontsize=8)
    ticks(ax, [4, 5, 6, 8, 10, 12, 16, 20])
    ax.grid(alpha=.3, which='major')
    ax.set_xlabel("Excluded-area parameter $Na^2$")
    ax.set_ylabel("$R_{\\mathrm{eff}}/(R_0 a)$  (units of one stick)")
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, name), dpi=200); plt.close(fig)


def make_panels(data, large, outdir):
    plt.rcParams.update({'font.size': 9.5})
    resistance_panel(data, outdir, "critical_sweep_resistance.png", shaded=True)
    resistance_panel(large, outdir, "large_film_resistance.png", shaded=False)

    plt.rcParams.update({'font.size': 10.5})
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.set_xscale('log'); ax.set_xlim(3.7, 21.5); ax.set_ylim(0, 2.6)
    shade(ax, 0, 2.6, 2.45)
    for a, d in data.items():
        lam, col, m = style(a)
        ax.plot(d["xm"], d["t"], color=col, marker=m, ms=4.5, lw=1.3, label=f"$\\Lambda = {lam:g}$, $c={d['c']:.2f}$")
    ax.axhline(T_2D, color='k', lw=1, ls='--', label=f"$t={T_2D}$ (2D conductivity exponent)")
    ax.axhline(1.0, color='0.4', lw=1, ls=':', label="$t=1$ (dense-wire value)")
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
    make_panels(data, load_large_film(), outdir)
    print(f"\nwrote three panels to {outdir}")
