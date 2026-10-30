"""
Fit the percolation critical form to the near-threshold sweep produced by
run_critical_sweep_task.py (critical_sweep_results/point_NN.json), per box
size and pooled, and print the numbers the thesis quotes.

    R_eff = A * a^q * (N a^2 - c)^(-t)

Per box size a the prefactor a^q is a constant, so each size is fitted as
log R = log A_a - t log(N a^2 - c) with (A_a, t, c) free; c is then the
effective threshold of that box. The pooled fit adds q. Intervals are a
cluster bootstrap over the (a, N) points. A window option restricts the fit
to N a^2 <= x_max, to show how t moves as the fit is confined to the
critical region.

Usage:  python critical_sweep_fit.py [--xmax 10] [--figure out.png]
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "critical_sweep_results")
T_2D = 1.31  # 2D conductivity exponent, t/nu = 0.9826 (Grassberger 1999), nu = 4/3


def load(xmax=None, results_dir=None):
    results_dir = results_dir or RESULTS_DIR
    pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results_dir, "point_*.json")))]
    pts = [p for p in pts if p["n"] >= 3 and (xmax is None or p["x"] <= xmax + 1e-9)]
    return pts


def fit_size(pts):
    """One box size: log R = lA - t log(x - c); returns (lA, t, c, r2, n)."""
    x = np.concatenate([[p["x"]] * p["n"] for p in pts]); lR = np.log10(np.concatenate([p["R_eff_values"] for p in pts]))
    r = least_squares(lambda q: q[0] - q[1] * np.log10(np.clip(x - q[2], 1e-9, None)) - lR,
                      [3.0, 1.0, min(x.min() - 0.5, 2.0)], bounds=([-10, 0, 0], [10, 10, x.min() - 1e-6]))
    pred = r.x[0] - r.x[1] * np.log10(x - r.x[2])
    r2 = 1 - np.sum((lR - pred) ** 2) / np.sum((lR - lR.mean()) ** 2)
    return r.x, r2, len(lR)


def fit_pooled(pts):
    """All sizes: log R = lA + q log a - t log(x - c)."""
    x = np.concatenate([[p["x"]] * p["n"] for p in pts]); a = np.concatenate([[p["a"]] * p["n"] for p in pts])
    lR = np.log10(np.concatenate([p["R_eff_values"] for p in pts]))
    r = least_squares(lambda q: q[0] + q[1] * np.log10(a) - q[2] * np.log10(np.clip(x - q[3], 1e-9, None)) - lR,
                      [3.0, 1.0, 1.0, min(x.min() - 0.5, 2.0)], bounds=([-10, -10, 0, 0], [10, 10, 10, x.min() - 1e-6]))
    pred = r.x[0] + r.x[1] * np.log10(a) - r.x[2] * np.log10(x - r.x[3])
    r2 = 1 - np.sum((lR - pred) ** 2) / np.sum((lR - lR.mean()) ** 2)
    return r.x, r2, len(lR)


def bootstrap(pts, fitter, B=300, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        sample = [pts[i] for i in rng.integers(0, len(pts), len(pts))]
        try:
            out.append(fitter(sample)[0])
        except Exception:
            pass
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmax", type=float, default=None, help="fit only points with N a^2 <= xmax")
    ap.add_argument("--figure", default=None, help="write the per-size collapse figure here")
    ap.add_argument("--results", default=RESULTS_DIR, help="results directory (default critical_sweep_results)")
    args = ap.parse_args()

    pts = load(args.xmax, args.results)
    if not pts:
        sys.exit("no points found (run the array first, or check --xmax)")
    sizes = sorted(set(p["a"] for p in pts), reverse=True)
    print(f"{len(pts)} points, {sum(p['n'] for p in pts)} realizations"
          + (f", N a^2 <= {args.xmax}" if args.xmax else ""))
    print(f"\nper box size (box = 1/a stick lengths across):")
    per_size = {}
    for a in sizes:
        ps = [p for p in pts if p["a"] == a]
        (lA, t, c), r2, n = fit_size(ps)
        lo, hi = bootstrap(ps, fit_size)
        per_size[a] = dict(t=t, c=c, r2=r2, n=n, t_ci=(lo[1], hi[1]), c_ci=(lo[2], hi[2]))
        rates = ", ".join(f"{p['x']:.0f}:{p['perc_rate']:.1%}" for p in sorted(ps, key=lambda p: p["x"])[:4])
        print(f"  a={a:.2f} (L={1/a:4.1f}): {len(ps):2d} points, {n:4d} realizations   "
              f"t = {t:.3f} [{lo[1]:.2f}, {hi[1]:.2f}]   c = {c:.2f} [{lo[2]:.2f}, {hi[2]:.2f}]   R^2 = {r2:.3f}"
              f"   (perc. rate at lowest x: {rates})")
    (lA, q, t, c), r2, n = fit_pooled(pts)
    lo, hi = bootstrap(pts, fit_pooled)
    print(f"\npooled over sizes: q = {q:.3f} [{lo[1]:.2f}, {hi[1]:.2f}]   t = {t:.3f} [{lo[2]:.2f}, {hi[2]:.2f}]   "
          f"c = {c:.2f} [{lo[3]:.2f}, {hi[3]:.2f}]   R^2 = {r2:.3f}")
    print(f"reference: 2D conductivity exponent t = {T_2D}")

    summary = {"xmax": args.xmax, "per_size": {str(a): v for a, v in per_size.items()},
               "pooled": dict(lA=lA, q=q, t=t, c=c, r2=r2, n=n, q_ci=(lo[1], hi[1]), t_ci=(lo[2], hi[2]), c_ci=(lo[3], hi[3]))}
    out = os.path.join(args.results, f"critical_fit_summary{'_x' + str(args.xmax) if args.xmax else ''}.json")
    json.dump(summary, open(out, "w"), indent=2, default=float)
    print(f"wrote {out}")

    if args.figure:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for a in sizes:
            ps = [p for p in pts if p["a"] == a]; v = per_size[a]
            xs = np.array([p["x"] for p in ps]) - v["c"]
            means = np.array([np.mean(p["R_eff_values"]) for p in ps])
            sems = np.array([np.std(p["R_eff_values"], ddof=1) / np.sqrt(p["n"]) if p["n"] > 1 else 0 for p in ps])
            ax.errorbar(xs, means, yerr=sems, fmt="o", capsize=3, label=f"a = {a:.2f}  (t = {v['t']:.2f}, c = {v['c']:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N a^2 - c  (c per box size)"); ax.set_ylabel("mean R_eff (ohm)")
        ax.legend(); ax.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        fig.savefig(args.figure, dpi=150, metadata={"Software": None}); print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
