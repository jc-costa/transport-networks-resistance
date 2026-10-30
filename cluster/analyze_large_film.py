"""
Analysis of the large-film sweep (large_film_results/point_NN.json, written by
run_large_film_task.py). Prints the numbers for the paper and writes one figure.

  1. Collapse: spread of R / (R0 a) between film sizes at each N a^2; the
     thesis found 6-17 % above N a^2 = 8 for sizes 5-10.
  2. Finite-size scaling below the threshold region: slope of log R/(R0 a)
     against log Lambda at fixed N a^2 (theory near the threshold: t/nu ~ 0.98).
  3. Percolation rate against N a^2 per size, and the density at which half of
     the draws percolate (log-odds interpolation, as in the thesis).
  4. Red bonds: mean number and share of R_eff against Lambda.

Usage: python analyze_large_film.py [--results DIR] [--figure out.pdf] [--json out.json]
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(d):
    pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "point_*.json")))]
    by = {}
    for p in pts:
        by.setdefault(p["a"], []).append(p)
    for a in by:
        by[a].sort(key=lambda p: p["x"])
    return by


def stats(p):
    if p["n"] == 0:
        return dict(y=np.nan, se=np.nan, red=np.nan, share=np.nan)
    y = np.array(p["R_eff_values"]) / (p["R0"] * p["a"])
    return dict(y=y.mean(), se=y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan,
                red=np.mean([m["n_red"] for m in p["networks"]]),
                share=np.mean([m["red_R_fraction"] for m in p["networks"]]))


def half_percolation(x, rate):
    x, rate = np.asarray(x), np.clip(np.asarray(rate), 1e-6, 1 - 1e-6)
    k = np.where((rate[:-1] < 0.5) & (rate[1:] >= 0.5))[0]
    if len(k) == 0:
        return np.nan
    k = k[0]; lg = np.log(rate / (1 - rate))
    return x[k] + (0 - lg[k]) / (lg[k + 1] - lg[k]) * (x[k + 1] - x[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "large_film_results"))
    ap.add_argument("--figure", default=None); ap.add_argument("--json", default=None)
    ap.add_argument("--min-n", type=int, default=10, help="networks a point needs to enter the comparisons")
    args = ap.parse_args()
    by = load(args.results)
    sizes = sorted(by, reverse=True)
    out = {"sizes": {}}
    for a in sizes:
        st = [stats(p) for p in by[a]]
        print(f"Lambda = {1/a:.0f}: N a^2 = " + " ".join(f"{p['x_target']:>7}" for p in by[a]))
        print(f"{'':6} " + " ".join(f"{s['y']:7.3f}" for s in st), " R/(R0 a)")
        print(f"{'':6} " + " ".join(f"{p['perc_rate']:7.2f}" for p in by[a]), " percolation rate")
        print(f"{'':6} " + " ".join(f"{s['red']:7.1f}" for s in st), " red bonds")
        xh = half_percolation([p["x"] for p in by[a]], [p["perc_rate"] for p in by[a]])
        print(f"{'':6} half-percolation N a^2 = {xh:.2f}")
        out["sizes"][str(1 / a)] = dict(x=[p["x"] for p in by[a]], n=[p["n"] for p in by[a]],
                                        perc_rate=[p["perc_rate"] for p in by[a]],
                                        y=[s["y"] for s in st], se=[s["se"] for s in st],
                                        red=[s["red"] for s in st], share=[s["share"] for s in st],
                                        half_percolation=xh)
    xts = sorted({p["x_target"] for a in sizes for p in by[a]})
    at = {a: {p["x_target"]: p for p in by[a]} for a in sizes}

    def usable(xt):
        return [(1 / a, stats(at[a][xt])["y"]) for a in sizes if xt in at[a] and at[a][xt]["n"] >= args.min_n]
    print(f"\nspread of R/(R0 a) between sizes with >= {args.min_n} networks, (max - min)/min:")
    for xt in xts:
        u = usable(xt)
        if len(u) >= 2:
            ys = [y for _, y in u]
            print(f"  N a^2 = {xt:>4}: {100 * (max(ys) - min(ys)) / min(ys):5.1f}%  over {len(u)} sizes")
    print("\nfinite-size slope d log(R/R0 a) / d log Lambda:")
    for xt in xts:
        u = usable(xt)
        if len(u) >= 3:
            L, ys = zip(*u)
            print(f"  N a^2 = {xt:>4}: slope {np.polyfit(np.log(L), np.log(ys), 1)[0]:.2f} over Lambda = {[round(l) for l in L]}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, default=float)
    if args.figure:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
        for a in sizes:
            d = out["sizes"][str(1 / a)]
            m = np.array(d["n"]) >= args.min_n
            ax[0].errorbar(np.array(d["x"])[m], np.array(d["y"])[m], yerr=np.array(d["se"])[m], marker="o", ms=3, label=f"L = {1/a:.0f} l")
            ax[1].plot(d["x"], d["perc_rate"], marker="o", ms=3)
            ax[2].plot(np.array(d["x"])[m], 100 * np.array(d["share"])[m], marker="o", ms=3)
        ax[0].set(xscale="log", yscale="log", xlabel="N a^2", ylabel="R/(R0 a)"); ax[0].legend(fontsize=7)
        ax[1].set(xscale="log", xlabel="N a^2", ylabel="percolation rate"); ax[1].axvline(5.64, ls=":", c="k")
        ax[2].set(xscale="log", xlabel="N a^2", ylabel="share of R in red bonds (%)")
        fig.tight_layout(); fig.savefig(args.figure); print("wrote", args.figure)


if __name__ == "__main__":
    main()
