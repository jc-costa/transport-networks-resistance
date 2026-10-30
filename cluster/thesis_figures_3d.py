"""
Thesis figures drawn from the scalar (N, a, d) grid: the a = 0.2 slice of mean
R_eff against film thickness d, one curve per stick count, and the raw
percolation rate of the same combinations. Reads cluster/effective_resistance_d_results
(or --results) and writes images/resultados/reff_vs_d_a02_{reff,rate}.png in
the thesis repository (--thesis).
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--results", default=os.path.join(HERE, "effective_resistance_d_results"))
ap.add_argument("--thesis", default=os.path.join(HERE, "..", "..", "MSc---Thesis")); ap.add_argument("--a", type=float, default=0.2)
args = ap.parse_args()
rows = []
for p in sorted(glob.glob(os.path.join(args.results, "chunk_*.json"))): rows += json.load(open(p))
sl = [r for r in rows if abs(r["a"] - args.a) < 1e-9]
Ns = sorted(set(r["N"] for r in sl)); ds = sorted(set(r["d"] for r in sl))
cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(Ns)))
out = os.path.join(args.thesis, "images", "resultados"); meta = {"Software": None}

fig, ax = plt.subplots(figsize=(6.5, 5))
for N, col in zip(Ns, cmap):
    pts = sorted([r for r in sl if r["N"] == N and r["n"] > 0], key=lambda r: r["d"])
    if not pts: continue
    x = [r["d"] for r in pts]; y = [np.mean(r["R_eff_values"]) for r in pts]
    e = [np.std(r["R_eff_values"], ddof=1) / np.sqrt(r["n"]) if r["n"] > 1 else 0 for r in pts]
    ax.errorbar(x, y, yerr=e, fmt="o-", capsize=3, color=col, label=f"$N$={N}")
# no draw percolates beyond the last d with data: show d up to the next sampled value and shade the gap
d_last = max(r["d"] for r in sl if r["n"] > 0); d_next = min(d for d in ds if d > d_last)
xs = [d for d in ds if d <= d_next]
ax.set_xscale("log"); ax.set_xlim(xs[0] / 1.05, d_next * 1.05)
ax.set_xticks(xs); ax.set_xticklabels([f"{d:g}" for d in xs]); ax.minorticks_off()
ax.axvspan(d_last, d_next * 1.05, color="0.9", lw=0, zorder=0)
ax.text(np.sqrt(d_last * d_next), 0.5, f"no percolating\nnetworks\nfor $d > {d_last:g}$", transform=ax.get_xaxis_transform(),
        ha="center", va="center", color="0.3", style="italic")
ax.set_xlabel("Film thickness $d$"); ax.set_ylabel(r"$R_{\mathrm{eff}}$ ($\Omega$)"); ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
plt.tight_layout(); fig.savefig(os.path.join(out, "reff_vs_d_a02_reff.png"), dpi=150, metadata=meta); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5))
for N, col in zip(Ns, cmap):
    pts = sorted([r for r in sl if r["N"] == N], key=lambda r: r["d"])
    ax.plot([r["d"] for r in pts], [r["perc_rate"] for r in pts], "o-", color=col, label=f"$N$={N}")
ax.set_xscale("log"); ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds]); ax.minorticks_off()
ax.set_xlabel("Film thickness $d$"); ax.set_ylabel("Percolation rate"); ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3); ax.legend(ncol=2)
plt.tight_layout(); fig.savefig(os.path.join(out, "reff_vs_d_a02_rate.png"), dpi=150, metadata=meta); plt.close(fig)
for N in Ns:
    pts = sorted([r for r in sl if r["N"] == N], key=lambda r: r["d"])
    print(f"N={N:3d}: " + "  ".join(f"d={r['d']}: {np.mean(r['R_eff_values']):6.0f} ({r['n']:2d}, {r['perc_rate']:.0%})" if r["n"] else f"d={r['d']}: none" for r in pts))
