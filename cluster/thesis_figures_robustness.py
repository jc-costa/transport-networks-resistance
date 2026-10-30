"""Thesis figures of the 1500-network robustness batch (cluster/random_networks_results):
percolation outcome over the sampled (N, a) pairs, and R_eff of the solved cases against a
and against N. Writes percolation_left.png, percolation_right.png, percolation_reff_vs_N.png."""
import os, json, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--results", default=os.path.join(HERE, "random_networks_results")); ap.add_argument("--thesis", required=True); args = ap.parse_args()
rows = []
for p in sorted(glob.glob(os.path.join(args.results, "chunk_*.json"))): rows += json.load(open(p))
out = os.path.join(args.thesis, "images", "resultados"); meta = {"Software": None}
N = np.array([r["N"] for r in rows]); A = np.array([r["a"] for r in rows]); perc = np.array([bool(r["has_percolation"]) for r in rows])
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(A[~perc], N[~perc], s=38, c="lightgray", edgecolors="k", linewidths=0.4, label="no percolation")
ax.scatter(A[perc], N[perc], s=38, c="seagreen", edgecolors="k", linewidths=0.4, label="percolating")
ax.set_xlabel("a (stick length)"); ax.set_ylabel("N (number of sticks)"); plt.tight_layout(); fig.savefig(os.path.join(out, "percolation_left.png"), dpi=150, metadata=meta); plt.close(fig)
sol = [r for r in rows if r["has_percolation"] and r["effective_resistance"] is not None]
R = np.array([r["effective_resistance"] for r in sol]); Ns = np.array([r["N"] for r in sol]); As = np.array([r["a"] for r in sol])
for x, name, lab in ((As, "percolation_right.png", "a (stick length)"), (Ns, "percolation_reff_vs_N.png", "N (number of sticks)")):
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); ax.scatter(x, R, s=22, c="steelblue", edgecolors="k", linewidths=0.3, alpha=0.8)
    ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel(r"$R_{\mathrm{eff}}$ ($\Omega$)"); ax.grid(True, alpha=0.3, which="both"); plt.tight_layout(); fig.savefig(os.path.join(out, name), dpi=150, metadata=meta); plt.close(fig)
print(f"{len(rows)} cases, {perc.sum()} percolating, {len(sol)} solved")
