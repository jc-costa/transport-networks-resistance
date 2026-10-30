"""
Redraw the extended-sweep box plot of the thesis (Fig. "extended-boxplots",
images/resultados/box_R_eff_vs_N_by_a_ext.png) from a stored sweep summary,
without re-running the sweep.

The figure is normally produced by notebooks/effective_resistance_vs_N_a.ipynb
Part B, which recomputes the 100-pair extended sweep (hours on a laptop; the
canonical run was made on Apuana). When only the figure's styling changes, this
script rebuilds it from the summary JSON that run wrote, with the same panel
layout as the notebook cell: one panel per a, boxes over N, panel label inside
the axes, no title (the LaTeX caption carries it).

Usage:
    python thesis_figures_2d.py --summary ../experiments/exp_871/effective_resistance_sweep_summary_ext.json --out PATH.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.load(open(args.summary))["sweep"]
    a_values = sorted({r["a"] for r in rows})
    by_a = {a: sorted([r for r in rows if r["a"] == a], key=lambda r: r["N"]) for a in a_values}

    fig, axes = plt.subplots(2, 5, figsize=(13, 7), sharey=True, constrained_layout=True)
    for ax, a in zip(axes.flatten(), a_values):
        nonempty = [r for r in by_a[a] if r["R_eff_values"]]
        ax.text(0.04, 0.95, f"a = {a:.1f}", transform=ax.transAxes, ha="left", va="top", fontsize=13,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        if not nonempty:
            ax.text(0.5, 0.5, "no percolating\ntrials found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            continue
        ax.boxplot([r["R_eff_values"] for r in nonempty], positions=range(len(nonempty)), widths=0.6)
        step = 1 if len(nonempty) <= 6 else 2
        ax.set_xticks(range(0, len(nonempty), step))
        ax.set_xticklabels([r["N"] for r in nonempty][::step], fontsize=10)
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.3, which="both")
    fig.supxlabel("N (number of sticks)", fontsize=14)
    fig.supylabel("R_eff (Ω, log scale)", fontsize=14)
    fig.savefig(args.out, dpi=150, metadata={"Software": None})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
