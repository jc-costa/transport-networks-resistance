"""
Single-node probe against face-to-face (check 7 of the thesis, Section "Validation design").

For the reference network (N=40, a=0.6, seed 11) and five further networks of the same
(N, a) (seeds 1 to 5), every pair of one source-strip node and one drain-strip node is read
with the Laplacian pseudoinverse and divided by the face-to-face resistance of the same
network (distance-dependent model). Rayleigh's law requires every ratio to be at least one.
The pair quoted in the text is marked: the first source node against the last drain node on
the reference network (Figure "probe", panel a) and against the first drain node on seeds 1-5.

Usage: python probe_ratio_figure.py [--thesis PATH]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pipeline_utils import generate_percolating_network, distance_dependent_impedance  # noqa: E402
from laplacian_utils import _weighted_laplacian_distance_dependent  # noqa: E402

N, A, R0 = 40, 0.6, 1000.0
NETWORKS = [("reference (seed 11)", 11, -1)] + [(f"seed {s}", s, 0) for s in range(1, 6)]


def ratios(seed):
    res, _ = generate_percolating_network(N, A, k_uniform=R0, seed=seed)
    inter = res["pre_merge_intersections"]; s2n = res["pre_merge_segment_to_nodes"]
    src, drn = res["pre_merge_source_nodes"], res["pre_merge_drain_nodes"]
    face = distance_dependent_impedance(res, R0)["impedance"]
    Lp = np.linalg.pinv(_weighted_laplacian_distance_dependent(np.asarray(inter, float), s2n, R0=R0))
    dg = np.diag(Lp)
    R = dg[:, None] + dg[None, :] - 2 * Lp
    return R[np.ix_(src, drn)] / face, len(src), len(drn), face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", default=os.path.expanduser("~/Desktop/MSc---Thesis"))
    args = ap.parse_args()
    ink, muted, accent, rule = "#1f2430", "#8a93a3", "#1f5fbf", "#b3412f"

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, (label, seed, drain_idx) in enumerate(NETWORKS):
        r, ns, nd, face = ratios(seed)
        y = len(NETWORKS) - 1 - k
        jitter = (np.random.default_rng(seed).random(r.size) - 0.5) * 0.35
        ax.scatter(r.ravel(), np.full(r.size, y) + jitter, s=14, color=muted, alpha=0.7,
                   edgecolors="none", label="every source-drain node pair" if k == 0 else None)
        i0, j0 = 0, (r.shape[1] - 1 if drain_idx == -1 else 0)
        ax.scatter([r[i0, j0]], [y], s=64, color=accent, edgecolors="white", linewidths=1.2, zorder=3,
                   label="pair quoted in the text" if k == 0 else None)
        print(f"{label}: {ns} source x {nd} drain nodes, face {face:.2f} Ohm, min {r.min():.3f}, max {r.max():.3f}, "
              f"quoted pair {r[i0, j0]:.3f}")
    ax.axvline(1.0, color=rule, lw=1.6)
    ax.text(1.0, len(NETWORKS) - 0.35, " ratio = 1: Rayleigh's lower limit", color=rule, fontsize=8.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 14)
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yticks(range(len(NETWORKS)))
    ax.set_yticklabels([n[0] for n in NETWORKS][::-1], fontsize=9, color=ink)
    ax.set_ylim(-0.6, len(NETWORKS) - 0.1)
    ax.set_xlabel(r"$R_{\mathrm{single}} / R_{\mathrm{face}}$ (single-node reading over face-to-face resistance)",
                  fontsize=9.5, color=ink)
    ax.grid(axis="x", color="#d9dde4", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = os.path.join(args.thesis, "images", "resultados", "probe_single_vs_face.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out)


if __name__ == "__main__":
    main()
