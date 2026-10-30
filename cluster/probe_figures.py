"""
Single-pair probes on the reference network of the thesis section
"Single-pair probing and its self-checks" (chapters/resultados): the numbers
quoted there and the two panels of Fig. "probe" (probe_sourcedrain.png,
probe_farpair.png, probe_near.png, probe_mid.png).

The reference network is the two-dimensional draw at N=40, a=0.6, seed 11 of
notebooks/multimeter_r_eff.ipynb (same generator call, so the same network).
Four pairs are probed: the first source-face node against the last drain-face
node; the most resistive pair of the whole network, found by scanning all
pairs with the Laplacian pseudoinverse; and, among interior nodes, the pair
whose separation is closest to 0.1 and the pair closest to 0.4. probe_summary (utils/multimeter.py)
supplies the self-checks and the series-path bound.

Usage:
    python probe_figures.py [--thesis PATH]   writes the two panels into
                                              PATH/images/resultados and prints the numbers
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pipeline_utils import generate_percolating_network, distance_dependent_impedance  # noqa: E402
from multimeter import probe_summary, edge_color_field  # noqa: E402
from laplacian_utils import _weighted_laplacian_distance_dependent  # noqa: E402
from network_visualization import plot_network_colored_by_resistance  # noqa: E402

N, A, SEED, R0 = 40, 0.6, 11, 1000.0


def build_graph(n_nodes, segment_to_nodes):
    g = [[] for _ in range(n_nodes)]
    for e, (n1, n2) in segment_to_nodes.items():
        g[n1].append(e); g[n2].append(e)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", default=os.path.expanduser("~/Desktop/MSc---Thesis"))
    args = ap.parse_args()
    outdir = os.path.join(args.thesis, "images", "resultados")

    res, attempts = generate_percolating_network(N, A, k_uniform=R0, seed=SEED)
    inter = res["pre_merge_intersections"]; s2n = res["pre_merge_segment_to_nodes"]
    graph = build_graph(len(inter), s2n)
    src, drn = res["pre_merge_source_nodes"], res["pre_merge_drain_nodes"]
    face = distance_dependent_impedance(res, R0)["impedance"]
    print(f"reference network: N={N} a={A} seed={SEED} boundary={res.get('boundary')} attempts={attempts} "
          f"sticks={res.get('n_sticks')} nodes={len(inter)} edges={len(s2n)} whole-face R_eff={face:.2f} Ohm")

    # every pairwise resistance at once, for the range and the most resistive pair
    Lp = np.linalg.pinv(_weighted_laplacian_distance_dependent(np.asarray(inter, float), s2n, R0=R0))
    dg = np.diag(Lp); Rall = dg[:, None] + dg[None, :] - 2 * Lp
    iu = np.triu_indices(len(inter), 1)
    vals = Rall[iu]
    k = int(np.argmax(vals)); far = (int(iu[0][k]), int(iu[1][k]))
    print(f"all {len(vals)} node pairs: min {vals.min():.2f}, max {vals.max():.2f} Ohm at pair {far}")

    # Two further probes at short and intermediate separation, so the figure also shows
    # what a measurement between nearby nodes looks like: among interior nodes
    # (0.1 < x < 0.9, so neither probe sits on a terminal face) the pair whose Euclidean
    # separation is closest to 0.1 and the pair closest to 0.4.
    P = np.asarray(inter, float)
    interior = np.where((P[:, 0] > 0.1) & (P[:, 0] < 0.9))[0]
    iu2 = np.triu_indices(len(interior), 1)
    sep = np.linalg.norm(P[interior[iu2[0]]] - P[interior[iu2[1]]], axis=1)
    near = tuple(int(interior[k]) for k in (iu2[0][np.argmin(abs(sep - 0.1))], iu2[1][np.argmin(abs(sep - 0.1))]))
    mid = tuple(int(interior[k]) for k in (iu2[0][np.argmin(abs(sep - 0.4))], iu2[1][np.argmin(abs(sep - 0.4))]))
    pairs = {"sourcedrain": (src[0], drn[-1]), "farpair": far, "near": near, "mid": mid}
    for tag, (i, j) in pairs.items():
        s = probe_summary(inter, graph, s2n, i, j, R0=R0, distance_dependent=True)
        sp = s["series_path"]
        print(f"[{tag}] nodes {i},{j}: R_dd={s['r_eff_dd']:.2f} R_uniform={s['r_eff_uniform']:.2f} Ohm, "
              f"separation {s['probe_distance']:.3f}; pinv dev {s['pinv_deviation']:.1e}, model dev {s['model_deviation']:.1e}, "
              f"KCL {s['kcl_residual']:.1e} A; series path {sp['hops']} hops {sp['resistance']:.2f} Ohm, "
              f"parallel factor {s['parallel_factor']:.2f}")
        values, cbar_label, note = edge_color_field(s["probe"], quantity="current")
        fig = plot_network_colored_by_resistance(
            inter, s2n, values, title="",  # the thesis caption carries the pair and its reading
            figsize=(8, 6.5), log_scale=True, cbar_label=cbar_label, highlight_nodes=[i, j], show_node_ids=False)
        fig.savefig(os.path.join(outdir, f"probe_{tag}.png"), dpi=200, bbox_inches="tight", metadata={"Software": None})
        plt.close(fig)
    print(f"wrote probe_sourcedrain.png, probe_farpair.png, probe_near.png and probe_mid.png to {outdir}")


if __name__ == "__main__":
    main()
