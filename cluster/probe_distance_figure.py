"""
Figure and numbers for the thesis subsection "Dependence of R_eff,ij on probe
separation" (Fig. "probe-distance", three two-dimensional
subfigures, and Fig. "probe-distance-3d").

The question behind it: is R_eff,ij linear in the Euclidean separation of the
probes, and if not, why not. Two families of networks answer it.

  * The three-dimensional preview onion curves of cluster/preview_curves_results/
    (three (N, a, d) combinations, 50 realizations each, Fig. "onion-preview"),
    replotted on linear axes and fitted.
  * The twenty two-dimensional networks embedded in the thesis' supplementary
    page (docs/index.html of the thesis repository, five (N, a) combinations,
    four seeds each), on which the shell protocol of run_onion_protocol_pinv is
    re-run from every source-boundary node and from every node near the box
    centre, together with the all-pairs parallel factor d_ij / R_eff,ij (series
    path bound over measured resistance) and the node-density profile against
    distance to the box edge.

Every number quoted in that subsection is printed by this script, so the text
can be checked against it line by line. The four panels are written to
<thesis>/images/resultados/probe_distance_{3d_linear,2d_linear,
parallel_factor,density_profile}.png.

Usage:
  python probe_distance_figure.py [--thesis PATH]   default PATH: ~/Desktop/MSc---Thesis

Only numpy and matplotlib are needed; the 2D solve is the same weighted
Laplacian pseudoinverse as utils/laplacian_utils.py, rebuilt here from the
node/edge lists of the supplementary page so the script runs without the
package.
"""
import os
import sys
import json
import glob
import heapq
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "preview_curves_results")
R0 = 1000.0
SHELLS = np.arange(1, 10) / 10           # 0.1 ... 0.9, tolerance 0.05, as in the onion protocol
TOL = 0.05
PREVIEW_COMBOS = [(150, 0.2, 0.1), (150, 0.2, 0.15), (125, 0.2, 0.1)]
KEYS = [str(round(0.1 * n, 2)) for n in range(1, 10)]
COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MARKERS = ['o', 's', '^', 'D', 'v']


# ---------------------------------------------------------------- 3D preview curves
def preview_curves():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "chunk_*.json"))):
        rows.extend(json.load(open(p))["results"])
    out = {}
    for ci, combo in enumerate(PREVIEW_COMBOS):
        M = np.array([[x["layer_means"].get(k, np.nan) for k in KEYS]
                      for x in rows if x["combo_index"] == ci and x["percolated"]])
        n = np.sum(~np.isnan(M), 0)
        out[combo] = (np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(n), n)
    return out


def report_3d(curves):
    print("=== Three-dimensional onion curves, linear axes ===")
    r = SHELLS
    for combo, (mu, se, n) in curves.items():
        gamma = np.polyfit(np.log(r), np.log(mu), 1)[0]
        fits = {}
        for name, X in [("linear", r), ("sqrt", np.sqrt(r)), ("log", np.log(r))]:
            A = np.c_[X, np.ones_like(X)]
            c, *_ = np.linalg.lstsq(A, mu, rcond=None)
            res = mu - A @ c
            fits[name] = (c[0], c[1], 1 - res.var() / mu.var())
        k = np.linalg.lstsq(np.c_[r], mu, rcond=None)[0][0]
        chi2 = (((mu - k * r) / se) ** 2).sum() / (len(r) - 1)
        print(f"{combo}: realizations per shell {n.tolist()}")
        print(f"   means      {np.round(mu).astype(int).tolist()}")
        print(f"   SE         {np.round(se).astype(int).tolist()}")
        print(f"   increments {np.round(np.diff(mu)).astype(int).tolist()}")
        print(f"   power-law exponent {gamma:.2f}; ratio R(0.9)/R(0.1) = {mu[-1] / mu[0]:.2f}")
        for name, (a, b, r2) in fits.items():
            print(f"   {name:6s} fit: slope {a:.0f}, intercept {b:.0f} ohm, R2 {r2:.4f}")
        print(f"   line through origin: slope {k:.0f} ohm, reduced chi2 {chi2:.2f}")


# ---------------------------------------------------------------- 2D supplement
def load_supplement(thesis):
    path = os.path.join(thesis, "docs", "index.html")
    s = open(path).read()
    s = s[s.index('<script id="netdata"'):]
    s = s[s.index('>') + 1:]
    s = s[:s.index('</script>')]
    return json.loads(s)["networks"]


def solve_2d(net):
    """All-pairs R_eff, Euclidean distances and series-path bound for one network."""
    P = np.array(net["nodes"])
    E = np.array(net["edges"])
    n = len(P)
    length = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    w = 1.0 / (R0 * length)
    L = np.zeros((n, n))
    for (u, v), ww in zip(E, w):
        L[u, u] += ww; L[v, v] += ww; L[u, v] -= ww; L[v, u] -= ww
    Lp = np.linalg.pinv(L)
    dg = np.diag(Lp)
    R = dg[:, None] + dg[None, :] - 2 * Lp
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    adj = defaultdict(list)
    for (u, v), ww in zip(E, w):
        adj[u].append((v, 1 / ww)); adj[v].append((u, 1 / ww))
    SP = np.full((n, n), np.inf)
    for s0 in range(n):
        dist = {s0: 0.0}
        pq = [(0.0, s0)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist.get(u, np.inf):
                continue
            for v, c in adj[u]:
                nd = dd + c
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        for kk, v in dist.items():
            SP[s0, kk] = v
    deg = np.bincount(E.ravel(), minlength=n)
    dedge = np.minimum.reduce([P[:, 0], 1 - P[:, 0], P[:, 1], 1 - P[:, 1]])
    return P, R, D, SP, deg, dedge


def shell_curve(R, D, i, mask=None):
    c = []
    for r in SHELLS:
        sel = np.abs(D[i] - r) <= TOL
        sel[i] = False
        if mask is not None:
            sel &= mask
        c.append(R[i, sel].mean() if sel.sum() else np.nan)
    return c


def mean_se(curves):
    X = np.array(curves)
    n = np.sum(~np.isnan(X), 0)
    return np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), n


def analyse_2d(nets):
    by_combo = defaultdict(lambda: {"boundary": [], "edge_j": [], "interior_j": [], "pairs": []})
    interior_anchor = []
    edge_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    counts = np.zeros(len(edge_bins) - 1)
    degsum = np.zeros(len(edge_bins) - 1)
    deg_edge, deg_int = [], []
    for net in nets:
        key = (net["N"], net["a"])
        P, R, D, SP, deg, dedge = solve_2d(net)
        n = len(P)
        for i in net["source"]:
            by_combo[key]["boundary"].append(shell_curve(R, D, i))
            by_combo[key]["edge_j"].append(shell_curve(R, D, i, dedge < 0.1))
            by_combo[key]["interior_j"].append(shell_curve(R, D, i, dedge >= 0.1))
        for i in np.where(np.linalg.norm(P - 0.5, axis=1) < 0.1)[0]:
            interior_anchor.append(shell_curve(R, D, i))
        iu = np.triu_indices(n, 1)
        by_combo[key]["pairs"].append((D[iu], R[iu], SP[iu]))
        idx = np.digitize(dedge, edge_bins) - 1
        for k in range(len(edge_bins) - 1):
            counts[k] += (idx == k).sum()
            degsum[k] += deg[idx == k].sum()
        deg_edge.append(deg[dedge < 0.1].mean())
        deg_int.append(deg[dedge >= 0.1].mean())
    area = np.array([(1 - 2 * edge_bins[k]) ** 2 - (1 - 2 * edge_bins[k + 1]) ** 2
                     for k in range(len(edge_bins) - 1)])
    density = dict(bins=edge_bins, per_area=counts / area / len(nets), mean_degree=degsum / counts,
                   deg_edge=np.mean(deg_edge), deg_int=np.mean(deg_int))
    return by_combo, interior_anchor, density


def report_2d(by_combo, interior_anchor, density):
    combos = sorted(by_combo, key=lambda k: k[0])
    print("\n=== Two-dimensional supplementary networks ===")
    print("combinations and Na^2:", [(c, round(c[0] * c[1] ** 2, 1)) for c in combos])
    pooled = [c for k in combos for c in by_combo[k]["boundary"]]
    mu, se, n = mean_se(pooled)
    print(f"boundary-anchored, pooled ({len(pooled)} curves):")
    print(f"   means      {np.round(mu).astype(int).tolist()}   SE {np.round(se).astype(int).tolist()}")
    print(f"   increments {np.round(np.diff(mu)).astype(int).tolist()}")
    print(f"   power-law exponent {np.polyfit(np.log(SHELLS), np.log(mu), 1)[0]:.2f}; "
          f"ratio R(0.9)/R(0.1) = {mu[-1] / mu[0]:.2f}")
    mu_i, se_i, n_i = mean_se(interior_anchor)
    ok = n_i > 100
    print(f"interior-anchored ({len(interior_anchor)} curves), shells with >100 anchors: "
          f"{np.round(mu_i[ok]).astype(int).tolist()}  increments {np.round(np.diff(mu_i[ok])).astype(int).tolist()}")
    for lab, fld in [("targets within 0.1 of a box edge", "edge_j"), ("targets >= 0.1 from every edge", "interior_j")]:
        m, s, nn = mean_se([c for k in combos for c in by_combo[k][fld]])
        print(f"boundary-anchored, {lab}: {np.round(m).astype(int).tolist()}  SE {np.round(s).astype(int).tolist()}"
              f"  increments {np.round(np.diff(m)).astype(int).tolist()}")
    print(f"mean degree: nodes within 0.1 of an edge {density['deg_edge']:.2f}, others {density['deg_int']:.2f}")
    print("all pairs, bins of width 0.1 from l=0.05:")
    for key in combos:
        l = np.concatenate([x[0] for x in by_combo[key]["pairs"]])
        R = np.concatenate([x[1] for x in by_combo[key]["pairs"]])
        SP = np.concatenate([x[2] for x in by_combo[key]["pairs"]])
        b = np.arange(0.05, 1.01, 0.1)
        idx = np.digitize(l, b)
        lb = np.array([l[idx == k].mean() for k in range(1, len(b))])
        Rb = np.array([R[idx == k].mean() for k in range(1, len(b))])
        pf = np.array([(SP[idx == k] / R[idx == k]).mean() for k in range(1, len(b))])
        slope = np.gradient(np.log(Rb), np.log(lb))
        print(f"   {key}: local exponent {np.round(slope, 2).tolist()}")
        print(f"   {key}: parallel factor {np.round(pf, 2).tolist()}")
    print("node density per unit area per network, by distance-to-edge band:",
          np.round(density['per_area']).astype(int).tolist())
    print("mean degree by band:", np.round(density['mean_degree'], 2).tolist())


# ---------------------------------------------------------------- figure
def make_panels(curves3d, by_combo, interior_anchor, density, outdir):
    plt.rcParams.update({'font.size': 10.5})  # full axes frame, like every other thesis figure
    combos = sorted(by_combo, key=lambda k: k[0])
    rr = np.linspace(0.02, 0.95, 100)

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for ci, (combo, (mu, se, n)) in enumerate(curves3d.items()):
        k = np.linalg.lstsq(np.c_[SHELLS], mu, rcond=None)[0][0]
        ax.errorbar(SHELLS, mu, yerr=se, color=COLS[ci], marker=MARKERS[ci], ms=4.5, lw=1.3, capsize=2,
                    label=f"$N$={combo[0]}, $a$={combo[1]}, $d$={combo[2]:.2f} ($n$={n.max()})")
        ax.plot([0, 0.95], [0, 0.95 * k], color=COLS[ci], lw=0.8, ls='--', alpha=.8)
    ref = curves3d[PREVIEW_COMBOS[0]][0][-1]
    ax.plot(rr, ref * np.sqrt(rr / 0.9), 'k:', lw=1.2, label="$\\propto\\sqrt{r}$ pinned at $r$=0.9")
    ax.plot([], [], 'k--', lw=0.8, label="$\\propto r$ through the origin (per combination)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1550); ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.set_xlabel("Distance $r$ from the fixed node $i$ to the moved node $j$")
    ax.set_ylabel("Mean $R_{\\mathrm{eff},ij}$ per shell ($\\Omega$)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "probe_distance_3d_linear.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for c, m, key in zip(COLS, MARKERS, combos):
        mu, se, n = mean_se(by_combo[key]["boundary"])
        ax.errorbar(SHELLS, mu, yerr=se, color=c, marker=m, ms=4.5, lw=1.3, capsize=2,
                    label=f"$N$={key[0]}, $a$={key[1]}, from source boundary")
    mu, se, n = mean_se(interior_anchor)
    ok = n > 100
    ax.errorbar(SHELLS[ok], mu[ok], yerr=se[ok], color='#0b0b0b', marker='x', ms=6, lw=1.3, capsize=2,
                label="all combinations, from nodes within 0.1 of the centre")
    ax.set_xlim(0, 1); ax.set_ylim(0, 320); ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.set_xlabel("Distance $r$ from the fixed node $i$ to the moved node $j$")
    ax.set_ylabel("Mean $R_{\\mathrm{eff},ij}$ per shell ($\\Omega$)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "probe_distance_2d_linear.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for c, m, key in zip(COLS, MARKERS, combos):
        l = np.concatenate([x[0] for x in by_combo[key]["pairs"]])
        R = np.concatenate([x[1] for x in by_combo[key]["pairs"]])
        SP = np.concatenate([x[2] for x in by_combo[key]["pairs"]])
        b = np.arange(0.05, 1.01, 0.1)
        idx = np.digitize(l, b)
        pf = [(SP[idx == k] / R[idx == k]).mean() for k in range(1, len(b))]
        # plotted at the bin centres 0.1 ... 0.9, the same abscissae as the shells of panel (a)
        ax.plot(SHELLS, pf, color=c, marker=m, ms=4.5, lw=1.3, label=f"$N$={key[0]}, $a$={key[1]}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 8); ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.set_xlabel("Distance $l$ between nodes $i$ and $j$ (all pairs)")
    ax.set_ylabel("Parallel factor $d_{ij}/R_{\\mathrm{eff},ij}$")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "probe_distance_parallel_factor.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    bins = density["bins"]
    mid = (bins[:-1] + bins[1:]) / 2
    ax.bar(mid, density["per_area"], width=np.diff(bins) * 0.9, color='#2a78d6', alpha=.85)
    for x, y, dg_ in zip(mid, density["per_area"], density["mean_degree"]):
        ax.text(x, y + 15, f"$\\langle k\\rangle$={dg_:.2f}", ha='center', fontsize=7.5, rotation=90, va='bottom')
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 300); ax.grid(alpha=.3, axis='y')
    ax.set_xlabel("Distance of a node to the nearest box edge (0.5 = box centre)")
    ax.set_ylabel("Nodes per unit area, per network")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "probe_distance_density_profile.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thesis", default=os.path.expanduser("~/Desktop/MSc---Thesis"),
                    help="thesis repository: docs/index.html is read, images/resultados/ is written")
    args = ap.parse_args()
    outdir = os.path.join(args.thesis, "images", "resultados")
    if not os.path.isdir(outdir):
        sys.exit(f"{outdir} does not exist; pass --thesis")

    curves3d = preview_curves()
    report_3d(curves3d)
    nets = load_supplement(args.thesis)
    by_combo, interior_anchor, density = analyse_2d(nets)
    report_2d(by_combo, interior_anchor, density)
    make_panels(curves3d, by_combo, interior_anchor, density, outdir)
    print(f"\nwrote four panels to {outdir}")
