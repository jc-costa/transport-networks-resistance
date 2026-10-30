"""
Analysis of the electrical-backbone sweep produced by run_backbone_task.py
(backbone_results/point_NN.json): the numbers and figures of the thesis
section on red bonds, cut vertices and the dead fraction of the pruned graph.

Prints, per (a, N a^2) point: pruned size, backbone size, dead fraction,
bridges, red bonds, red-bond share of R_eff, cut vertices, and the total
number of disagreements between the electrical, current and topological
criteria (which must be zero). Then fits the red-bond count at the lowest
N a^2 against the box size L = 1/a, log N_red = const + (1/nu) log L, to
compare with the two-dimensional value 1/nu = 3/4 (Coniglio 1981).

Usage:  python backbone_analysis.py [--figures DIR] [--json]
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "backbone_results")
INV_NU_2D = 0.75    # 1/nu with nu = 4/3 in two dimensions

FIELDS = ["n_nodes", "n_edges", "n_backbone_nodes", "n_backbone_edges", "dead_edge_fraction",
          "dead_node_fraction", "dead_length_fraction", "n_bridges_elec", "n_red_elec",
          "red_R_fraction", "red_length_fraction", "n_cut_elec", "bypass_mean"]
MISMATCH = ["bridge_elec_vs_topo_mismatch", "red_elec_vs_curr_mismatch",
            "red_elec_vs_topo_mismatch", "cut_elec_vs_topo_mismatch"]


def load(results_dir=None):
    pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results_dir or RESULTS_DIR, "point_*.json")))]
    return [p for p in pts if p["n"] >= 2]


def stats(p, field):
    v = np.array([m[field] for m in p["networks"]], dtype=float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def size_exponent(pts, x_target):
    """Slope of log <N_red> against log L over the box sizes at one N a^2 target."""
    ps = [p for p in pts if p["x_target"] == x_target]
    if len(ps) < 2:
        return None
    L = np.array([1.0 / p["a"] for p in ps])
    nred = [np.array([m["n_red_elec"] for m in p["networks"]], dtype=float) for p in ps]
    means = np.array([v.mean() for v in nred])
    if np.any(means <= 0):
        return None
    slope = np.polyfit(np.log(L), np.log(means), 1)[0]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        mb = np.array([rng.choice(v, len(v)).mean() for v in nred])
        if np.all(mb > 0):
            boot.append(np.polyfit(np.log(L), np.log(mb), 1)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return dict(x_target=x_target, L=L.tolist(), n_red=means.tolist(), slope=slope, ci=(lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--figures", default=None, help="directory to write the PNG figures into")
    ap.add_argument("--json", action="store_true", help="write backbone_summary.json next to the results")
    args = ap.parse_args()

    pts = load(args.results)
    if not pts:
        sys.exit("no points found")
    sizes = sorted(set(p["a"] for p in pts), reverse=True)
    n_real = sum(p["n"] for p in pts)
    mism = sum(m[k] for p in pts for m in p["networks"] for k in MISMATCH)
    print(f"{len(pts)} points, {n_real} realizations, criterion disagreements: {mism}")

    print(f"\n{'a':>5} {'L':>5} {'Na2':>5} {'n':>3} | {'pruned':>7} {'backb.':>7} {'dead%':>6} | "
          f"{'bridges':>7} {'red':>6} {'redR%':>6} {'cut':>6} | {'bypass':>6}")
    rows = []
    for a in sizes:
        for p in sorted([p for p in pts if p["a"] == a], key=lambda p: p["x"]):
            s = {f: stats(p, f) for f in FIELDS}
            rows.append(dict(a=a, L=1 / a, x=p["x"], x_target=p["x_target"], n=p["n"],
                             **{f: s[f][0] for f in FIELDS}, **{f + "_sem": s[f][1] for f in FIELDS}))
            print(f"{a:5.2f} {1/a:5.1f} {p['x']:5.1f} {p['n']:3d} | {s['n_nodes'][0]:7.0f} {s['n_backbone_nodes'][0]:7.0f} "
                  f"{100*s['dead_edge_fraction'][0]:6.1f} | {s['n_bridges_elec'][0]:7.1f} {s['n_red_elec'][0]:6.1f} "
                  f"{100*s['red_R_fraction'][0]:6.1f} {s['n_cut_elec'][0]:6.1f} | {s['bypass_mean'][0]:6.3f}")

    # where the red bonds disappear
    print("\nlowest N a^2 at which the mean red-bond count is below 1, per box size:")
    for a in sizes:
        rs = [r for r in rows if r["a"] == a and r["n_red_elec"] < 1]
        print(f"  a={a:.2f}: " + (f"N a^2 = {min(r['x'] for r in rs):.0f}" if rs else "never within the sweep"))

    # red-bond share of R_eff and dead fraction at the lowest x
    print("\nat the lowest N a^2 of each box size:")
    for a in sizes:
        r = min([r for r in rows if r["a"] == a], key=lambda r: r["x"])
        print(f"  a={a:.2f} (N a^2={r['x']:.0f}): {r['n_red_elec']:.1f} +- {r['n_red_elec_sem']:.1f} red bonds carrying "
              f"{100*r['red_R_fraction']:.0f}% of R_eff in {100*r['red_length_fraction']:.1f}% of the wire length; "
              f"{100*r['dead_edge_fraction']:.0f}% of the pruned edges carry no current; "
              f"{r['n_cut_elec']:.1f} cut vertices")

    exps = [e for e in (size_exponent(pts, xt) for xt in sorted(set(p["x_target"] for p in pts))) if e]
    print(f"\nred-bond count against box size, log N_red = c + s log L (2D reference 1/nu = {INV_NU_2D}):")
    for e in exps:
        print(f"  N a^2 = {e['x_target']:4.1f}: L = {', '.join(f'{L:.1f}' for L in e['L'])};  "
              f"N_red = {', '.join(f'{v:.1f}' for v in e['n_red'])};  s = {e['slope']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]")

    if args.json:
        out = os.path.join(args.results, "backbone_summary.json")
        json.dump(dict(rows=rows, size_exponents=exps, mismatches=mism, n_realizations=n_real),
                  open(out, "w"), indent=2, default=float)
        print(f"wrote {out}")

    if args.figures:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(args.figures, exist_ok=True)
        labels = {a: f"a = {a:.2f}  (L = {1/a:.1f})" for a in sizes}

        def series(field, ylabel, name, logy=False, scale=1.0):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for a in sizes:
                rs = sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"])
                ax.errorbar([r["x"] for r in rs], [scale * r[field] for r in rs],
                            yerr=[scale * r[field + "_sem"] for r in rs], fmt="o-", capsize=3, label=labels[a])
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(r"$N a^2$"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3, which="both")
            plt.tight_layout(); path = os.path.join(args.figures, name)
            fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")

        series("n_red_elec", "red bonds per realization", "backbone_red_bonds.png")
        series("red_R_fraction", r"share of $R_{\mathrm{eff}}$ in red bonds (%)", "backbone_red_fraction.png", scale=100)
        series("dead_edge_fraction", "pruned edges carrying no current (%)", "backbone_dead_fraction.png", scale=100)
        series("n_cut_elec", "cut vertices per realization", "backbone_cut_vertices.png")

        # bypass-ratio distribution at three densities, largest box
        a0 = sizes[0]
        ps = sorted([p for p in pts if p["a"] == a0], key=lambda p: p["x"])
        pick = [ps[0], ps[len(ps) // 2], ps[-1]] if len(ps) >= 3 else ps
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for p in pick:
            h = np.sum([m["bypass_hist"] for m in p["networks"]], axis=0).astype(float)
            edges = np.linspace(0, 1, p["hist_bins"] + 1); centers = (edges[:-1] + edges[1:]) / 2
            ax.step(centers, h / h.sum() / (edges[1] - edges[0]), where="mid", label=f"$N a^2$ = {p['x']:.0f}")
        ax.set_yscale("log"); ax.set_xlabel(r"bypass ratio $R_{\mathrm{eff},uv} / r_{uv}$"); ax.set_ylabel("density")
        ax.set_title(f"a = {a0:.2f}"); ax.legend(); ax.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        path = os.path.join(args.figures, "backbone_bypass_hist.png")
        fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")

        # red-bond positions along the transport axis, lowest densities
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for a in sizes:
            p = min([p for p in pts if p["a"] == a], key=lambda p: p["x"])
            xs = np.concatenate([m["red_mid_x"] for m in p["networks"]])
            if xs.size:
                ax.hist(xs, bins=np.linspace(0, 1, 21), histtype="step", density=True,
                        label=f"{labels[a]}, $N a^2$ = {p['x']:.0f}, {xs.size} red bonds")
        ax.set_xlabel("position of the red bond along the transport axis"); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); plt.tight_layout()
        path = os.path.join(args.figures, "backbone_red_positions.png")
        fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")


if __name__ == "__main__":
    main()
