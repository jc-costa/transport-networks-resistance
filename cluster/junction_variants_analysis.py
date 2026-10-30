"""
Analysis of run_junction_variants_task.py (junction_variants_results/):
distributed (lognormal) and bimodal (metallic / semiconducting) junction
resistances on the near-threshold sweep.

Prints the ratio R_eff(distributed) / R_eff(uniform, same mean) per
(N a^2, rho_bar, sigma) against the series limit (1) and the parallel limit
exp(-sigma^2) of a lognormal, and the geometric-mean value exp(-sigma^2 / 2);
and R_eff(kappa) / R_eff(1) with the share of the power dissipated in the
mixed junctions, against the densities at which the semiconducting
(2/3 N a^2 = 5.64) and metallic (1/3 N a^2 = 5.64) sub-networks percolate.

Usage:  python junction_variants_analysis.py [--figures DIR] [--json]
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "junction_variants_results")
X_C = 5.64   # infinite-system threshold of N a^2


def load(results_dir=None):
    pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results_dir or RESULTS_DIR, "point_*.json")))]
    return [p for p in pts if p["n"] >= 2]


def sem(v):
    v = np.asarray(v, dtype=float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def per_point(p):
    row = dict(a=p["a"], x=p["x"], x_target=p["x_target"], n=p["n"], dist={}, bimodal={})
    for rb in p["rho_bar"]:
        for sg in p["sigmas"]:
            v = [r["ratio_mean"] for m in p["networks"] for r in m["distributed"] if r["rho_bar"] == rb and r["sigma"] == sg]
            row["dist"][(rb, sg)] = (float(np.mean(v)), sem(v))
    for k in p["kappas"]:
        v = [b["ratio"] for m in p["networks"] for b in m["bimodal"] if b["kappa"] == k]
        d = [b["mixed_dissipation"] for m in p["networks"] for b in m["bimodal"] if b["kappa"] == k]
        row["bimodal"][k] = (float(np.mean(v)), sem(v), float(np.mean(d)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    pts = load(args.results)
    if not pts:
        sys.exit("no points found")
    p0 = pts[0]; RB, SG, KP, PM = p0["rho_bar"], p0["sigmas"], p0["kappas"], p0["p_metal"]
    rows = [per_point(p) for p in pts]
    sizes = sorted(set(r["a"] for r in rows), reverse=True)
    print(f"{len(pts)} points, {sum(p['n'] for p in pts)} realizations; rho_bar {RB}, sigma {SG}, kappa {KP}, p_metal {PM:.3f}")

    print("\nDistributed R_j: R_eff(distributed) / R_eff(uniform at the same mean)")
    print("lognormal limits: series 1, geometric mean exp(-s^2/2) = " + ", ".join(f"{np.exp(-s*s/2):.3f}" for s in SG)
          + ", parallel exp(-s^2) = " + ", ".join(f"{np.exp(-s*s):.3f}" for s in SG))
    print(f"{'a':>5} {'Na2':>4} | " + "  ".join(f"rb={rb:<4} s={sg:<3}" for rb in RB for sg in SG))
    for a in sizes:
        for r in sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"]):
            print(f"{a:5.2f} {r['x_target']:4d} | " + "  ".join(f"{r['dist'][(rb, sg)][0]:12.3f}" for rb in RB for sg in SG))
    dense = [r for r in rows if r["x_target"] >= 12]
    print("\n dense networks (N a^2 >= 12), rho_bar = 10: mean ratio vs exp(-s^2/2)")
    for sg in SG:
        v = [r["dist"][(RB[-1], sg)][0] for r in dense]
        print(f"   sigma={sg}: ratio {np.mean(v):.3f} (range {min(v):.3f}-{max(v):.3f})   exp(-s^2/2) = {np.exp(-sg*sg/2):.3f}   exp(-s^2) = {np.exp(-sg*sg):.3f}")

    print(f"\nBimodal R_j: R_eff(kappa) / R_eff(1) and mixed-junction dissipation share; semiconducting sub-network percolates at N a^2 = {X_C/(1-PM):.1f}, metallic at {X_C/PM:.1f}")
    print(f"{'a':>5} {'Na2':>4} | " + "  ".join(f"k={k:<6.0f}" for k in KP) + " | diss " + " ".join(f"k={k:<5.0f}" for k in KP))
    for a in sizes:
        for r in sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"]):
            print(f"{a:5.2f} {r['x_target']:4d} | " + "  ".join(f"{r['bimodal'][k][0]:8.2f}" for k in KP)
                  + " |      " + " ".join(f"{100*r['bimodal'][k][2]:6.0f}%" for k in KP))

    if args.json:
        out = os.path.join(args.results, "junction_variants_summary.json")
        json.dump([dict(a=r["a"], x=r["x"], x_target=r["x_target"], n=r["n"],
                        dist={f"{rb},{sg}": v for (rb, sg), v in r["dist"].items()},
                        bimodal={str(k): v for k, v in r["bimodal"].items()}) for r in rows], open(out, "w"), indent=1)
        print(f"wrote {out}")

    if args.figures:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(args.figures, exist_ok=True)
        markers = {0.20: "o", 0.15: "s", 0.10: "^"}

        # (a) dispersion: ratio against N a^2 for each sigma at the largest rho_bar, with the lognormal references
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, sg in enumerate(SG):
            for a in sizes:
                rs = sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"])
                ax.errorbar([r["x"] for r in rs], [r["dist"][(RB[-1], sg)][0] for r in rs], yerr=[r["dist"][(RB[-1], sg)][1] for r in rs],
                            fmt=markers[a] + "-", ms=4, capsize=2, color=colors[i], label=f"$\\sigma$ = {sg}" if a == sizes[0] else None)
            ax.axhline(np.exp(-sg * sg / 2), color=colors[i], ls="--", lw=0.8)
            ax.axhline(np.exp(-sg * sg), color=colors[i], ls=":", lw=0.8)
        ax.set_xlabel(r"$Na^2$"); ax.set_ylabel(r"$R_{\mathrm{eff}}$(lognormal) / $R_{\mathrm{eff}}$(uniform, same mean)")
        ax.set_title(rf"$\bar\rho$ = {RB[-1]:g}"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); path = os.path.join(args.figures, "junction_dispersion.png")
        fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")

        # (b) bimodal: R(kappa)/R(1) against N a^2 with the sub-network thresholds
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, k in enumerate(KP):
            for a in sizes:
                rs = sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"])
                ax.errorbar([r["x"] for r in rs], [r["bimodal"][k][0] for r in rs], yerr=[r["bimodal"][k][1] for r in rs],
                            fmt=markers[a] + "-", ms=4, capsize=2, color=colors[i], label=f"$\\kappa$ = {k:g}" if a == sizes[0] else None)
        ax.axvline(X_C / (1 - PM), color="gray", ls="--", lw=0.8); ax.axvline(X_C / PM, color="gray", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel(r"$Na^2$"); ax.set_ylabel(r"$R_{\mathrm{eff}}(\kappa) / R_{\mathrm{eff}}(1)$")
        ax.set_title(rf"$p_m$ = 1/3, $\rho$ = {p0['rho_c']:g}")
        ax.legend(); ax.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        path = os.path.join(args.figures, "junction_bimodal.png"); fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")

        # (c) mixed-junction dissipation share
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, k in enumerate(KP):
            for a in sizes:
                rs = sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"])
                ax.plot([r["x"] for r in rs], [100 * r["bimodal"][k][2] for r in rs], markers[a] + "-", ms=4, color=colors[i], label=f"$\\kappa$ = {k:g}" if a == sizes[0] else None)
        ax.axvline(X_C / (1 - PM), color="gray", ls="--", lw=0.8)
        ax.set_xlabel(r"$Na^2$"); ax.set_ylabel("power dissipated in mixed junctions (%)"); ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        path = os.path.join(args.figures, "junction_mixed_dissipation.png"); fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")


if __name__ == "__main__":
    main()
