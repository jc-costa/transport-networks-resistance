"""
Analysis of the junction-resistance sweep of run_junction_task.py
(junction_results/point_NN.json): the numbers and figures of the thesis
section on the crossover from the wire-limited to the junction-limited regime.

Per (a, N a^2) point it reports
  * R_eff(rho) / R_eff(0), the crossover curve, and rho* where the junction
    contribution equals the wire contribution;
  * beta = dR_eff / dR_j, the slope of R_eff in the junction resistance (Benda
    et al. 2019 write R_eff as additive in the wire and junction resistances;
    on a fixed network the slope is the effective number of junctions in
    series), with its drift across rho as the measure of non-linearity;
  * the red segments and red junctions, and the share of R_eff they carry as
    a function of rho;
  * the exponent of R_eff in N a^2 over the upper part of the sweep at rho = 0
    and at the largest rho, the wire-limited and junction-limited exponents.

Usage:  python junction_analysis.py [--figures DIR] [--json]
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "junction_results")
X_DENSE = (8, 20)   # N a^2 window for the exponent fits (above the red-bond regime)


def load(results_dir=None):
    pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results_dir or RESULTS_DIR, "point_*.json")))]
    return [p for p in pts if p["n"] >= 2]


def sem(v):
    v = np.asarray(v, dtype=float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def per_point(p):
    rhos = np.array(p["rhos"]); a = p["a"]; R0a = p["R0"] * a
    R = np.array([[r["R_eff"] for r in m["by_rho"]] for m in p["networks"]])      # (n, n_rho)
    ratio = R / R[:, [0]]
    # slope in R_j between consecutive rho > 0 (per network), then mean
    Rj = rhos * R0a
    beta = (R[:, 2:] - R[:, 1:-1]) / (Rj[2:] - Rj[1:-1])                         # between rho_k and rho_{k+1}, k>=1
    beta_mid = np.sqrt(rhos[2:] * rhos[1:-1])
    # rho* per network: R_eff(rho*) = 2 R_eff(0), by log interpolation
    rho_star = []
    for row in ratio:
        idx = np.searchsorted(row[1:], 2.0) + 1
        if 1 <= idx < len(rhos):
            x0, x1 = np.log(rhos[idx - 1]), np.log(rhos[idx]); y0, y1 = row[idx - 1], row[idx]
            rho_star.append(np.exp(x0 + (2.0 - y0) / (y1 - y0) * (x1 - x0)))
    red_seg = np.array([m["n_red_seg"] for m in p["networks"]], dtype=float)
    red_jun = np.array([m["n_red_jun"] for m in p["networks"]], dtype=float)
    seg_share = np.array([[r["red_seg_R"] / r["R_eff"] for r in m["by_rho"]] for m in p["networks"]])
    jun_share = np.array([[r["red_jun_R"] / r["R_eff"] for r in m["by_rho"]] for m in p["networks"]])
    jun_diss = np.array([[r["jun_R_total_dissipated"] / r["R_eff"] for r in m["by_rho"]] for m in p["networks"]])
    return dict(
        a=a, x=p["x"], x_target=p["x_target"], n=p["n"], rhos=rhos.tolist(),
        R_mean=R.mean(0).tolist(), R_sem=[sem(R[:, k]) for k in range(len(rhos))],
        ratio_mean=ratio.mean(0).tolist(), ratio_sem=[sem(ratio[:, k]) for k in range(len(rhos))],
        beta_mid=beta_mid.tolist(), beta_mean=beta.mean(0).tolist(), beta_sem=[sem(beta[:, k]) for k in range(beta.shape[1])],
        rho_star=float(np.mean(rho_star)) if rho_star else None, rho_star_sem=sem(rho_star) if len(rho_star) > 1 else None,
        red_seg=red_seg.mean(), red_seg_sem=sem(red_seg), red_jun=red_jun.mean(), red_jun_sem=sem(red_jun),
        seg_share=seg_share.mean(0).tolist(), jun_share=jun_share.mean(0).tolist(), jun_diss=jun_diss.mean(0).tolist(),
    )


def exponent(rows, a, k):
    """Slope of log R_eff(rho_k) against log N a^2 over X_DENSE, per box, with a bootstrap over points."""
    rs = [r for r in rows if r["a"] == a and X_DENSE[0] <= r["x_target"] <= X_DENSE[1]]
    x = np.log([r["x"] for r in rs]); y = np.log([r["R_mean"][k] for r in rs])
    return np.polyfit(x, y, 1)[0], len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    pts = load(args.results)
    if not pts:
        sys.exit("no points found")
    rows = [per_point(p) for p in pts]
    rhos = rows[0]["rhos"]; sizes = sorted(set(r["a"] for r in rows), reverse=True)
    print(f"{len(pts)} points, {sum(p['n'] for p in pts)} realizations, rho in {rhos}")

    print(f"\n{'a':>5} {'Na2':>4} | {'R(0)':>7} {'R(1)/R(0)':>9} {'R(100)/R(0)':>11} {'rho*':>6} | {'beta@0.03':>9} {'beta@30':>8} | {'red seg':>7} {'red jun':>7} {'jun share@1':>11} {'@100':>6}")
    for a in sizes:
        for r in sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"]):
            k1 = rhos.index(1.0); k100 = len(rhos) - 1
            b_lo = r["beta_mean"][0]; b_hi = r["beta_mean"][-1]
            print(f"{a:5.2f} {r['x_target']:4d} | {r['R_mean'][0]:7.0f} {r['ratio_mean'][k1]:9.2f} {r['ratio_mean'][k100]:11.0f} "
                  f"{(r['rho_star'] or float('nan')):6.2f} | {b_lo:9.2f} {b_hi:8.2f} | {r['red_seg']:7.1f} {r['red_jun']:7.1f} "
                  f"{100*r['jun_share'][k1]:10.0f}% {100*r['jun_share'][k100]:5.0f}%")

    print("\nnon-linearity: beta at the lowest and highest rho interval, and their ratio (1 = exactly additive)")
    for a in sizes:
        for r in sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"]):
            if r["x_target"] in (4, 6, 8, 12, 20):
                print(f"  a={a:.2f} Na2={r['x_target']:2d}: beta {r['beta_mean'][0]:.3f} -> {r['beta_mean'][-1]:.3f}, ratio {r['beta_mean'][-1]/r['beta_mean'][0]:.2f}")

    print(f"\nexponent of R_eff in N a^2 over {X_DENSE[0]} <= N a^2 <= {X_DENSE[1]}:")
    exps = {}
    for a in sizes:
        e0, n0 = exponent(rows, a, 0); e1, _ = exponent(rows, a, len(rhos) - 1)
        exps[a] = dict(wire=e0, junction=e1, n_points=n0)
        print(f"  a={a:.2f}: wire-limited (rho=0) {e0:.2f}, junction-limited (rho={rhos[-1]:.0f}) {e1:.2f}  ({n0} points)")

    print("\nred junctions vs cut vertices of the wire-limited graph, and share of R_eff in red junctions at rho = 1 and 100:")
    for r in rows:
        if r["x_target"] == 4:
            print(f"  a={r['a']:.2f}: red segments {r['red_seg']:.1f} +- {r['red_seg_sem']:.1f}, red junctions {r['red_jun']:.1f} +- {r['red_jun_sem']:.1f}; "
                  f"segment share {100*r['seg_share'][0]:.0f}% -> {100*r['seg_share'][-1]:.0f}%, junction share {100*r['jun_share'][rhos.index(1.0)]:.0f}% (rho=1), {100*r['jun_share'][-1]:.0f}% (rho=100); "
                  f"all junctions dissipate {100*r['jun_diss'][rhos.index(1.0)]:.0f}% (rho=1), {100*r['jun_diss'][-1]:.0f}% (rho=100)")

    if args.json:
        out = os.path.join(args.results, "junction_summary.json")
        json.dump(dict(rows=rows, exponents={str(a): v for a, v in exps.items()}, x_dense=X_DENSE), open(out, "w"), indent=1, default=float)
        print(f"wrote {out}")

    if args.figures:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(args.figures, exist_ok=True)
        rho_pos = np.array(rhos[1:])

        # (a) crossover: R(rho)/R(0) for three densities at each box
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        styles = {4: "-", 8: "--", 20: ":", 12: ":"}
        for a in sizes:
            for r in sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"]):
                xt = r["x_target"]
                if xt not in (4, 8, 20) and not (a == 0.10 and xt == 12):
                    continue
                ax.errorbar(rho_pos, r["ratio_mean"][1:], yerr=r["ratio_sem"][1:], fmt="o" + styles[xt], ms=4, capsize=2,
                            label=f"a = {a:.2f}, $Na^2$ = {xt}")
        ax.axhline(2, color="gray", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\rho = R_j / (R_0 a)$"); ax.set_ylabel(r"$R_{\mathrm{eff}}(\rho) / R_{\mathrm{eff}}(0)$")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        path = os.path.join(args.figures, "junction_crossover.png"); fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")

        # (b) beta = dR/dR_j against N a^2 at the lowest and highest rho interval
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for a in sizes:
            rs = sorted([r for r in rows if r["a"] == a], key=lambda r: r["x"])
            ax.errorbar([r["x"] for r in rs], [r["beta_mean"][0] for r in rs], yerr=[r["beta_sem"][0] for r in rs], fmt="o-", capsize=2, label=f"a = {a:.2f}, $\\rho \\to 0$")
            ax.errorbar([r["x"] for r in rs], [r["beta_mean"][-1] for r in rs], yerr=[r["beta_sem"][-1] for r in rs], fmt="s--", capsize=2, mfc="none", label=f"a = {a:.2f}, $\\rho \\to 100$")
            ax.plot([r["x"] for r in rs], [r["red_jun"] for r in rs], "x:", color="gray",
                    label="red junctions" if a == sizes[0] else None)
        ax.set_yscale("log"); ax.set_xlabel(r"$N a^2$"); ax.set_ylabel(r"$\beta = \mathrm{d}R_{\mathrm{eff}} / \mathrm{d}R_j$")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        path = os.path.join(args.figures, "junction_beta.png"); fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")

        # (c) share of R_eff in red segments and red junctions against rho, Na^2 = 4
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for a in sizes:
            r = next(r for r in rows if r["a"] == a and r["x_target"] == 4)
            ax.plot(rho_pos, 100 * np.array(r["seg_share"][1:]), "o-", ms=4, label=f"a = {a:.2f}, red segments")
            ax.plot(rho_pos, 100 * np.array(r["jun_share"][1:]), "s--", ms=4, mfc="none", label=f"a = {a:.2f}, red junctions")
        ax.set_xscale("log"); ax.set_xlabel(r"$\rho = R_j / (R_0 a)$"); ax.set_ylabel(r"share of $R_{\mathrm{eff}}$ (%)")
        ax.set_title(r"$Na^2 = 4$"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        path = os.path.join(args.figures, "junction_red_share.png"); fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {path}")


if __name__ == "__main__":
    main()
