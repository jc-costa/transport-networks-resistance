#!/usr/bin/env python3
"""Sensitivity of the matched-group ANOVA to the grouping tolerance.

Reproduces, outside the notebook, the two robustness tables of the onion
analysis: the realization-to-realization coefficient of variation of the
scalar R_eff within each (N, a, d) combination, and the grouping-and-ANOVA
pipeline of notebooks/r_eff_ij_vs_N_a_d.ipynb (Part C) re-run at several
relative tolerances. The grouping rule and the per-layer one-way ANOVA on
self-normalized realization curves are the same as in the notebook, so the
row at the notebook's tolerance (15%) must reproduce its totals exactly.

Usage:
    python onion_tolerance_sweep.py [--onion DIR] [--scalar DIR] [--json OUT]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TOLERANCES = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]


def load_rows(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "chunk_*.json"))):
        with open(p) as f:
            rows.extend(json.load(f))
    return {(r["N"], r["a"], r["d"]): r for r in rows}


def aligned_curves(row, shared):
    out = []
    for cv in row["per_realization_curves"]:
        vals = [cv.get(str(d)) for d in shared]
        if all(v is not None for v in vals):
            out.append(vals)
    return np.array(out, dtype=float)


def build_groups(paired, scalar, tol):
    groups, used = [], set()
    for i, k1 in enumerate(paired):
        if k1 in used:
            continue
        r1 = scalar[k1]["mean_ohm"]
        group = [k1]
        for k2 in paired[i + 1:]:
            if k2 in used:
                continue
            if abs(scalar[k2]["mean_ohm"] - r1) / r1 <= tol and k2[:2] != k1[:2]:
                group.append(k2)
        if len(group) >= 2:
            groups.append(group)
            used.update(group)
    return groups


def anova_over_groups(groups, onion):
    pvals, per_group = [], []
    for gi, group in enumerate(groups):
        shared = sorted(set.intersection(*[set(float(d) for d in onion[k]["onion_layer_curves"]) for k in group]))
        if len(shared) < 2:
            continue
        curves = {}
        for k in group:
            c = aligned_curves(onion[k], shared)
            if c.shape[0] >= 3:
                curves[k] = c / c[:, -1:]
        if len(curves) < 2:
            continue
        layer_p = []
        for li in range(len(shared) - 1):
            _, p = stats.f_oneway(*[c[:, li] for c in curves.values()])
            layer_p.append(p)
        pvals.extend(layer_p)
        per_group.append({"group": gi, "n_triples": len(curves), "n_layers": len(layer_p),
                          "min_p": float(min(layer_p)), "frac_sig_p05": float(np.mean([p < 0.05 for p in layer_p]))})
    return np.array(pvals), per_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onion", default=os.path.join(HERE, "onion_grid_results"))
    ap.add_argument("--scalar", default=os.path.join(HERE, "effective_resistance_d_results"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    onion, scalar = load_rows(args.onion), load_rows(args.scalar)
    paired = [k for k in onion if k in scalar and onion[k]["onion_layer_curves"] and scalar[k]["mean_ohm"] is not None]
    paired.sort(key=lambda k: scalar[k]["mean_ohm"])
    print(f"{len(paired)} combinations with a scalar R_eff and a non-empty onion curve")

    # realization-to-realization spread of the scalar R_eff
    cvs = np.array([np.std(r["R_eff_values"], ddof=1) / np.mean(r["R_eff_values"])
                    for r in scalar.values() if len(r["R_eff_values"]) >= 2])
    n_real = sorted({len(r["R_eff_values"]) for r in scalar.values() if len(r["R_eff_values"]) >= 2})
    q25, q50, q75, q90 = np.percentile(cvs, [25, 50, 75, 90])
    print(f"coefficient of variation over {len(cvs)} combinations with >=2 realizations (n per combination: {n_real}):")
    print(f"   median {q50:.1%}   IQR {q25:.1%}-{q75:.1%}   90th pct {q90:.1%}   share above 15%: {np.mean(cvs > 0.15):.1%}")
    print(f"   implied standard error of a 24-realization mean at the median CV: {q50/np.sqrt(24):.1%}")

    table = []
    print(f"\n{'tol':>6} {'groups':>7} {'tests':>6} {'% sig':>7} {'Bonf.':>6} {'median p':>9}")
    for tol in TOLERANCES:
        groups = build_groups(paired, scalar, tol)
        pv, per_group = anova_over_groups(groups, onion)
        n = len(pv)
        row = {"tolerance": tol, "n_groups": len(per_group), "n_tests": n,
               "frac_sig_p05": float(np.mean(pv < 0.05)), "n_bonferroni": int(np.sum(pv < 0.05 / n)),
               "median_p": float(np.median(pv)), "per_group": per_group}
        table.append(row)
        print(f"{tol:>6.1%} {row['n_groups']:>7d} {n:>6d} {row['frac_sig_p05']:>7.1%} {row['n_bonferroni']:>6d} {row['median_p']:>9.3f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"n_paired": len(paired), "cv": {"n": int(len(cvs)), "median": float(q50), "q25": float(q25),
                       "q75": float(q75), "q90": float(q90), "frac_above_015": float(np.mean(cvs > 0.15))},
                       "tolerances": table}, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
