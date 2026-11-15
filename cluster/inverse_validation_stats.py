"""
Statistics quoted in the paper for the inverse validation (Section III-B,
Supplementary S3) and for the red bonds at N a^2 = 4 (Section III-D), computed
from the stored per-network results.

  * Inverse validation, from validation_plan.json and validation_results/
    (thesis run), optionally with candidate 19 replaced by
    figures/data/inverse_candidate19_extended.json.
  * Red bonds at N a^2 = 4 in films 5 and 10 stick lengths across, from
    large_film_results/ (the large-film sweep), with bootstrap intervals.

Usage (PIPELINE_DIR = the ColabProject folder):
  PIPELINE_DIR=... python inverse_validation_stats.py [--extended] [--json OUT]
"""
import os
import sys
import glob
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.environ.get("PIPELINE_DIR", os.path.normpath(os.path.join(HERE, "..")))
EXT = os.path.join(HERE, "..", "figures", "data", "inverse_candidate19_extended.json")
N_BOOT, SEED = 2000, 20261021


def inverse_stats(extended):
    plan = json.load(open(os.path.join(PIPE, "cluster", "validation_plan.json")))
    fit = plan["fit"]
    A, p, q = fit["Ap"], fit["p_exp"], fit["q_exp"]
    cands = []
    for i in range(len(plan["plan"])):
        path = os.path.join(PIPE, "cluster", "validation_results", f"candidate_{i:02d}.json")
        if not os.path.exists(path):
            continue
        c = json.load(open(path))
        if extended and i == 19:
            c = json.load(open(EXT))
        if c["n"] > 0:
            cands.append(c)

    rows, factors, nets = [], [], 0
    for c in cands:
        R = np.asarray(c["R_eff_values"], float)
        Rt, N, a = c["R_target"], c["N"], c["a"]
        err = (R.mean() - Rt) / Rt
        gerr = (np.exp(np.log(R).mean()) - Rt) / Rt
        miss = np.maximum(R / Rt, Rt / R)
        N_est = (R / (A * a ** q)) ** (1.0 / p)          # invert R = A N^p a^q at the true a
        f = np.maximum(N_est / N, N / N_est)
        factors.extend(f); nets += len(R)
        rows.append(dict(R_target=Rt, N=N, a=a, n=len(R), achieved=R.mean(), err=err, gerr=gerr,
                         miss_median=float(np.median(miss))))

    abs_err = np.abs([r["err"] for r in rows])
    by_a = {a: float(np.mean([abs(r["err"]) for r in rows if r["a"] == a])) for a in sorted({r["a"] for r in rows})}
    factors = np.asarray(factors)
    misses = [r["miss_median"] for r in rows]
    return dict(
        n_candidates=len(rows), n_networks=nets,
        median_abs_error=float(np.median(abs_err)),
        within_30=int(np.sum(abs_err <= 0.30)),
        median_abs_error_without_below_Na2_10=float(np.median([abs(r["err"]) for r in rows if r["N"] * r["a"] ** 2 >= 10])),
        mean_abs_error_by_a=by_a,
        median_abs_error_geometric=float(np.median(np.abs([r["gerr"] for r in rows]))),
        per_network_miss_median=float(np.median(misses)), per_network_miss_range=[float(min(misses)), float(max(misses))],
        N_factor_median=float(np.median(factors)), N_factor_p90=float(np.percentile(factors, 90)),
        N_factor_within_1p2=float(np.mean(factors <= 1.2)),
        candidates=rows)


def red_bonds_at_4(rng):
    out = {}
    for f in sorted(glob.glob(os.path.join(PIPE, "cluster", "large_film_results", "point_*.json"))):
        d = json.load(open(f))
        if d["x_target"] != 4 or d["Lambda"] not in (5.0, 10.0):
            continue
        share = np.array([m["red_R_fraction"] for m in d["networks"]])
        nred = np.array([m["n_red"] for m in d["networks"]], float)
        bs = rng.integers(0, len(share), size=(N_BOOT, len(share)))
        out[f"L={d['Lambda']:g}l"] = dict(
            n=int(d["n"]), draws=int(d["draws"]), perc_rate=float(d["perc_rate"]),
            share_mean=float(share.mean()), share_ci95=[float(x) for x in np.percentile(share[bs].mean(1), [2.5, 97.5])],
            n_red_mean=float(nred.mean()), n_red_ci95=[float(x) for x in np.percentile(nred[bs].mean(1), [2.5, 97.5])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true", help="use the 30-network candidate 19")
    ap.add_argument("--json")
    args = ap.parse_args()
    res = dict(inverse=inverse_stats(args.extended), red_bonds_Na2_4=red_bonds_at_4(np.random.default_rng(SEED)))
    inv = res["inverse"]
    print(f"inverse ({'candidate 19 extended' if args.extended else 'thesis run'}): {inv['n_candidates']} candidates, {inv['n_networks']} networks")
    for k in ("median_abs_error", "within_30", "median_abs_error_without_below_Na2_10", "mean_abs_error_by_a",
              "median_abs_error_geometric", "per_network_miss_median", "per_network_miss_range",
              "N_factor_median", "N_factor_p90", "N_factor_within_1p2"):
        print(f"  {k}: {inv[k]}")
    c19 = [r for r in inv["candidates"] if r["N"] == 80 and r["a"] == 0.2][0]
    print(f"  candidate (80, 0.2): n = {c19['n']}, achieved {c19['achieved']:.2f} ohm, error {c19['err']:+.1%}")
    for k, v in res["red_bonds_Na2_4"].items():
        print(f"red bonds at Na^2 = 4, {k}: {v}")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
