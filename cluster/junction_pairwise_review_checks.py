"""
EXPLORATORY checks prompted by the pre-submission review (TNANO-review.md). Not pre-registered;
every number here is post hoc and is labeled so wherever it is used.

What it measures, on the dense-grid records (cluster/junction_pairwise_dense_results):
  1. Cohort flow: generated networks, percolation yield, networks with a complete all-pairs profile
     (P) and with every profile complete (C1, C4, C16, P), by stick length a.
  2. Baselines, with whole cells held out (5 folds, the registered seeding): R alone, R with the
     mean level of the nine log P values, R with the centered shape of the profile, R with P.
     On the common cohort (every profile complete) and on the broader P cohort.
  3. Error statistics for R+P in factor terms: 10^RMSE, the median and the 90th percentile of the
     factor error.
  4. Transfer to a held-out junction level: whole cells AND one rho level out of training.
  5. What the variable-junction rules mean in terms of a base value (arithmetic, geometric).
  6. A film whose wire resistance per length R0 is not known: one common scale factor multiplies R
     and every shell value; the estimator trained with exact R0 is tested with the error, and a
     scale-free estimator (shell-to-terminal ratios only) is scored.

Usage: python junction_pairwise_review_checks.py <partial.jsonl> [--out summary.json]
"""
import sys, json
import numpy as np
import junction_pairwise_regime_analysis as rg

SEED, FOLDS = 0, 5


def load_all(path):
    """One row per (cell, network, uniform condition), with raw (possibly missing) profiles."""
    rows, gen = [], {}
    for line in open(path):
        r = json.loads(line)
        gen[r["id"]] = dict(a=r["a"], generated=r["n"], attempts=r["attempts_total"])
        for k, net in enumerate(r["networks"]):
            for b in net["by_cond"]:
                if b["variant"] != "uniform":
                    continue
                rows.append(dict(cell=r["id"], net=k, a=r["a"], x=r["x"], rho=b["rho"], S=b["S"],
                                 C1=b["C1"], C4=b["C4"], C16=b["C16"], P=b["P"]))
    return rows, gen


def complete(v):
    return all(x is not None and np.isfinite(x) and x > 0 for x in v)


def folds_for(cells):
    cells = sorted(set(cells))
    order = np.random.default_rng(SEED).permutation(len(cells))
    return {c: int(order[i] % FOLDS) for i, c in enumerate(cells)}


def fit_predict(Xtr, ytr, Xte, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    A, B = rg.design_like((Xtr - mu) / sd), rg.design_like((Xte - mu) / sd)
    pen = lam * np.eye(A.shape[1]); pen[0, 0] = 0
    w = np.linalg.solve(A.T @ A + len(A) * pen, A.T @ ytr)
    return B @ w


def r2(y, p):
    return 1 - np.mean((p - y) ** 2) / np.var(y)


def features(rows, name):
    S = np.log10([r["S"] for r in rows])[:, None]
    P = np.log10(np.array([r["P"] for r in rows]))
    if name == "S":
        return S
    if name == "S+level":
        return np.hstack([S, P.mean(1, keepdims=True)])
    if name == "S+shape":
        return np.hstack([S, P - P.mean(1, keepdims=True)])
    if name == "S+P":
        return np.hstack([S, P])
    key = name.split("+")[1]
    return np.hstack([S, np.log10(np.array([r[key] for r in rows]))])


def cv(rows, name, y, fold):
    X = features(rows, name)
    p = np.zeros(len(y))
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        p[te] = fit_predict(X[tr], y[tr], X[te])
    return p


def main():
    path = sys.argv[1]
    rows, gen = load_all(path)
    out = {}

    # ---- 1. cohort flow, by a
    nets = {}
    for r in rows:
        nets.setdefault((r["cell"], r["net"]), r)
    flow = {}
    for (cell, k), r in nets.items():
        f = flow.setdefault(r["a"], dict(generated=0, P=0, common=0))
        f["generated"] += 1
        if complete(r["P"]):
            f["P"] += 1
            if complete(r["C1"]) and complete(r["C4"]) and complete(r["C16"]):
                f["common"] += 1
    att = {}
    for g in gen.values():
        d = att.setdefault(g["a"], [0, 0]); d[0] += g["generated"]; d[1] += g["attempts"]
    print("Cohort flow by a (percolating networks generated -> complete P -> every profile complete)")
    print("   a    attempts  percolating  yield   with P   common")
    for a in sorted(flow):
        f = flow[a]
        print(f"  {a:4.2f} {att[a][1]:9d} {f['generated']:11d} {att[a][0]/att[a][1]:6.1%} {f['P']:8d} {f['common']:8d}")
    tot = {k: sum(f[k] for f in flow.values()) for k in ("generated", "P", "common")}
    print(f"  all  {sum(v[1] for v in att.values()):9d} {tot['generated']:11d} {sum(v[0] for v in att.values())/sum(v[1] for v in att.values()):6.1%} {tot['P']:8d} {tot['common']:8d}")
    out["flow"] = {str(a): dict(flow[a], attempts=att[a][1]) for a in sorted(flow)}

    # ---- 2. baselines, cells held out
    pos = [r for r in rows if r["rho"] > 0]
    cohorts = {"common (every profile complete)": [r for r in pos if complete(r["P"]) and complete(r["C1"]) and complete(r["C4"]) and complete(r["C16"])],
               "broader (P complete)": [r for r in pos if complete(r["P"])]}
    sets = {"common (every profile complete)": ["S", "S+level", "S+shape", "S+C1", "S+C4", "S+C16", "S+P"],
            "broader (P complete)": ["S", "S+level", "S+shape", "S+P"]}
    out["r2"] = {}
    for cname, cohort in cohorts.items():
        y = np.log10([r["rho"] for r in cohort])
        fold = np.array([folds_for([r["cell"] for r in cohort])[r["cell"]] for r in cohort])
        nn = len({(r["cell"], r["net"]) for r in cohort})
        print(f"\nR^2 of log10 rho, whole cells held out, {cname}: {nn} networks, {len({r['cell'] for r in cohort})} cells")
        out["r2"][cname] = {}
        for s in sets[cname]:
            p = cv(cohort, s, y, fold)
            out["r2"][cname][s] = float(r2(y, p))
            print(f"   {s:9s} {r2(y, p):.3f}   RMSE {np.sqrt(np.mean((p - y) ** 2)):.3f}")
        if cname.startswith("common"):
            p = cv(cohort, "S+P", y, fold); res = np.abs(p - y)
            fac = 10 ** res
            out["factor_error"] = dict(rmse_log10=float(np.sqrt(np.mean((p - y) ** 2))),
                                       ten_to_rmse=float(10 ** np.sqrt(np.mean((p - y) ** 2))),
                                       median=float(np.median(fac)), p90=float(np.percentile(fac, 90)))
            print(f"   S+P in factor terms: 10^RMSE {out['factor_error']['ten_to_rmse']:.2f}, "
                  f"median factor error {out['factor_error']['median']:.2f}, 90th percentile {out['factor_error']['p90']:.2f}")

    # ---- 4. whole rho level held out, on the common cohort
    cohort = cohorts["common (every profile complete)"]
    y = np.log10([r["rho"] for r in cohort]); rho = np.array([r["rho"] for r in cohort])
    fmap = folds_for([r["cell"] for r in cohort]); fold = np.array([fmap[r["cell"]] for r in cohort])
    print("\nWhole cells and one rho level held out (R+P, common cohort)")
    print("    rho   RMSE   median error (log10)")
    X = features(cohort, "S+P"); allp = np.zeros(len(y)); out["heldout_rho"] = {}
    for lv in sorted(set(rho)):
        for f in range(FOLDS):
            te = (fold == f) & (rho == lv); tr = (fold != f) & (rho != lv)
            allp[te] = fit_predict(X[tr], y[tr], X[te])
        m = rho == lv
        e = allp[m] - y[m]
        out["heldout_rho"][str(lv)] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), median_error=float(np.median(e)))
        print(f"  {lv:6.2f} {np.sqrt(np.mean(e ** 2)):6.3f} {np.median(e):+8.3f}")
    print(f"  all levels, R^2 {r2(y, allp):.3f}, RMSE {np.sqrt(np.mean((allp - y) ** 2)):.3f} (against {out['r2']['common (every profile complete)']['S+P']:.3f} with the level in training)")
    out["heldout_rho_all"] = dict(r2=float(r2(y, allp)), rmse=float(np.sqrt(np.mean((allp - y) ** 2))))

    # ---- 5. what the variable-junction rules mean
    p_m, kappa, sig = 1 / 3, 100.0, 1.5
    mixed = 2 * p_m * (1 - p_m)
    print("\nVariable-junction rules, in units of the base value rho (R_j = rho R0 a):")
    print(f"   bimodal: mixed fraction {mixed:.3f}; arithmetic mean {(1 - mixed) + mixed * kappa:.1f}x; geometric mean {kappa ** mixed:.2f}x")
    print(f"   log-normal (sigma {sig}): arithmetic mean 1.00x (by construction); geometric mean/median {np.exp(-sig ** 2 / 2):.3f}x")
    out["rules"] = dict(bimodal_arith=float((1 - mixed) + mixed * kappa), bimodal_geo=float(kappa ** mixed),
                        lognormal_geo=float(np.exp(-sig ** 2 / 2)), mixed_fraction=float(mixed))
    # ---- 6. unknown R0
    cohort = cohorts["common (every profile complete)"]
    y = np.log10([r["rho"] for r in cohort]); fmap = folds_for([r["cell"] for r in cohort])
    fold = np.array([fmap[r["cell"]] for r in cohort])
    S = np.log10([r["S"] for r in cohort])[:, None]; P = np.log10(np.array([r["P"] for r in cohort]))
    Xf, Xr = np.hstack([S, P]), P - S
    rng = np.random.default_rng(SEED)

    def score(Xtr_all, Xte_all):
        p = np.zeros(len(y))
        for f in range(FOLDS):
            tr, te = fold != f, fold == f
            p[te] = fit_predict(Xtr_all[tr], y[tr], Xte_all[te])
        return float(r2(y, p)), float(np.sqrt(np.mean((p - y) ** 2)))

    print("\nUnknown R0: one common scale factor per film on R and on every shell (error sigma, natural log)")
    print("   sigma | trained with exact R0, tested with the error | scale-free ratios only")
    out["unknown_R0"] = {}
    for sg in (0.0, 0.10, 0.20, 0.30, 0.50):
        z = np.log10(np.exp(sg * rng.standard_normal(len(y))))[:, None]
        a, b = score(Xf, Xf + z), score(Xr, Xr)
        out["unknown_R0"][str(sg)] = dict(with_R0_error=a, ratios_only=b)
        print(f"   {sg:5.2f} | R2 {a[0]:.3f}  RMSE {a[1]:.3f} | R2 {b[0]:.3f}  RMSE {b[1]:.3f}")
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
