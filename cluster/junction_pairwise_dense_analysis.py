"""
Pre-registered analysis of the dense grid (PREREGISTRATION_JUNCTION_DENSE.md).
Everything that decides the outcome is fixed in that file and in the constants
below; nothing here may change after the run.

Part 1, regression with whole cells held out. Quadratic ridge regression
(junction_pairwise_regime_analysis.cv_predict) of log10 rho (samples with rho > 0),
log10 a and log10 N a^2 (all samples) on log10 S alone, and on log10 S with the nine
values of C1, C4, C16 or P. Folds are 5 groups of whole cells (seeded permutation of
the cells), so a test cell never appears in training. R^2 = 1 - MSE / variance of the
target over the test samples. Intervals: 2000 bootstrap resamples of the networks
within each cell, applied to the out-of-fold predictions.

Part 2, Cramer-Rao bound. Features y = log10 (S, and the shell values), parameters
theta = (log10 a, log10 N a^2, log10 rho), samples with rho > 0. The mean of y is a
cubic polynomial in theta fitted to the group means (cell, rho); the covariance is the
pooled within-group covariance of the networks, plus an isotropic measurement noise
of relative size sigma_meas. Fisher information F = J' Sigma^-1 J at each group,
bound = diag(F^-1), reported as the standard deviation in log10 units. A feature set
whose F is singular (S alone, one equation for three unknowns) has an infinite bound.

Usage: python junction_pairwise_dense_analysis.py <partial.jsonl> [--boot 2000] [--out summary.json]
"""
import sys, json
import numpy as np
import junction_pairwise_analysis as ja
import junction_pairwise_regime_analysis as rg

FOLDS, SEED, BOOT = 5, 0, 2000
FEATURES = ("S", "S+C1", "S+C4", "S+C16", "S+P")
D1_MARGIN, D2_MARGIN, D3_MARGIN, D4_MARGIN = 0.80, 0.15, 0.40, -0.05
SIGMA_MEAS = (0.0, 0.05, 0.10)


def feature_matrix(rows, name):
    S = np.log10([r["S"] for r in rows])[:, None]
    if name == "S":
        return S
    key = name.split("+")[1]
    return np.hstack([S, np.log10(np.array([r[key] for r in rows]))])


def r2(y, pred, w):
    mse = np.sum(w * (pred - y) ** 2) / w.sum()
    mu = np.sum(w * y) / w.sum()
    var = np.sum(w * (y - mu) ** 2) / w.sum()
    return 1 - mse / var


def rmse(y, pred, w):
    return float(np.sqrt(np.sum(w * (pred - y) ** 2) / w.sum()))


def regression(rows, boot, rng):
    cells = sorted({r["cell"] for r in rows})
    order = np.random.default_rng(SEED).permutation(len(cells))
    cf = {c: int(order[i] % FOLDS) for i, c in enumerate(cells)}
    fold = np.array([cf[r["cell"]] for r in rows])
    a = np.array([r["a"] for r in rows]); x = np.array([r["x"] for r in rows]); rho = np.array([r["rho"] for r in rows])
    cell = np.array([r["cell"] for r in rows]); net = np.array([f"{r['cell']}|{r['net']}" for r in rows])
    targets = {"log10 rho": (np.log10(np.where(rho > 0, rho, 1.0)), rho > 0),
               "log10 a": (np.log10(a), np.ones(len(rows), bool)),
               "log10 Na2": (np.log10(x), np.ones(len(rows), bool))}
    nets = sorted(set(net)); nid = {n: i for i, n in enumerate(nets)}
    sample_net = np.array([nid[n] for n in net])
    net_cell = {}
    for n in nets:
        net_cell.setdefault(n.split("|")[0], []).append(nid[n])
    boots = []
    for _ in range(boot):
        w = np.zeros(len(nets))
        for ids in net_cell.values():
            np.add.at(w, rng.choice(ids, len(ids)), 1)
        boots.append(w)
    out, preds = {}, {}
    for tname, (y, m) in targets.items():
        out[tname] = {}
        for fs in FEATURES:
            X = feature_matrix([r for r, k in zip(rows, m) if k], fs)
            pred = rg.cv_predict(X, y[m], fold[m])
            preds[(tname, fs)] = (pred, y[m], sample_net[m], m)
            w1 = np.ones(m.sum())
            draws = np.array([r2(y[m], pred, w[sample_net[m]]) for w in boots])
            out[tname][fs] = {"r2": float(r2(y[m], pred, w1)), "rmse": rmse(y[m], pred, w1),
                              "sd": float(y[m].std()), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    return out, preds, boots, dict(a=a, x=x, rho=rho, cell=cell)


def diff_ci(preds, boots, tname, fs1, fs2):
    p1, y, sn, _ = preds[(tname, fs1)]; p2 = preds[(tname, fs2)][0]
    d = np.array([r2(y, p1, w[sn]) - r2(y, p2, w[sn]) for w in boots])
    w1 = np.ones(len(y))
    return float(r2(y, p1, w1) - r2(y, p2, w1)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def cramer_rao(rows):
    """Standard deviation of the bound, by group, for each feature set and measurement noise."""
    rows = [r for r in rows if r["rho"] > 0]
    theta = np.array([[np.log10(r["a"]), np.log10(r["x"]), np.log10(r["rho"])] for r in rows])
    gid = {}
    for i, r in enumerate(rows):
        gid.setdefault((r["cell"], r["rho"]), []).append(i)
    keys = sorted(gid)
    th_g = np.array([theta[gid[k][0]] for k in keys])
    mu_c, sd_c = th_g.mean(0), th_g.std(0)

    def poly(t, deriv=None):
        z = (np.atleast_2d(t) - mu_c) / sd_c
        terms, dterms = [], []
        for i in range(4):
            for j in range(4 - i):
                for k in range(4 - i - j):
                    terms.append(z[:, 0] ** i * z[:, 1] ** j * z[:, 2] ** k)
                    if deriv is not None:
                        e = [i, j, k]; e2 = list(e)
                        if e[deriv] == 0:
                            dterms.append(np.zeros(len(z)))
                        else:
                            e2[deriv] -= 1
                            dterms.append(e[deriv] * z[:, 0] ** e2[0] * z[:, 1] ** e2[1] * z[:, 2] ** e2[2] / sd_c[deriv])
        return np.column_stack(dterms if deriv is not None else terms)

    out = {}
    for fs in FEATURES:
        Y = feature_matrix(rows, fs)
        Ym = np.array([Y[gid[k]].mean(0) for k in keys])
        A = poly(th_g)
        coef, *_ = np.linalg.lstsq(A, Ym, rcond=None)
        fit_r2 = float(1 - np.sum((A @ coef - Ym) ** 2) / np.sum((Ym - Ym.mean(0)) ** 2))
        res = np.vstack([Y[gid[k]] - Ym[g] for g, k in enumerate(keys)])
        Sigma = res.T @ res / (len(res) - len(keys))
        J = np.stack([poly(th_g, d) @ coef for d in range(3)], axis=-1)      # groups x d_features x 3
        out[fs] = {"mean_fit_r2": fit_r2, "by_noise": {}}
        for sm in SIGMA_MEAS:
            Sn = Sigma + (sm / np.log(10)) ** 2 * np.eye(Sigma.shape[0])
            Si = np.linalg.inv(Sn)
            sig = np.full((len(keys), 3), np.inf)
            for g in range(len(keys)):
                F = J[g].T @ Si @ J[g]
                if np.linalg.matrix_rank(F, tol=1e-9 * np.trace(F)) == 3:
                    sig[g] = np.sqrt(np.diag(np.linalg.inv(F)))
            out[fs]["by_noise"][sm] = {"keys": keys, "sig": sig}
    return out, th_g, keys


def main():
    path = sys.argv[1]
    boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else BOOT
    rows = [r for r in ja.load(path) if r["variant"] == "uniform"]
    rng = np.random.default_rng(SEED)
    print(f"{len(rows)} samples; cells {len({r['cell'] for r in rows})}; "
          f"networks {len({(r['cell'], r['net']) for r in rows})}; rho {sorted({r['rho'] for r in rows})}")
    reg, preds, boots, meta = regression(rows, boot, rng)
    summary = {"regression": reg}
    print("\nPart 1: whole cells held out (R^2 [95% interval])")
    print(f"{'target':10s} " + " ".join(f"{f:>22s}" for f in FEATURES))
    for t in reg:
        print(f"{t:10s} " + " ".join(f"{reg[t][f]['r2']:6.3f} [{reg[t][f]['ci'][0]:.3f},{reg[t][f]['ci'][1]:.3f}]".rjust(22) for f in FEATURES))
    d1 = reg["log10 rho"]["S+P"]; lo = d1["ci"][0]
    h = {"D1": {"r2": d1["r2"], "ci": d1["ci"], "confirmed": bool(lo > D1_MARGIN)}}
    e, ci = diff_ci(preds, boots, "log10 a", "S+P", "S"); h["D2"] = {"diff": e, "ci": ci, "confirmed": bool(ci[0] > D2_MARGIN)}
    e, ci = diff_ci(preds, boots, "log10 Na2", "S+P", "S"); h["D3"] = {"diff": e, "ci": ci, "confirmed": bool(ci[0] > D3_MARGIN)}
    e, ci = diff_ci(preds, boots, "log10 rho", "S+C4", "S+P"); h["D4"] = {"diff": e, "ci": ci, "confirmed": bool(ci[0] > D4_MARGIN)}
    print(f"\nD1 R2 of log rho, S+P: {d1['r2']:.3f} [{d1['ci'][0]:.3f}, {d1['ci'][1]:.3f}] -> {'CONFIRMED' if h['D1']['confirmed'] else 'not confirmed'} (lower bound > {D1_MARGIN})")
    print(f"D2 R2 of log a, (S+P) - S: {h['D2']['diff']:+.3f} [{h['D2']['ci'][0]:+.3f}, {h['D2']['ci'][1]:+.3f}] -> {'CONFIRMED' if h['D2']['confirmed'] else 'not confirmed'} (lower bound > {D2_MARGIN})")
    print(f"D3 R2 of log Na2, (S+P) - S: {h['D3']['diff']:+.3f} [{h['D3']['ci'][0]:+.3f}, {h['D3']['ci'][1]:+.3f}] -> {'CONFIRMED' if h['D3']['confirmed'] else 'not confirmed'} (lower bound > {D3_MARGIN})")
    print(f"D4 R2 of log rho, (S+C4) - (S+P): {h['D4']['diff']:+.3f} [{h['D4']['ci'][0]:+.3f}, {h['D4']['ci'][1]:+.3f}] -> {'CONFIRMED' if h['D4']['confirmed'] else 'not confirmed'} (lower bound > {D4_MARGIN})")
    summary["hypotheses"] = h

    # descriptive: calibration of log rho, and RMSE of log rho by regime, S+P
    p, y, sn, m = preds[("log10 rho", "S+P")]
    rho_m = meta["rho"][m]; a_m = meta["a"][m]; x_m = meta["x"][m]
    print("\nDescriptive: predicted log10 rho by true rho (S+P): median [25%, 75%]")
    cal = {}
    for lv in sorted(set(rho_m)):
        q = np.percentile(p[rho_m == lv], [25, 50, 75]); cal[str(lv)] = q.tolist()
        print(f"   rho {lv:7.2f} (log {np.log10(lv):+.2f}): {q[1]:+.2f} [{q[0]:+.2f}, {q[2]:+.2f}]")
    summary["calibration"] = cal
    abins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.1)]; xbins = [(0, 7), (7, 10), (10, 13), (13, 17)]
    print("\nDescriptive: RMSE of log10 rho by regime, S+P (rows: a, columns: N a^2)")
    print("          " + "".join(f"{f'{lo}-{hi}':>10s}" for lo, hi in xbins))
    reg_map = {}
    for alo, ahi in abins:
        line = f"a {alo:.1f}-{ahi:.1f} "
        for xlo, xhi in xbins:
            k = (a_m > alo) & (a_m <= ahi) & (x_m > xlo) & (x_m <= xhi)
            v = float(np.sqrt(np.mean((p[k] - y[k]) ** 2))) if k.sum() else float("nan")
            reg_map[f"a{alo}-{ahi}|x{xlo}-{xhi}"] = v; line += f"{v:10.3f}"
        print(line)
    summary["rmse_logrho_by_regime"] = reg_map

    # Part 2
    cr, th_g, keys = cramer_rao(rows)
    print("\nPart 2: Cramer-Rao bound, standard deviation in log10 units, median over groups with rho > 0")
    print(f"{'features':8s} {'noise':>6s} {'log a':>9s} {'log Na2':>9s} {'log rho':>9s}   (fit R2 of the mean model)")
    summary["cramer_rao"] = {}
    for fs in FEATURES:
        for sm, v in cr[fs]["by_noise"].items():
            sig = v["sig"]
            med = [float(np.median(sig[:, j][np.isfinite(sig[:, j])])) if np.isfinite(sig[:, j]).any() else float("inf") for j in range(3)]
            frac = float(np.isfinite(sig[:, 0]).mean())
            summary["cramer_rao"][f"{fs}|{sm}"] = {"median_sd": med, "finite_fraction": frac, "mean_fit_r2": cr[fs]["mean_fit_r2"]}
            print(f"{fs:8s} {sm:6.2f} {med[0]:9.3f} {med[1]:9.3f} {med[2]:9.3f}   ({cr[fs]['mean_fit_r2']:.3f}; finite in {frac:.0%})")
    if "--out" in sys.argv:
        json.dump(summary, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
