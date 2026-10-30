"""
Inversion of (N, a, d) from the onion curve, as a regression validated out
of sample, on the 3D grid of cluster/onion_grid_results.

The ANOVA of r_eff_ij_vs_N_a_d.ipynb shows that curve shape differs between
parameter triples that share the same scalar R_eff. This script asks the
question the other way round: given one realization's curve, how well can
(N, a, d) be predicted, and how much better than from the scalar alone?

Feature sets (all in log10):
  scalar       : log R_eff of the realization (1 feature)
  curve        : log R_eff,ij at the 9 shells (9 features)
  curve+scalar : both (10 features)
Targets: (log10 N, log10 a, log10 d).
Models: ridge regression with a quadratic feature expansion, and k-nearest
neighbours in standardized feature space; both numpy-only.
Validation: grouped K-fold by (N, a, d) combination, so every test
realization comes from a combination the model never saw; the metric is the
RMSE of each log10 target, i.e. the typical factor error, and R^2.

Two datasets are used, because the stored per-layer lists are aligned by
realization only when every realization has every shell (see
run_onion_grid_task.py, per_realization_curves):
  realizations : combinations with complete shells, one sample per realization
  means        : every usable combination, one sample = its mean curve
Usage: python curve_inversion.py [--folds 10] [--k 5] [--json out.json]
"""
import os, sys, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [round(0.1 * i, 2) for i in range(1, 10)]


def load_rows(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "chunk_*.json"))):
        rows += json.load(open(p))
    return [r for r in rows if r["n"] > 0 and r["onion_layer_curves"]]


def dataset_realizations(rows):
    X, Y, G, S = [], [], [], []
    for r in rows:
        c = r["onion_layer_curves"]
        if "per_realization_curves" in r:                       # aligned record (new task script)
            curves = r["per_realization_curves"]
            for i, cv in enumerate(curves):
                vals = [cv.get(str(l)) for l in LAYERS]
                if any(v is None for v in vals): continue
                X.append(vals); S.append(r["R_eff_values"][i]); Y.append([r["N"], r["a"], r["d"]]); G.append((r["N"], r["a"], r["d"]))
        else:                                                    # old record: usable only when complete
            if len(c) != 9 or any(c[str(l)]["n_realizations"] != r["n"] for l in LAYERS): continue
            for i in range(r["n"]):
                X.append([c[str(l)]["per_realization_means"][i] for l in LAYERS]); S.append(r["R_eff_values"][i])
                Y.append([r["N"], r["a"], r["d"]]); G.append((r["N"], r["a"], r["d"]))
    return np.log10(np.array(X)), np.log10(np.array(Y)), np.array([f"{n}|{a}|{d}" for n, a, d in G]), np.log10(np.array(S))[:, None]


def dataset_means(rows):
    X, Y, G, S = [], [], [], []
    for r in rows:
        c = r["onion_layer_curves"]
        if any(str(l) not in c for l in LAYERS): continue
        X.append([c[str(l)]["mean"] for l in LAYERS]); S.append(float(np.mean(r["R_eff_values"])))
        Y.append([r["N"], r["a"], r["d"]]); G.append((r["N"], r["a"], r["d"]))
    return np.log10(np.array(X)), np.log10(np.array(Y)), np.array([f"{n}|{a}|{d}" for n, a, d in G]), np.log10(np.array(S))[:, None]


def quad(X):
    n, p = X.shape
    cols = [np.ones(n)] + [X[:, i] for i in range(p)] + [X[:, i] * X[:, j] for i in range(p) for j in range(i, p)]
    return np.column_stack(cols)


def lin(X):
    return np.column_stack([np.ones(len(X)), X])


def ridge_fit(Xtr, Ytr, lam, expand=lin):
    """Ridge regression on standardized features; expand=quad adds pairwise
    products (66 parameters for 10 features), which overfits below a few
    hundred samples, so the linear expansion is the default."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Z = expand((Xtr - mu) / sd)
    reg = lam * np.eye(Z.shape[1]); reg[0, 0] = 0
    W = np.linalg.solve(Z.T @ Z + reg, Z.T @ Ytr)
    return lambda Xte: expand((Xte - mu) / sd) @ W


def knn_fit(Xtr, Ytr, k):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Ztr = (Xtr - mu) / sd
    def predict(Xte):
        Zte = (Xte - mu) / sd
        d2 = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d2, axis=1)[:, :k]
        return Ytr[idx].mean(1)
    return predict


def grouped_folds(groups, n_folds, seed=0):
    keys = sorted(set(groups.tolist())); rng = np.random.default_rng(seed); rng.shuffle(keys)
    fold_of = {k: i % n_folds for i, k in enumerate(keys)}
    return np.array([fold_of[g] for g in groups.tolist()])


def cross_validate(X, Y, groups, fitter, n_folds):
    folds = grouped_folds(groups, n_folds); pred = np.zeros_like(Y)
    for f in range(n_folds):
        te = folds == f; tr = ~te
        pred[te] = fitter(X[tr], Y[tr])(X[te])
    rmse = np.sqrt(((pred - Y) ** 2).mean(0)); r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return rmse, r2, pred


def plot_averaging(av, ks, path):
    """Out-of-sample R^2 of each target against the number of realizations averaged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for j, sym in enumerate(("N", "a", "d")):
        ax = axes[j]
        for fname, mk, lab in (("scalar", "s--", "scalar R_eff only"), ("curve+scalar", "o-", "onion curve + scalar")):
            ax.plot(ks, [av[f"{fname}/k={k}"]["r2"][j] for k in ks], mk, label=lab)
        ax.axhline(0, color="gray", lw=0.8); ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("realizations averaged, k"); ax.grid(True, alpha=0.3)
        # panel label inside the axes rather than a title (the thesis caption carries the title)
        ax.text(0.04, 0.95, f"target: $\\log_{{10}} {sym}$", transform=ax.transAxes, ha="left", va="top", fontsize=11)
    axes[0].set_ylabel("out-of-sample R^2"); axes[0].set_ylim(-0.15, 0.85); axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig)
    print(f"wrote {path}")

    # the same three panels as separate files (<path>_N.png, _a.png, _d.png), for the
    # thesis, which sets them as subfigures (a), (b), (c) and describes them in the caption
    stem, ext = os.path.splitext(path)
    for j, sym in enumerate(("N", "a", "d")):
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        for fname, mk, lab in (("scalar", "s--", "scalar R_eff only"), ("curve+scalar", "o-", "onion curve + scalar")):
            ax.plot(ks, [av[f"{fname}/k={k}"]["r2"][j] for k in ks], mk, label=lab)
        ax.axhline(0, color="gray", lw=0.8); ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("realizations averaged, k"); ax.set_ylabel("out-of-sample R^2"); ax.set_ylim(-0.15, 0.85); ax.grid(True, alpha=0.3)
        ax.text(0.04, 0.95, f"target: $\\log_{{10}} {sym}$", transform=ax.transAxes, ha="left", va="top", fontsize=11)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); out = f"{stem}_{sym}{ext}"
        fig.savefig(out, dpi=150, metadata={"Software": None}); plt.close(fig); print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=10); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--json", default=None); ap.add_argument("--results", default=os.path.join(HERE, "onion_grid_results"))
    ap.add_argument("--quad", action="store_true", help="also report the quadratic ridge expansion")
    ap.add_argument("--figure", default=None, help="write the averaging plot (out-of-sample R^2 against k) to this PNG")
    args = ap.parse_args()
    rows = load_rows(args.results)
    out = {}
    for name, builder in (("realizations", dataset_realizations), ("means", dataset_means)):
        X, Y, G, S = builder(rows)
        if len(Y) == 0: print(f"{name}: no samples"); continue
        n_comb = len(set(G.tolist()))
        print(f"\n== {name}: {len(Y)} samples from {n_comb} combinations, {args.folds}-fold grouped by combination")
        print(f"   {'features':14s} {'model':10s}   RMSE log10 (N, a, d)          R^2 (N, a, d)")
        out[name] = {"n_samples": int(len(Y)), "n_combinations": n_comb, "results": {}}
        base = np.sqrt(((Y - Y.mean(0)) ** 2).mean(0))
        print(f"   {'(none)':14s} {'mean':10s}   {base[0]:.3f} {base[1]:.3f} {base[2]:.3f}          0     0     0")
        for fname, F in (("scalar", S), ("curve", X), ("curve+scalar", np.hstack([X, S]))):
            models = [("ridge", lambda a, b: ridge_fit(a, b, args.lam)), (f"knn-{args.k}", lambda a, b: knn_fit(a, b, args.k))]
            if args.quad: models.append(("ridge-quad", lambda a, b: ridge_fit(a, b, args.lam, quad)))
            for mname, fitter in models:
                rmse, r2, _ = cross_validate(F, Y, G, fitter, args.folds)
                print(f"   {fname:14s} {mname:10s}   {rmse[0]:.3f} {rmse[1]:.3f} {rmse[2]:.3f}      {r2[0]:5.2f} {r2[1]:5.2f} {r2[2]:5.2f}")
                out[name]["results"][f"{fname}/{mname}"] = {"rmse_log10": rmse.tolist(), "r2": r2.tolist(), "factor_error": (10 ** rmse).tolist()}
    # how the inversion improves when the measurement is repeated: features
    # averaged over k realizations of the same combination (complete
    # combinations only), test combinations never seen in training
    X, Y, G, S = dataset_realizations(rows)
    if len(Y):
        KS = (1, 2, 3, 6, 12, 24)
        # a fixed set of combinations for every k, so the scores stay comparable:
        # only combinations with at least max(KS) realizations
        counts = {g: int((G == g).sum()) for g in set(G.tolist())}
        keys = sorted(g for g, c in counts.items() if c >= max(KS)); rng = np.random.default_rng(0)
        print(f"\n== averaging over k realizations per combination (ridge, {args.folds}-fold grouped by combination, 10 random draws; {len(keys)} combinations with >= {max(KS)} realizations)")
        print(f"   {'features':14s} k   RMSE log10 (N, a, d)          R^2 (N, a, d)")
        out["averaging"] = {}
        for fname, F in (("scalar", S), ("curve+scalar", np.hstack([X, S]))):
            for k in KS:
                rms, r2s = [], []
                for _ in range(10):
                    Xs, Ys, Gs = [], [], []
                    for key in keys:
                        idx = np.where(G == key)[0]
                        if len(idx) < k: continue
                        for _r in range(max(1, len(idx) // k)):
                            sel = rng.choice(idx, k, replace=False); Xs.append(F[sel].mean(0)); Ys.append(Y[idx[0]]); Gs.append(key)
                    rmse, r2, _ = cross_validate(np.array(Xs), np.array(Ys), np.array(Gs), lambda a, b: ridge_fit(a, b, args.lam), args.folds)
                    rms.append(rmse); r2s.append(r2)
                rmse, r2 = np.mean(rms, 0), np.mean(r2s, 0)
                print(f"   {fname:14s} {k}   {rmse[0]:.3f} {rmse[1]:.3f} {rmse[2]:.3f}      {r2[0]:5.2f} {r2[1]:5.2f} {r2[2]:5.2f}")
                out["averaging"][f"{fname}/k={k}"] = {"rmse_log10": rmse.tolist(), "r2": r2.tolist()}
        if args.figure:
            plot_averaging(out["averaging"], KS, args.figure)
    print("\nRMSE in log10 units: 0.1 means a typical factor of 1.26, 0.3 a factor of 2.")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2); print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
