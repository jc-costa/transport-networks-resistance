"""
Pre-registered analysis of the confirmation run (PREREGISTRATION_JUNCTION.md).
Everything that decides the outcome is fixed in that file and in the constants
below; nothing here may change after the run.

Usage: python junction_pairwise_confirm_analysis.py <partial.jsonl> [--boot 2000] [--out summary.json]
"""
import sys, json
import numpy as np
import junction_pairwise_analysis as ja

BIN = 0.10             # decades of log10 S
MIN_PER_CLASS = 8
FOLDS, SEED, SHRINK = 5, 0, 0.1
BOOT = 2000
H1_MARGIN, H2_MARGIN, H3_MARGIN = 0.20, 0.10, 0.10
NOISE_SIGMA, NOISE_DRAWS = 0.10, 5


def oof(rows, task, feats, folds_of):
    """Per bin: the samples, their labels and out-of-fold correctness for each feature set."""
    key = {"rho": "rho", "a": "a"}[task]
    logS = np.log10([r["S"] for r in rows])
    b = np.floor(logS / BIN).astype(int)
    out = []
    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        lab = np.array([rows[i][key] for i in idx])
        keep = [c for c in np.unique(lab) if (lab == c).sum() >= MIN_PER_CLASS]
        if len(keep) < 2:
            continue
        sel = idx[np.isin(lab, keep)]
        y = np.array([rows[i][key] for i in sel])
        fold = np.array([folds_of[(rows[i]["cell"], rows[i]["net"])] for i in sel])
        corr = {}
        for name in feats:
            X = ja.features([rows[i] for i in sel], name)
            c = np.zeros(len(sel), bool)
            classes = np.unique(y)
            for f in range(FOLDS):
                tr, te = fold != f, fold == f
                if not te.any():
                    continue
                mu = np.array([X[tr & (y == k)].mean(0) for k in classes])
                Xc = X[tr] - mu[np.searchsorted(classes, y[tr])]
                cov = Xc.T @ Xc / max(len(Xc) - len(classes), 1)
                cov = (1 - SHRINK) * cov + SHRINK * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
                W = np.linalg.solve(cov, mu.T)
                score = X[te] @ W - 0.5 * np.sum(mu.T * W, axis=0)
                c[te] = classes[np.argmax(score, 1)] == y[te]
            corr[name] = c
        out.append(dict(sel=sel, y=y, corr=corr, group=np.array([rows[i]["gid"] for i in sel])))
    return out


def metric(bins, name, w):
    """Weighted accuracy above the majority rate, averaged over bins by weight."""
    num = den = 0.0
    for bn in bins:
        ww = w[bn["group"]]
        tot = ww.sum()
        if tot == 0:
            continue
        acc = (ww * bn["corr"][name]).sum() / tot
        chance = max(ww[bn["y"] == k].sum() for k in np.unique(bn["y"])) / tot
        num += tot * (acc - chance); den += tot
    return num / den if den else np.nan


def analyse(rows, task, feats, boot, rng):
    groups = sorted({(r["cell"], r["net"]) for r in rows})
    gid = {g: i for i, g in enumerate(groups)}
    for r in rows:
        r["gid"] = gid[(r["cell"], r["net"])]
    order = np.random.default_rng(SEED).permutation(len(groups))
    folds_of = {g: int(order[i] % FOLDS) for i, g in enumerate(groups)}
    bins = oof(rows, task, feats, folds_of)
    n = len(groups)
    w0 = np.ones(n)
    point = {f: metric(bins, f, w0) for f in feats}
    cells = {}
    for g, i in gid.items():
        cells.setdefault(g[0], []).append(i)
    draws = {f: [] for f in feats}
    for _ in range(boot):
        w = np.zeros(n)
        for ids in cells.values():                       # resample networks within each cell
            np.add.at(w, rng.choice(ids, len(ids)), 1)
        for f in feats:
            draws[f].append(metric(bins, f, w))
    return bins, point, {f: np.array(v) for f, v in draws.items()}


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    path = sys.argv[1]
    boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else BOOT
    rows = ja.load(path)
    rng = np.random.default_rng(SEED)
    summary = {}
    uni = [r for r in rows if r["variant"] == "uniform"]
    print(f"{len(rows)} samples; uniform {len(uni)}; cells {len({r['cell'] for r in uni})}; "
          f"networks {len({(r['cell'], r['net']) for r in uni})}")

    # H1: S+P identifies rho better than S alone
    bins, pt, dr = analyse(uni, "rho", ("S", "P", "C16"), boot, rng)
    d = dr["P"] - dr["S"]; lo, hi = ci(d)
    h1 = {"S": pt["S"], "S+P": pt["P"], "S+C16": pt["C16"], "diff": pt["P"] - pt["S"], "ci": [lo, hi],
          "bins": len(bins), "confirmed": bool(lo > H1_MARGIN)}
    print(f"\nH1 identify rho: S {pt['S']:+.3f}, S+P {pt['P']:+.3f}, diff {h1['diff']:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]  ({len(bins)} bins)  -> {'CONFIRMED' if h1['confirmed'] else 'not confirmed'} "
          f"(lower bound > {H1_MARGIN})")
    summary["H1"] = h1

    # H2: S+P does not identify a (equivalence-style: upper bound below the margin)
    bins, pt, dr = analyse(uni, "a", ("S", "P"), boot, rng)
    lo, hi = ci(dr["P"])
    h2 = {"S+P": pt["P"], "ci": [lo, hi], "bins": len(bins), "confirmed": bool(hi < H2_MARGIN)}
    print(f"H2 identify a: S+P {pt['P']:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({len(bins)} bins)  -> "
          f"{'CONFIRMED (not identified)' if h2['confirmed'] else 'not confirmed'} (upper bound < {H2_MARGIN})")
    summary["H2"] = h2

    # H3: the same for the two variable-junction rules, classes = the mean ratio
    h3 = {}
    for var in sorted({r["variant"] for r in rows} - {"uniform"}):
        sub = [r for r in rows if r["variant"] == var]
        bins, pt, dr = analyse(sub, "rho", ("S", "P"), boot, rng)
        d = dr["P"] - dr["S"]; lo, hi = ci(d)
        h3[var] = {"S": pt["S"], "S+P": pt["P"], "diff": pt["P"] - pt["S"], "ci": [lo, hi], "bins": len(bins),
                   "confirmed": bool(lo > H3_MARGIN)}
        print(f"H3 {var}: S {pt['S']:+.3f}, S+P {pt['P']:+.3f}, diff {pt['P'] - pt['S']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  ({len(bins)} bins)  -> "
              f"{'CONFIRMED' if lo > H3_MARGIN else 'not confirmed'} (lower bound > {H3_MARGIN})")
    summary["H3"] = h3

    # H4 (descriptive): 10 % independent noise on S and on every shell value
    accs = []
    for d_ in range(NOISE_DRAWS):
        nrows = ja.noisy(uni, NOISE_SIGMA, np.random.default_rng(1000 + d_))
        for r, o in zip(nrows, uni):
            r["gid"] = o["gid"]
        gids = {(r["cell"], r["net"]) for r in nrows}
        order = np.random.default_rng(SEED).permutation(len(gids))
        folds_of = {g: int(order[i] % FOLDS) for i, g in enumerate(sorted(gids))}
        bins = oof(nrows, "rho", ("S", "P"), folds_of)
        w0 = np.ones(max(r["gid"] for r in nrows) + 1)
        accs.append(metric(bins, "P", w0) - metric(bins, "S", w0))
    summary["H4"] = {"sigma": NOISE_SIGMA, "diff_mean": float(np.mean(accs)), "draws": [float(a) for a in accs]}
    print(f"H4 (descriptive) 10 % noise: S+P minus S for rho, mean of {NOISE_DRAWS} draws {np.mean(accs):+.3f}")
    if "--out" in sys.argv:
        json.dump(summary, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
