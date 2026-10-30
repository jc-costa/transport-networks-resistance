"""
Analysis of the confirmatory 2D pairwise run, exactly as fixed in
PREREGISTRATION.md (primary: exact McNemar of S+C16 and S+P against S on one
stratified 5-fold split per target, Holm over the two tests at alpha = 0.05;
secondary: effect sizes, the thesis's pre-set criterion, per-target accuracy).

Reuses pairwise2d_analysis (fill_curve, matrix, lda_fit, classify) unchanged.
Usage: python pairwise2d_confirm_analysis.py [--results pairwise2d_confirm_results] [--json out.json]
"""
import os, sys, json, argparse
from math import comb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv; sys.argv = sys.argv[:1]
import pairwise2d_analysis as pa  # noqa: E402
sys.argv = _argv

TESTED = ("S+C16", "S+P")
MIN_NETWORKS = 10
ALPHA = 0.05


def load(results):
    rows = [json.loads(l) for l in open(os.path.join(results, "partial.jsonl")) if l.strip()]
    nets, excluded = [], []
    for r in rows:
        if r["n"] < MIN_NETWORKS:
            excluded.append((r["id"], r["n"])); continue
        for i in range(r["n"]):
            feats, ok = {}, True
            for s in pa.SETS:
                ly, _ = pa.fill_curve(r[s][i])
                if ly is None:
                    ok = False; break
                feats[s] = ly
            if ok:
                nets.append({"unit": r["id"], "R_target": r["R_target"], "logS": np.log10(r["R_eff_values"][i]), "feats": feats})
    return rows, nets, excluded


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "pairwise2d_confirm_results"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows, nets, excluded = load(args.results)
    targets = sorted({n["R_target"] for n in nets})
    subs = {t: [n for n in nets if n["R_target"] == t] for t in targets}
    print(f"{len(nets)} networks, {len({n['unit'] for n in nets})} candidates, {len(targets)} targets; excluded: {excluded}")

    # ---- primary: one stratified split per target (seed 0), as pairwise2d_robustness.py
    corr = {i: [] for i in pa.INFO}; units = []
    for t in targets:
        sub = subs[t]; y = np.array([n["unit"] for n in sub])
        rng = np.random.default_rng(0); fold = np.empty(len(y), int)
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0]); fold[idx] = np.arange(len(idx)) % 5
        for i in pa.INFO:
            F = pa.matrix(sub, i); ok = np.zeros(len(y), bool)
            for f in range(5):
                te = fold == f; ok[te] = pa.lda_fit(F[~te], y[~te])(F[te]) == y[te]
            corr[i].append(ok)
        units.append(y)
    units = np.concatenate(units)
    S = np.concatenate(corr["S"])
    raw = {}
    for i in TESTED:
        X = np.concatenate(corr[i]); b = int((~S & X).sum()); c = int((S & ~X).sum())
        raw[i] = dict(b=b, c=c, p=mcnemar(b, c))
    order = sorted(TESTED, key=lambda i: raw[i]["p"])          # Holm step-down
    adj, running = {}, 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(TESTED) - r) * raw[i]["p"])); adj[i] = running
    out = {"n_networks": len(nets), "excluded": excluded, "primary": {}}
    print("\nPRIMARY (pre-registered): exact McNemar vs S, Holm over two tests")
    for i in TESTED:
        b, c = raw[i]["b"], raw[i]["c"]
        verdict = ("CONFIRMED" if adj[i] < ALPHA and b > c else
                   "REFUTED IN DIRECTION" if adj[i] < ALPHA and b < c else "NOT CONFIRMED")
        # effect: paired difference in share correct, bootstrap within candidates
        X = np.concatenate(corr[i]); d = X.astype(float) - S.astype(float)
        rng = np.random.default_rng(0); boots = []
        groups = [np.where(units == u)[0] for u in np.unique(units)]
        for _ in range(2000):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]); boots.append(d[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["primary"][i] = dict(gained=b, lost=c, p=raw[i]["p"], p_holm=adj[i], verdict=verdict,
                                 diff=float(d.mean()), ci95=[float(lo), float(hi)])
        print(f"  {i:6s} gained {b}, lost {c}; p = {raw[i]['p']:.2g}, Holm p = {adj[i]:.2g} -> {verdict}; "
              f"difference in share correct {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nSECONDARY (descriptive): pre-set procedure, accuracy above chance, mean over targets")
    ex = {}
    for i in pa.INFO:
        vals = []
        for t in targets:
            y = np.array([n["unit"] for n in subs[t]])
            vals.append(pa.classify(pa.matrix(subs[t], i), y, 1) - 1 / len(set(y)))
        ex[i] = float(np.mean(vals))
    for i in pa.INFO:
        extra = "" if i == "S" else f"  minus S {ex[i] - ex['S']:+.3f}  pre-set criterion (>= 0.20): {'pass' if ex[i] >= 0.20 else 'fail'}"
        print(f"  {i:6s} {ex[i]:+.3f}{extra}")
    out["secondary"] = {"excess_over_chance": ex}
    print("\nper-target accuracy (primary split):")
    per = {}
    for k, t in enumerate(targets):
        per[str(t)] = {i: float(corr[i][k].mean()) for i in pa.INFO}
        print(f"  {t:>6}: " + "  ".join(f"{i} {per[str(t)][i]:.2f}" for i in pa.INFO))
    out["per_target"] = per
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1); print("wrote", args.json)


if __name__ == "__main__":
    main()
