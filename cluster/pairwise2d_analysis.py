"""
E7 analysis: reads <results>/partial.jsonl written by run_pairwise2d_task.py and scores the
information sets against the PRE-SET criteria of the revision plan (unchanged):

  GAIN        Part A accuracy at k = 1 at least 0.20 above chance
              OR Part B R^2 for log a at k = 1 at least 0.10 above the scalar S
  NEGLIGIBLE  both within 0.05: |accuracy - chance| <= 0.05 AND |R^2(set) - R^2(S)| <= 0.05
  NEITHER     anything else
One verdict per information set: S+C1, S+C4, S+C16, S+P (S alone is the baseline row).

Choices fixed BEFORE any E7 result was seen (written here so they can be checked):
 * Features are log10 of the stored values (ohm). S = [log R_term]; S+X = [log R_term] plus the
   9 shells of X, X in {C1, C4, C16, P} (definitions in run_pairwise2d_task.py).
 * Empty shells (no node in that shell, common in sparse networks) are filled by linear
   interpolation of log10 R over shell distance between the nearest non-empty shells, and by the
   nearest non-empty shell beyond the ends (numpy.interp). A network with a curve that has no
   non-empty shell is dropped from every set. The share of filled values is reported.
 * Part A "accuracy at k = 1": for each information set, the MEAN OVER TARGETS of
   (accuracy - chance), chance = 1 / number of candidates that share the target. Per-target
   accuracies are reported as well. Candidates with fewer than 10 networks are excluded
   (candidate 19, 890 ohm at a = 0.2, has 4), which leaves 21 candidates over 7 targets.
 * Part A classifier: linear discriminant analysis on features standardized with the training
   fold; pooled within-class covariance shrunk toward its mean variance (gamma = 0.1); equal
   priors. Validation: stratified 5-fold, 10 repeats, seed 0; accuracy pooled over the folds of
   a repeat, then averaged over repeats. Every information set sees the same folds.
 * Part B "R^2 for log a at k = 1": ridge, lambda = 1, linear, standardized features
   (curve_inversion.ridge_fit); 10-fold CV grouped by (N, a) cell (curve_inversion.grouped_folds,
   seed 0); R^2 of the pooled out-of-fold predictions; every network used once. Evaluated on the
   fixed set of cells that reached 30 usable networks, so k = 1 and k > 1 use the same cells.
   Factor error = 10^RMSE(log10).
 * k > 1 is descriptive only. Each candidate's / cell's networks are shuffled and cut into
   disjoint groups of k whose log features are averaged; 10 shuffles (seed 0), scores averaged.
   In Part A the groups are formed inside each fold, separately for test and training networks.
 * Per-target ANOVA (descriptive): one-way F of each feature across the candidates sharing a
   target; p-value from 9999 permutations of the candidate labels within the target (seed 0);
   counts significant after Benjamini-Hochberg (q = 0.05) and after Bonferroni (0.05 / tests).
Usage: python pairwise2d_analysis.py [--results pairwise2d_results] [--json out.json]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curve_inversion as ci  # noqa: E402  (ridge_fit, grouped_folds, cross_validate)

SETS = ("C1", "C4", "C16", "P")
INFO = ("S",) + tuple(f"S+{x}" for x in SETS)
LAYERS = [round(0.1 * n, 2) for n in range(1, 10)]
MIN_NETWORKS_A = 10
GAIN_ACC, GAIN_R2, NEGL = 0.20, 0.10, 0.05
KS_A, KS_B = (1, 3, 6), (1, 3, 6, 30)


# ----------------------------------------------------------------------------- data
def fill_curve(c):
    """log10 curve with empty shells filled; None if no shell is present."""
    x = np.array(LAYERS); y = np.array([np.nan if v is None else v for v in c], float)
    ok = np.isfinite(y) & (y > 0)
    if not ok.any():
        return None, 9
    ly = np.full(9, np.nan); ly[ok] = np.log10(y[ok])
    ly[~ok] = np.interp(x[~ok], x[ok], ly[ok])
    return ly, int((~ok).sum())


def load(results):
    rows = [json.loads(l) for l in open(os.path.join(results, "partial.jsonl")) if l.strip()]
    plan = json.load(open(os.path.join(HERE, "pairwise2d_plan.json")))["units"]
    by_id = {r["id"]: r for r in rows}
    missing = [u["id"] for u in plan if u["id"] not in by_id]
    for u in plan:                                    # Part A must have reproduced every stored network
        r = by_id.get(u["id"])
        if r is not None and u["part"] == "A" and (r["n"] != len(u["stored_R_eff_values"]) or r["reproduced_stored"] != r["n"]):
            raise SystemExit(f"{u['id']}: {r['n']} networks, {len(u['stored_R_eff_values'])} stored")
    nets = []                                          # one record per usable network
    for r in rows:
        for i in range(r["n"]):
            feats, filled, ok = {}, {}, True
            for s in SETS:
                ly, nf = fill_curve(r[s][i])
                if ly is None:
                    ok = False; break
                feats[s], filled[s] = ly, nf
            if not ok:
                continue
            nets.append({"unit": r["id"], "part": r["part"], "N": r["N"], "a": r["a"],
                         "R_target": r.get("R_target"), "logS": np.log10(r["R_eff_values"][i]),
                         "feats": feats, "filled": filled})
    return rows, nets, missing


def matrix(nets, info):
    S = np.array([[n["logS"]] for n in nets])
    if info == "S":
        return S
    return np.hstack([S, np.array([n["feats"][info[2:]] for n in nets])])


# ----------------------------------------------------------------------------- Part A
def lda_fit(Xtr, ytr, gamma=0.1):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Z = (Xtr - mu) / sd
    classes = np.unique(ytr)
    M = np.array([Z[ytr == c].mean(0) for c in classes])
    Rz = Z - M[np.searchsorted(classes, ytr)]
    Sw = Rz.T @ Rz / max(len(Z) - len(classes), 1)
    Sw = (1 - gamma) * Sw + gamma * np.mean(np.diag(Sw)) * np.eye(Sw.shape[0])
    W = np.linalg.solve(Sw, M.T)
    b = -0.5 * np.sum(M * W.T, axis=1)
    return lambda X: classes[np.argmax(((X - mu) / sd) @ W + b, axis=1)]


def _groups(F, y, mask, k, rng):
    X, Y = [], []
    for c in np.unique(y[mask]):
        idx = rng.permutation(np.where(mask & (y == c))[0])
        for g in range(len(idx) // k):
            X.append(F[idx[g * k:(g + 1) * k]].mean(0)); Y.append(c)
    return np.array(X), np.array(Y)


def classify(F, y, k, n_rep=10, n_folds=5, seed=0):
    rng = np.random.default_rng(seed); accs = []
    for _ in range(n_rep):
        fold = np.empty(len(y), int)
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0]); fold[idx] = np.arange(len(idx)) % n_folds
        correct = total = 0
        for f in range(n_folds):
            Xtr, ytr = _groups(F, y, fold != f, k, rng); Xte, yte = _groups(F, y, fold == f, k, rng)
            if len(yte) == 0 or len(np.unique(ytr)) < 2:
                continue
            correct += int((lda_fit(Xtr, ytr)(Xte) == yte).sum()); total += len(yte)
        accs.append(correct / total)
    return float(np.mean(accs))


def perm_anova_p(x, y, n_perm=9999, rng=None):
    classes = np.unique(y); oh = (y[:, None] == classes[None, :]).astype(float); nc = oh.sum(0)
    def ssb(xx):
        m = (xx @ oh) / nc; return ((m - x.mean()) ** 2 * nc).sum(-1)
    F0 = ssb(x)
    perms = np.array([rng.permutation(len(x)) for _ in range(n_perm)])
    return (1 + int((ssb(x[perms]) >= F0 - 1e-12).sum())) / (1 + n_perm)


def bh(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    kmax = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(m, bool); sig[o[:kmax]] = True
    return sig


def part_a(nets):
    A = [n for n in nets if n["part"] == "A"]
    count = {}
    for n in A:
        count[n["unit"]] = count.get(n["unit"], 0) + 1
    excluded = sorted(u for u, c in count.items() if c < MIN_NETWORKS_A)
    A = [n for n in A if count[n["unit"]] >= MIN_NETWORKS_A]
    targets = sorted({n["R_target"] for n in A})
    out = {"excluded_candidates": excluded, "targets": {}, "summary": {}}
    for t in targets:
        sub = [n for n in A if n["R_target"] == t]
        y = np.array([n["unit"] for n in sub]); C = len(np.unique(y))
        if C < 2:
            continue
        rec = {"candidates": {u: {"N": next(n["N"] for n in sub if n["unit"] == u),
                                  "a": next(n["a"] for n in sub if n["unit"] == u),
                                  "n": int((y == u).sum())} for u in np.unique(y)},
               "chance": 1.0 / C, "accuracy": {}, "filled_share": {}}
        for info in INFO:
            F = matrix(sub, info)
            rec["accuracy"][info] = {f"k={k}": classify(F, y, k) for k in KS_A}
        for s in SETS:
            rec["filled_share"][s] = float(np.mean([n["filled"][s] for n in sub]) / 9)
        rng = np.random.default_rng(0); pv = {}
        for info in ("S",) + SETS:
            cols = [matrix(sub, "S")[:, 0]] if info == "S" else list(np.array([n["feats"][info] for n in sub]).T)
            pv[info] = [perm_anova_p(np.asarray(c, float), y, rng=rng) for c in cols]
        rec["anova_p"] = pv
        out["targets"][str(t)] = rec
    # summary over targets
    allp = [(t, f, i, p) for t, r in out["targets"].items() for f, ps in r["anova_p"].items() for i, p in enumerate(ps)]
    sig_bh = bh([x[3] for x in allp]); bonf = 0.05 / len(allp)
    out["anova_tests"] = len(allp)
    out["anova_significant"] = {f: {"BH": int(sum(s for x, s in zip(allp, sig_bh) if x[1] == f)),
                                    "Bonferroni": int(sum(x[3] <= bonf for x in allp if x[1] == f)),
                                    "tests": int(sum(x[1] == f for x in allp))} for f in ("S",) + SETS}
    for info in INFO:
        for k in KS_A:
            ex = [r["accuracy"][info][f"k={k}"] - r["chance"] for r in out["targets"].values()]
            out["summary"].setdefault(info, {})[f"k={k}"] = {"mean_excess_over_chance": float(np.mean(ex)),
                                                             "targets_ge_0.20": int(sum(e >= GAIN_ACC for e in ex)),
                                                             "n_targets": len(ex)}
    return out


# ----------------------------------------------------------------------------- Part B
def regress(F, Y, G, cells, k, n_shuffles=10, seed=0):
    rng = np.random.default_rng(seed); r2s, rms = [], []
    for _ in range(n_shuffles if k > 1 else 1):
        Xs, Ys, Gs = [], [], []
        for c in cells:
            idx = rng.permutation(np.where(G == c)[0])
            for g in range(len(idx) // k):
                sel = idx[g * k:(g + 1) * k]; Xs.append(F[sel].mean(0)); Ys.append(Y[sel[0]]); Gs.append(c)
        rmse, r2, _ = ci.cross_validate(np.array(Xs), np.array(Ys), np.array(Gs),
                                        lambda a, b: ci.ridge_fit(a, b, 1.0), 10)
        r2s.append(r2); rms.append(rmse)
    r2, rmse = np.mean(r2s, 0), np.mean(rms, 0)
    return {"r2_logN": float(r2[0]), "r2_loga": float(r2[1]),
            "factor_N": float(10 ** rmse[0]), "factor_a": float(10 ** rmse[1])}


def part_b(nets):
    B = [n for n in nets if n["part"] == "B"]
    G = np.array([n["unit"] for n in B]); Y = np.log10(np.array([[n["N"], n["a"]] for n in B], float))
    counts = {c: int((G == c).sum()) for c in np.unique(G)}
    fixed = sorted(c for c, m in counts.items() if m >= 30)
    out = {"cells_total": len(counts), "cells_fixed": len(fixed), "networks_fixed": int(sum(counts[c] for c in fixed)),
           "networks_all": len(B), "results": {}, "all_networks_k1": {},
           "filled_share": {s: float(np.mean([n["filled"][s] for n in B]) / 9) for s in SETS}}
    for info in INFO:
        F = matrix(B, info)
        out["results"][info] = {f"k={k}": regress(F, Y, G, fixed, k) for k in KS_B}
        out["all_networks_k1"][info] = regress(F, Y, G, sorted(counts), 1)
    return out


# ----------------------------------------------------------------------------- verdict
def verdicts(a, b):
    v = {}
    r2S = b["results"]["S"]["k=1"]["r2_loga"]
    for info in INFO[1:]:
        ex = a["summary"][info]["k=1"]["mean_excess_over_chance"]
        d = b["results"][info]["k=1"]["r2_loga"] - r2S
        if ex >= GAIN_ACC or d >= GAIN_R2:
            word = "GAIN"
        elif abs(ex) <= NEGL and abs(d) <= NEGL:
            word = "NEGLIGIBLE"
        else:
            word = "NEITHER"
        v[info] = {"A_excess_over_chance": ex, "A_pass": ex >= GAIN_ACC, "B_delta_r2_loga": d,
                   "B_pass": d >= GAIN_R2, "verdict": word}
    s = a["summary"]["S"]["k=1"]["mean_excess_over_chance"]
    v["S_baseline"] = {"A_excess_over_chance": s, "note": "S alone; if this also clears 0.20, the Part A excess "
                       "of an information set is not attributable to the extra information"}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "pairwise2d_results"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows, nets, missing = load(args.results)
    nA = sum(r["part"] == "A" for r in rows); nB = len(rows) - nA
    print(f"{len(rows)} units ({nA} Part A, {nB} Part B); {len(nets)} usable networks; missing units: {missing or 'none'}")
    print("Part A: every regenerated terminal R_eff reproduced validation_results to 1e-9 (checked by the task)")
    a = part_a(nets); b = part_b(nets); v = verdicts(a, b)

    print(f"\n== Part A: classification among candidates sharing a target (LDA, stratified 5-fold x 10)")
    print(f"   excluded (< {MIN_NETWORKS_A} networks): {a['excluded_candidates']}")
    hdr = "   target  C chance  " + "  ".join(f"{i:>6s}" for i in INFO) + "   [accuracy, k=1]"
    print(hdr)
    for t, r in a["targets"].items():
        print(f"   {float(t):6g} {len(r['candidates'])} {r['chance']:6.2f}  " + "  ".join(f"{r['accuracy'][i]['k=1']:6.2f}" for i in INFO))
    for k in KS_A:
        print(f"   mean (accuracy - chance), k={k}: " + "  ".join(f"{i} {a['summary'][i][f'k={k}']['mean_excess_over_chance']:+.3f}" for i in INFO))
    print(f"   ANOVA, {a['anova_tests']} tests: " + "; ".join(f"{f}: BH {x['BH']}/{x['tests']}, Bonf {x['Bonferroni']}" for f, x in a["anova_significant"].items()))

    print(f"\n== Part B: ridge, 10-fold grouped by cell; {b['cells_fixed']} of {b['cells_total']} cells reached 30 networks ({b['networks_fixed']} networks)")
    print("   set        k    R2 logN  R2 loga   factor N  factor a")
    for info in INFO:
        for k in KS_B:
            x = b["results"][info][f"k={k}"]
            print(f"   {info:8s} {k:3d}   {x['r2_logN']:6.3f}   {x['r2_loga']:6.3f}    {x['factor_N']:6.3f}    {x['factor_a']:6.3f}")
    print("   all cells, k=1: " + "  ".join(f"{i} R2 loga {b['all_networks_k1'][i]['r2_loga']:.3f}" for i in INFO))
    print("   share of filled shells (Part B): " + ", ".join(f"{s} {x:.1%}" for s, x in b["filled_share"].items()))

    print("\n== Verdict against the pre-set criteria (k = 1)")
    print(f"   S alone, Part A mean excess over chance: {v['S_baseline']['A_excess_over_chance']:+.3f}")
    for info in INFO[1:]:
        x = v[info]
        print(f"   {info:6s}  A: {x['A_excess_over_chance']:+.3f} (>= 0.20? {x['A_pass']})   "
              f"B: dR2 log a {x['B_delta_r2_loga']:+.3f} (>= 0.10? {x['B_pass']})   -> {x['verdict']}")
    out = {"part_A": a, "part_B": b, "verdicts": v, "missing_units": missing}
    path = args.json or os.path.join(args.results, "analysis.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
