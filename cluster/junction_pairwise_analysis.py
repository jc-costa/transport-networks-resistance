"""
Analysis of run_junction_pairwise_task.py (paper item A).

Question: at (nearly) the same terminal resistance S, do the pairwise
resistances tell apart films that differ in the junction ratio rho, or in the
stick length a?

Method (the logic of the E7 test, applied to the new three-parameter family).
Every (cell, rho, network) is one sample with features log10 of S, of the 9
shell values of C1 / C4 / C16, or of the 9 values of P. Samples are grouped into
narrow bins of log10 S (width BIN decades). In each bin the classes are the
values of rho present with at least MIN_PER_CLASS samples (task "rho"), or the
stick lengths a (task "a"); a bin is used when it has at least two classes.
Within a bin a regularized linear discriminant is cross-validated (5 folds,
stratified, fixed seed). Accuracy above chance is averaged over bins, weighted
by the number of samples. Baseline: S alone, the residual of S within the bin.

Usage: python junction_pairwise_analysis.py <partial.jsonl> [--bin 0.1] [--out report.json]
"""
import sys, json
import numpy as np

BIN, MIN_PER_CLASS, FOLDS, SEED = 0.10, 8, 5, 0


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        for k, net in enumerate(r["networks"]):
            for b in net.get("by_cond") or net["by_rho"]:
                if any(v is None for key in ("C1", "C4", "C16", "P") for v in b[key]):
                    continue
                if any(v <= 0 or not np.isfinite(v) for key in ("C1", "C4", "C16", "P") for v in b[key]):
                    raise ValueError(f"{r['id']} network {k}: non-positive or non-finite pairwise value; "
                                     "the data come from a version of the task script that did not handle "
                                     "disconnected graphs")
                rows.append(dict(cell=r["id"], variant=b.get("variant", "uniform"), N=r["N"], a=r["a"], x=r["x"], net=k, rho=b["rho"],
                                 S=b["S"], C1=b["C1"], C4=b["C4"], C16=b["C16"], P=b["P"]))
    return rows


def features(rows, name):
    S = np.log10([r["S"] for r in rows])[:, None]
    if name == "S":
        return S
    return np.hstack([S, np.log10(np.array([r[name] for r in rows]))])


def lda_cv(X, y, rng, shrink=0.1):
    """Stratified 5-fold cross-validated accuracy of a shrunk-covariance LDA."""
    classes = np.unique(y)
    folds = np.empty(len(y), int)
    for c in classes:
        idx = rng.permutation(np.where(y == c)[0])
        folds[idx] = np.arange(len(idx)) % FOLDS
    correct = 0
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        mu = np.array([X[tr & (y == c)].mean(0) for c in classes])
        Xc = X[tr] - mu[np.searchsorted(classes, y[tr])]
        cov = Xc.T @ Xc / max(len(Xc) - len(classes), 1)
        cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        W = np.linalg.solve(cov, mu.T)                      # d x k
        score = X[te] @ W - 0.5 * np.sum(mu.T * W, axis=0)
        correct += int((classes[np.argmax(score, 1)] == y[te]).sum())
    return correct / len(y)


def noisy(rows, sigma, rng):
    """Independent log-normal noise (sigma = relative error of one measured value) on S
    and on every shell value; the measurer bins and classifies on the noisy values."""
    if sigma == 0:
        return rows
    f = lambda v: np.asarray(v) * np.exp(sigma * rng.standard_normal(np.shape(v)))
    return [dict(r, S=float(f(r["S"])), C1=f(r["C1"]), C4=f(r["C4"]), C16=f(r["C16"]), P=f(r["P"]))
            for r in rows]


def run(rows, task, bin_w):
    key = {"rho": "rho", "a": "a", "x": "x"}[task]
    logS = np.log10([r["S"] for r in rows])
    b = np.floor(logS / bin_w).astype(int)
    out = {n: [] for n in ("S", "C1", "C4", "C16", "P")}
    used = []
    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        lab = np.array([rows[i][key] for i in idx])
        keep = [c for c in np.unique(lab) if (lab == c).sum() >= MIN_PER_CLASS]
        if len(keep) < 2:
            continue
        sel = idx[np.isin(lab, keep)]
        y = np.array([rows[i][key] for i in sel])
        chance = max(np.mean(y == c) for c in np.unique(y))      # majority rate, conservative
        used.append((len(sel), len(np.unique(y)), chance))
        for name in out:
            rng = np.random.default_rng(SEED)
            out[name].append((len(sel), lda_cv(features([rows[i] for i in sel], name), y, rng) - chance))
    n_used = sum(u[0] for u in used)
    res = {"bins": len(used), "samples": n_used,
           "mean_classes": float(np.mean([u[1] for u in used])) if used else None,
           "mean_majority_rate": float(np.average([u[2] for u in used], weights=[u[0] for u in used])) if used else None}
    for name, v in out.items():
        res[name] = float(np.average([d for _, d in v], weights=[w for w, _ in v])) if v else None
    return res


def sweep(rows, bin_w, sigmas, draws, tasks):
    """tasks: list of (label, task, filter); filter = (field, value) or None."""
    table = {}
    for sigma in sigmas:
        for label, task, flt in tasks:
            accs = {n: [] for n in ("S", "C1", "C4", "C16", "P")}
            for d in range(draws if sigma else 1):
                rng = np.random.default_rng(1000 + d)
                sub = rows if flt is None else [r for r in rows if r[flt[0]] == flt[1]]
                res = run(noisy(sub, sigma, rng), task, bin_w)
                for n in accs:
                    if res[n] is not None:
                        accs[n].append(res[n])
            table[(sigma, label)] = {n: (float(np.mean(v)) if v else None) for n, v in accs.items()}
    return table


if __name__ == "__main__" and "--noise" in sys.argv:
    path = sys.argv[1]
    rows = load(path)
    bin_w = float(sys.argv[sys.argv.index("--bin") + 1]) if "--bin" in sys.argv else BIN
    sigmas = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    avals = sorted(set(r["a"] for r in rows)); rvals = sorted(set(r["rho"] for r in rows))
    tasks = [("rho (all)", "rho", None), ("a (all)", "a", None), ("x=Na2 (all)", "x", None),
             ("x within rho=0", "x", ("rho", 0.0)), ("a within rho=0", "a", ("rho", 0.0))]
    table = sweep(rows, bin_w, sigmas, 5, tasks)
    print("accuracy above majority rate, features S+P (S+C16 in brackets)\n")
    print(f"{'task':18s}" + "".join(f"{'s=%.2f' % s:>16s}" for s in sigmas))
    for label, _, _ in tasks:
        line = f"{label:18s}"
        for s in sigmas:
            t = table[(s, label)]
            fmt = lambda v: "  n/a" if v is None else f"{v:+.2f}"
            line += f"{fmt(t['P'])} ({fmt(t['C16'])})".rjust(16)
        print(line)
    print("\nS alone: " + ", ".join(f"s={s:.2f}: {table[(s, 'rho (all)')]['S']:+.2f}" for s in sigmas))
    sys.exit(0)

if __name__ == "__main__":
    path = sys.argv[1]
    bin_w = float(sys.argv[sys.argv.index("--bin") + 1]) if "--bin" in sys.argv else BIN
    rows = load(path)
    print(f"{len(rows)} samples, {len(set(r['cell'] for r in rows))} cells, rho = {sorted(set(r['rho'] for r in rows))}")
    report = {}
    for task in ("rho", "a"):
        res = run(rows, task, bin_w)
        report[task] = res
        print(f"\ntask {task}: {res['bins']} bins, {res['samples']} samples, "
              f"{res['mean_classes']:.1f} classes/bin, majority rate {res['mean_majority_rate']:.2f}")
        for name in ("S", "C1", "C4", "C16", "P"):
            print(f"   {name:4s} accuracy above majority rate: {res[name]:+.3f}")
    if "--out" in sys.argv:
        json.dump(report, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
