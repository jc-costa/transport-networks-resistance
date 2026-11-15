"""
EXPLORATORY (post hoc, revision plan item 3.2), on the pre-registered follow-up run (E2b) joined to E2:

 1. Paired intervals for the low-contact result. Two tips with a contact of 0.1 times the terminal resistance of the
    film, 2187 attempts, against the all-pairs profile: RMSE, median factor and 90th-percentile factor, and their
    differences. The geometries are resampled with all their networks and films (the unit of generalization), 2000
    times, keeping the predictions of the protocols together. This is not the registered bootstrap, which resamples
    networks inside fixed geometries.
 2. A contact that is not what the estimator assumes. The regression is fitted on films whose tip contact is
    kappa_train times their terminal resistance and applied to films whose contact is kappa_test (calibration error),
    and a case where the contact varies from film to film (log-uniform over a factor 10, 0.05 to 0.5) in training and
    in test, unknown to the estimator.

Everything else as in junction_strategies_analysis.py (margin 10 %, base landing, 5 % noise, whole geometries held out,
the same folds). Usage: python junction_strategies_contact_robust.py <E2 partial.jsonl> <E2b partial.jsonl> [--out summary.json]
"""
import sys, json
import numpy as np
import junction_strategies_more_analysis as ma
import junction_strategies_analysis as sa
import junction_pairwise_review_checks as rc

BOOT = 2000
films = ma.load_join(sys.argv[1], sys.argv[2]); D = ma.Data(films)
orig = sa.Data([dict(cell=f["cell"], net=f["net"], a=f["a"], N=f["N"], rho=f["rho"], S=f["S"], P=f["P"], uc=f["uc"], ua=f["ua"], uN=f["uN"], strat=None) for f in films])
rt, ra = sa.evaluate(orig, "terminal", ma.MARGIN), sa.evaluate(orig, "allpairs", ma.MARGIN)
y = D.y
fac = lambda e: float(10 ** np.median(np.abs(e)))
p90 = lambda e: float(10 ** np.percentile(np.abs(e), 90))
rm = lambda e: float(np.sqrt(np.mean(e ** 2)))


def features(kind, design, kappa_rel):
    """Feature matrix with a per-film contact kappa_rel (array) times the terminal resistance of the film."""
    rng = np.random.default_rng(ma.SEED)
    c = 1 + ma.MARGIN * D.uc
    Sobs = c * D.S * np.exp(ma.NOISE * rng.standard_normal(D.n))
    prior = np.column_stack([np.log10(D.a * (1 + ma.MARGIN * D.ua)), np.log10(D.N * (1 + ma.MARGIN * D.uN))])
    val, cnt = D.reading(kind, "random", design, "mean")
    tip = {"one": 1.0, "two": 2.0, "four": 0.0}[kind]
    contact = tip * kappa_rel * (c * D.S)
    v = (c[:, None] * val + contact[:, None]) * np.exp(ma.NOISE / np.sqrt(np.maximum(cnt, 1)) * rng.standard_normal(val.shape))
    lv = np.where(cnt > 0, np.log10(np.where(v > 0, v, np.nan)), np.nan)
    return np.hstack([np.log10(Sobs)[:, None], sa.impute(lv, np.log10(Sobs)), prior])


def cv_mismatch(kind, design, k_train, k_test):
    Xtr, Xte = features(kind, design, k_train), features(kind, design, k_test)
    p = np.zeros(D.n)
    for f in range(5):
        tr, te = D.fold != f, D.fold == f
        p[te] = rc.fit_predict(Xtr[tr], y[tr], Xte[te])
    return p


out = {"terminal": dict(rmse=rt["rmse"], median_factor=rt["median_factor"]), "allpairs": dict(rmse=ra["rmse"], median_factor=ra["median_factor"], p90_factor=ra["p90_factor"])}
e_ap = ra["pred"]["log10 rho"] - y
e_tm = rt["pred"]["log10 rho"] - y

# 1. paired geometry bootstrap
cells = sorted(set(D.cell)); idx = {c: np.where(D.cell == c)[0] for c in cells}
rng = np.random.default_rng(1)
draws = [np.concatenate([idx[c] for c in rng.choice(cells, len(cells))]) for _ in range(BOOT)]
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
print(f"films {D.n}, geometries {len(cells)}; all pairs RMSE {rm(e_ap):.3f}, median {fac(e_ap):.3f}, P90 {p90(e_ap):.2f}; terminal {rm(e_tm):.3f}")
out["paired"] = {}
for kd, design, att, kr in (("two", "9x243", 2187, 0.1), ("two", "9x27", 243, 0.1), ("two", "9x243", 2187, 0.5), ("two", "9x243", 2187, 1.0)):
    p, _ = D.predict(kd, "random", design, kappa_rel=kr)
    e = p - y
    row = {}
    for name, fn in (("rmse", rm), ("median_factor", fac), ("p90_factor", p90)):
        d = np.array([fn(e[ix]) - fn(e_ap[ix]) for ix in draws])
        row[name] = dict(protocol=fn(e), allpairs=fn(e_ap), diff=fn(e) - fn(e_ap), ci=ci(d))
    out["paired"][f"{kd}|{att}|{kr}"] = row
    print(f"{kd} {att} attempts, contact {kr} x film: " + "; ".join(f"{n} {r['protocol']:.3f} vs {r['allpairs']:.3f} (diff {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}])" for n, r in row.items()))

# 2. contact not what the estimator assumes
print("\nfitted with contact 0.1 x film, tested with the contact below (two tips, 2187 attempts): RMSE (median factor)")
out["mismatch"] = {}
for design, att in (("9x243", 2187), ("9x27", 243)):
    for kt in (0.05, 0.1, 0.2, 0.5, 1.0):
        e = cv_mismatch("two", design, 0.1, kt) - y
        out["mismatch"][f"{att}|train0.1|test{kt}"] = dict(rmse=rm(e), median_factor=fac(e))
        print(f"  {att:5d} attempts, test contact {kt:4.2f}: {rm(e):.3f} ({fac(e):.2f})")
rngk = np.random.default_rng(2)
kvar = 10 ** rngk.uniform(np.log10(0.05), np.log10(0.5), D.n)
for design, att in (("9x243", 2187), ("9x27", 243)):
    e = cv_mismatch("two", design, kvar, kvar) - y
    out["mismatch"][f"{att}|varying"] = dict(rmse=rm(e), median_factor=fac(e))
    print(f"  {att:5d} attempts, contact varying 0.05 to 0.5 between films (unknown): {rm(e):.3f} ({fac(e):.2f})")
if "--out" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
