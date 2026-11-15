"""
Post hoc (revision plan, item 3.1): how much signal reaches the estimator in the strategies run (E2).

For every protocol and design the stored records give, per film, the number of successful readings at each of the
positions. A film is a SESSION with no signal when no position has a reading; the estimator then repeats the terminal
resistance (the registered imputation, junction_strategies_analysis.impute). This script reports, on the same films,
folds, margin, contact and noise as the registered analysis:

  * the fraction of attempts that yield an accepted reading (denominator: attempts of all films of the common cohort),
  * the fraction of sessions with no reading at all, and the fraction of position classes that stayed empty,
  * RMSE, median and 90th-percentile factor error over all films, over films with signal and over films without,
    each against the terminal-only predictor on the same films,
  * the same with missing-data indicators added to the features (the number of positions with a reading).

Nothing here is pre-registered. The registered results are the reference and are not changed.

Usage: python junction_strategies_signal.py <partial.jsonl> [--out summary.json]
"""
import sys, json
import numpy as np
import junction_strategies_analysis as sa

DESIGN_LIST = ("9x3", "9x27")


def factor_stats(e):
    a = np.abs(e)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "median_factor": float(10 ** np.median(a)), "p90_factor": float(10 ** np.percentile(a, 90))}


def features_with_indicators(D, kind, design):
    X, T, _ = D.features("readings", sa.MARGIN, np.random.default_rng(sa.SEED), kind, sa.CFG, design)
    _, cnt = D.reading(kind, sa.CFG, design)
    ind = np.column_stack([(cnt > 0).sum(1) / cnt.shape[1], (cnt.sum(1) == 0).astype(float)])
    ind = ind[:, ind.std(0) > 0]                # a constant indicator carries nothing and breaks the standardization
    return np.hstack([X, ind]), T


def main():
    films = sa.load(sys.argv[1])
    films = [f for f in films if all(v is not None for v in f["P"])]
    D = sa.Data(films)
    print(f"{D.n} films, {D.nnet} networks, {len(set(D.cell))} geometries (common cohort)")
    term = sa.evaluate(D, "terminal", sa.MARGIN)
    e_term = term["pred"]["log10 rho"] - D.y
    out = {"n_films": int(D.n), "terminal": factor_stats(e_term)}
    for design in DESIGN_LIST:
        att = sa.ATTEMPTS[design]
        for kind in sa.KINDS:
            res = sa.evaluate(D, "readings", sa.MARGIN, kind, sa.CFG, design)
            e = res["pred"]["log10 rho"] - D.y
            _, cnt = D.reading(kind, sa.CFG, design)
            none = cnt.sum(1) == 0
            empty = float((cnt == 0).mean())
            Xi, T = features_with_indicators(D, kind, design)
            ei = D.cv(Xi, T["log10 rho"]) - D.y
            row = {"attempts": att, "accepted_share": float(cnt.sum() / (D.n * att)),
                   "sessions_without_reading": float(none.mean()), "empty_position_classes": empty,
                   "all": factor_stats(e), "terminal_same_films": factor_stats(e_term),
                   "with_indicators": factor_stats(ei)}
            if none.any():
                row["without_signal"] = {**factor_stats(e[none]), "terminal": factor_stats(e_term[none]), "n": int(none.sum())}
            row["with_signal"] = {**factor_stats(e[~none]), "terminal": factor_stats(e_term[~none]), "n": int((~none).sum())}
            out[f"{kind}|{design}"] = row
            print(f"{kind:4s} {att:4d}: accepted {row['accepted_share']:.3f}, sessions without reading {row['sessions_without_reading']:.4f}, "
                  f"empty classes {empty:.3f}; RMSE all {row['all']['rmse']:.3f} (terminal {row['terminal_same_films']['rmse']:.3f}), "
                  f"with signal {row['with_signal']['rmse']:.3f} (terminal {row['with_signal']['terminal']['rmse']:.3f}), "
                  f"with indicators {row['with_indicators']['rmse']:.3f}, P90 {row['all']['p90_factor']:.2f}")
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
