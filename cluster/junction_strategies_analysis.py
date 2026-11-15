"""
Pre-registered analysis of the measurement-strategies run E2 (PREREGISTRATION_JUNCTION_STRATEGIES.md).
Everything that decides the outcome is fixed in that file and in the constants below; nothing here may change
after the run.

The scenario (the advisor's): the wire resistance per length R0, the stick length a and the number of sticks N
are known only within a margin (a relative error drawn uniformly in +-margin, independently), and the junction
resistance is completely unknown (rho log-uniform, 0.03 to 100; equal junctions). For every film and every
strategy the analysis builds what a user would have: the terminal resistance, the readings of the strategy, and
the nominal a and N, and trains a quadratic ridge regression (the registered penalty) to estimate
log10 rho and the three errors of the nominal values, with WHOLE GEOMETRIES held out (5 folds, seed 0).

Added here, to the stored noise-free readings: the error of R0 (c), the tip contact resistance (kappa wire
resistances R0 l per tip contact, one tip contact for the one-tip strategy and two for the two-tip one, none
for four points), and multiplicative reading noise of 5 % (a shell value of r readings has noise 5 %/sqrt(r)).
A shell without a successful reading is imputed by linear interpolation of the others in the logarithm.

Usage: python junction_strategies_analysis.py <partial.jsonl> [--boot 2000] [--out summary.json]
"""
import sys, json
import numpy as np
import junction_pairwise_review_checks as rc

SEED, BOOT = 0, 2000
MARGIN, KAPPA, CFG, NOISE = 0.10, 1.0, "base", 0.05
MARGINS, KAPPAS = (0.05, 0.10, 0.20, 0.50), (0.0, 1.0, 10.0)
CFGS = ("base", "f01", "f04", "p03")
KINDS = ("one", "two", "four")
DESIGNS = ("3x1", "3x3", "9x1", "9x3", "9x9", "9x27")
ATTEMPTS = {"3x1": 3, "3x3": 9, "9x1": 9, "9x3": 27, "9x9": 81, "9x27": 243}
SWEEP_DESIGNS = ("9x3", "9x9")
RMSE_TARGET = 0.35                      # the error of the transfer run, the criterion for the minimal budget
H1_MARGIN, H3_LO, H3_HI = 0.05, 0.90, 0.95
REGIMES = (("wire-limited (rho < 0.1)", 0.0, 0.1), ("intermediate (0.1 to 10)", 0.1, 10.0),
           ("junction-limited (rho > 10)", 10.0, 1e9))
DESIGN_SHELLS = {"3x1": 3, "3x3": 3, "9x1": 9, "9x3": 9, "9x9": 9, "9x27": 9}


def load(path):
    films, geo = [], {}
    for line in open(path):
        r = json.loads(line)
        for n in r["networks"]:
            for f in n["films"]:
                films.append(dict(cell=r["id"], net=n["net"], a=r["a"], N=r["N"], rho=f["rho"], S=f["S"], P=f["P"],
                                  uc=f["u_c"], ua=f["u_a"], uN=f["u_N"], strat=f["strat"]))
    return films


def impute(logv, fallback):
    """Fill missing shells (NaN) by linear interpolation in the logarithm; all missing -> the fallback value."""
    out = logv.copy()
    for i in np.where(np.isnan(out).any(1))[0]:
        ok = ~np.isnan(out[i])
        out[i] = np.interp(np.arange(out.shape[1]), np.where(ok)[0], out[i][ok]) if ok.any() else fallback[i]
    return out


class Data:
    def __init__(self, films):
        self.n = len(films)
        self.rho = np.array([f["rho"] for f in films]); self.S = np.array([f["S"] for f in films])
        self.a = np.array([f["a"] for f in films]); self.N = np.array([f["N"] for f in films], float)
        self.uc = np.array([f["uc"] for f in films]); self.ua = np.array([f["ua"] for f in films]); self.uN = np.array([f["uN"] for f in films])
        self.P = np.array([[np.nan if v is None else v for v in f["P"]] for f in films])
        self.cell = np.array([f["cell"] for f in films])
        nid = {}; self.net = np.array([nid.setdefault((f["cell"], f["net"]), len(nid)) for f in films]); self.nnet = len(nid)
        self.films = films
        fm = rc.folds_for(list(self.cell)); self.fold = np.array([fm[c] for c in self.cell])
        self.y = np.log10(self.rho)
        self._cache = {}

    def reading(self, kind, cfg, design):
        key = (kind, cfg, design)
        if key not in self._cache:
            m = DESIGN_SHELLS[design]
            mean = np.full((self.n, m), np.nan); cnt = np.zeros((self.n, m))
            for i, f in enumerate(self.films):
                d = f["strat"][kind][cfg][design]
                mean[i] = [np.nan if v is None else v for v in d["mean"]]; cnt[i] = d["cnt"]
            self._cache[key] = (mean, cnt)
        return self._cache[key]

    def features(self, strategy, margin, rng, kind=None, cfg=CFG, design=None, kappa=KAPPA):
        c = 1 + margin * self.uc
        Sobs = c * self.S * np.exp(NOISE * rng.standard_normal(self.n))
        prior = np.column_stack([np.log10(self.a * (1 + margin * self.ua)), np.log10(self.N * (1 + margin * self.uN))])
        cols = [np.log10(Sobs)[:, None]]
        succ = None
        if strategy == "allpairs":
            cols.append(np.log10(c[:, None] * self.P))
        elif strategy == "readings":
            mean, cnt = self.reading(kind, cfg, design)
            tip = {"one": 1.0, "two": 2.0, "four": 0.0}[kind]
            val = c[:, None] * mean + tip * kappa * self.a[:, None]            # R0 l = a in units of R0 L
            val = val * np.exp(NOISE / np.sqrt(np.maximum(cnt, 1)) * rng.standard_normal(mean.shape))
            lv = np.where(cnt > 0, np.log10(np.where(val > 0, val, np.nan)), np.nan)
            cols.append(impute(lv, np.log10(Sobs)))
            succ = float(cnt.sum() / (self.n * ATTEMPTS[design]))
        X = np.hstack(cols + [prior])
        targets = {"log10 rho": self.y, "log10 R0": np.log10(c),
                   "log10 a": -np.log10(1 + margin * self.ua), "log10 N": -np.log10(1 + margin * self.uN)}
        return X, targets, succ

    def cv(self, X, y):
        p = np.zeros(self.n)
        for f in range(5):
            tr, te = self.fold != f, self.fold == f
            p[te] = rc.fit_predict(X[tr], y[tr], X[te])
        return p


def rmse(e, w=None):
    return float(np.sqrt(np.mean(e ** 2))) if w is None else float(np.sqrt(np.sum(w * e ** 2) / w.sum()))


def evaluate(D, strategy, margin, kind=None, cfg=CFG, design=None, kappa=KAPPA):
    rng = np.random.default_rng(SEED)
    X, T, succ = D.features(strategy, margin, rng, kind, cfg, design, kappa)
    pred = {k: D.cv(X, v) for k, v in T.items()}
    out = {"success": succ, "pred": pred, "targets": T}
    e = pred["log10 rho"] - T["log10 rho"]
    out["rmse"] = rmse(e); out["median_factor"] = float(10 ** np.median(np.abs(e))); out["p90_factor"] = float(10 ** np.percentile(np.abs(e), 90))
    out["by_regime"] = {lab: rmse(e[(D.rho >= lo) & (D.rho < hi)]) for lab, lo, hi in REGIMES}
    out["narrowing"] = {}
    for k in ("log10 R0", "log10 a", "log10 N"):
        prior = float(np.std(T[k])); out["narrowing"][k] = {
            "all": rmse(pred[k] - T[k]) / prior,
            **{lab: rmse((pred[k] - T[k])[(D.rho >= lo) & (D.rho < hi)]) / float(np.std(T[k][(D.rho >= lo) & (D.rho < hi)])) for lab, lo, hi in REGIMES}}
    return out


def boot_weights(D, boot, rng):
    net_cell = {}
    for i in range(D.n):
        net_cell.setdefault(D.cell[i], set()).add(D.net[i])
    net_cell = {c: sorted(s) for c, s in net_cell.items()}
    W = []
    for _ in range(boot):
        w = np.zeros(D.nnet)
        for ids in net_cell.values():
            np.add.at(w, rng.choice(ids, len(ids)), 1)
        W.append(w[D.net])
    return np.array(W)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    path = sys.argv[1]
    boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else BOOT
    films = load(path)
    bad = [f for f in films if any(v is not None and (not np.isfinite(v) or v <= 0) for v in f["P"])]
    if bad:
        raise ValueError(f"{len(bad)} films have a non-positive or non-finite all-pairs value (first: {bad[0]['cell']} "
                         f"network {bad[0]['net']}); the run is not analysed until the fault is found")
    films = [f for f in films if all(v is not None for v in f["P"])]
    D = Data(films)
    print(f"{D.n} films from {D.nnet} networks in {len(set(D.cell))} geometries; margin {MARGIN:.0%}, kappa {KAPPA:g}, noise {NOISE:.0%}")
    out = {"main": {}, "sweeps": {}}

    ref = {"terminal": evaluate(D, "terminal", MARGIN), "allpairs": evaluate(D, "allpairs", MARGIN)}
    res = {}
    for kd in KINDS:
        for dz in DESIGNS:
            res[(kd, dz)] = evaluate(D, "readings", MARGIN, kd, CFG, dz)
    print("\nMain case: RMSE of log10 rho (median factor error) [reading success rate]")
    print(f"  terminal only            {ref['terminal']['rmse']:.3f}  ({ref['terminal']['median_factor']:.2f})")
    print(f"  all pairs (reference)    {ref['allpairs']['rmse']:.3f}  ({ref['allpairs']['median_factor']:.2f})")
    print(f"  {'attempts':>9s}  " + "  ".join(f"{kd:>20s}" for kd in KINDS))
    for dz in DESIGNS:
        print(f"  {ATTEMPTS[dz]:9d}  " + "  ".join(f"{res[(kd, dz)]['rmse']:.3f} ({res[(kd, dz)]['median_factor']:.2f}) [{res[(kd, dz)]['success']:.2f}]".rjust(20) for kd in KINDS))
    for kd in KINDS:
        reach = [ATTEMPTS[dz] for dz in DESIGNS if res[(kd, dz)]["rmse"] <= RMSE_TARGET]
        out["main"][kd] = {dz: {k: res[(kd, dz)][k] for k in ("rmse", "median_factor", "p90_factor", "success", "by_regime", "narrowing")} for dz in DESIGNS}
        out["main"][kd]["min_attempts_for_target"] = min(reach) if reach else None
        print(f"  {kd}: smallest design with RMSE <= {RMSE_TARGET}: {min(reach) if reach else 'none'} attempts")
    out["main"]["terminal"] = {k: ref["terminal"][k] for k in ("rmse", "median_factor", "p90_factor", "by_regime", "narrowing")}
    out["main"]["allpairs"] = {k: ref["allpairs"][k] for k in ("rmse", "median_factor", "p90_factor", "by_regime", "narrowing")}

    print("\nBy regime of the true rho, RMSE of log10 rho: terminal / one / two / four at 9x9, all pairs")
    for lab, _, _ in REGIMES:
        print(f"  {lab:28s} {ref['terminal']['by_regime'][lab]:.3f} / " + " / ".join(f"{res[(kd, '9x9')]['by_regime'][lab]:.3f}" for kd in KINDS) + f" / {ref['allpairs']['by_regime'][lab]:.3f}")
    print("\nHow much the readings narrow the margins (RMSE after / prior spread; below 1 is narrowing), all pairs and 9x9 readings")
    for k in ("log10 R0", "log10 a", "log10 N"):
        print(f"  {k:9s} all pairs: " + "  ".join(f"{lab.split(' ')[0]} {ref['allpairs']['narrowing'][k][lab]:.2f}" for lab in ["all"] + [r[0] for r in REGIMES])
              + " | four 9x9: " + "  ".join(f"{lab.split(' ')[0]} {res[('four', '9x9')]['narrowing'][k][lab]:.2f}" for lab in ["all"] + [r[0] for r in REGIMES]))

    # sweeps, one factor at a time, at two designs
    print("\nSweeps (RMSE of log10 rho), one factor at a time")
    for dz in SWEEP_DESIGNS:
        for kd in KINDS:
            row = {}
            for cfg in CFGS:
                row[f"cfg={cfg}"] = res[(kd, dz)]["rmse"] if cfg == CFG else evaluate(D, "readings", MARGIN, kd, cfg, dz)["rmse"]
            for kp in KAPPAS:
                row[f"kappa={kp:g}"] = res[(kd, dz)]["rmse"] if kp == KAPPA else evaluate(D, "readings", MARGIN, kd, CFG, dz, kp)["rmse"]
            for mg in MARGINS:
                row[f"margin={mg:g}"] = res[(kd, dz)]["rmse"] if mg == MARGIN else evaluate(D, "readings", mg, kd, CFG, dz)["rmse"]
            out["sweeps"][f"{kd}|{dz}"] = row
            print(f"  {kd:4s} {dz:4s} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    out["sweeps"]["terminal_by_margin"] = {f"{mg:g}": evaluate(D, "terminal", mg)["rmse"] for mg in MARGINS}
    out["sweeps"]["allpairs_by_margin"] = {f"{mg:g}": evaluate(D, "allpairs", mg)["rmse"] for mg in MARGINS}
    print("  terminal only by margin:", {k: round(v, 3) for k, v in out["sweeps"]["terminal_by_margin"].items()})
    print("  all pairs by margin:    ", {k: round(v, 3) for k, v in out["sweeps"]["allpairs_by_margin"].items()})

    # hypotheses
    rng = np.random.default_rng(SEED)
    W = boot_weights(D, boot, rng)
    e_term = ref["terminal"]["pred"]["log10 rho"] - D.y
    h = {}
    print("\nHypotheses (main case: margin 10 %, kappa 1, base landing, design 9x3 = 27 attempts unless stated)")
    for kd in KINDS:
        e_k = res[(kd, "9x3")]["pred"]["log10 rho"] - D.y
        d = np.array([rmse(e_term, w) - rmse(e_k, w) for w in W])
        pt = rmse(e_term) - rmse(e_k); lo = ci(d)
        h[f"H1_{kd}"] = dict(gain=pt, ci=lo, confirmed=bool(lo[0] > H1_MARGIN))
        print(f"  H1 {kd:4s}: RMSE(terminal) - RMSE(strategy) = {pt:+.3f} [{lo[0]:+.3f}, {lo[1]:+.3f}]  (lower bound > {H1_MARGIN}) -> {'CONFIRMED' if lo[0] > H1_MARGIN else 'not confirmed'}")
    r2k = evaluate(D, "readings", MARGIN, "two", CFG, "9x9", 10.0); r4k = evaluate(D, "readings", MARGIN, "four", CFG, "9x9", 10.0)
    e2, e4 = r2k["pred"]["log10 rho"] - D.y, r4k["pred"]["log10 rho"] - D.y
    d = np.array([rmse(e2, w) - rmse(e4, w) for w in W]); pt = rmse(e2) - rmse(e4); lo = ci(d)
    h["H2"] = dict(diff=pt, ci=lo, confirmed=bool(lo[0] > 0))
    print(f"  H2 (kappa 10, 9x9): RMSE(two tips) - RMSE(four-point) = {pt:+.3f} [{lo[0]:+.3f}, {lo[1]:+.3f}]  (lower bound > 0) -> {'CONFIRMED' if lo[0] > 0 else 'not confirmed'}")
    for lab, lo_r, hi_r in (REGIMES[0], REGIMES[2]):
        m = (D.rho >= lo_r) & (D.rho < hi_r)
        pr = ref["allpairs"]["pred"]["log10 R0"][m] - ref["allpairs"]["targets"]["log10 R0"][m]
        sd = float(np.std(ref["allpairs"]["targets"]["log10 R0"][m]))
        b = np.array([rmse(pr, w[m]) / sd for w in W]); h[f"H3_{lab.split(' ')[0]}"] = dict(ratio=rmse(pr) / sd, ci=ci(b))
    wl, jl = h["H3_wire-limited"], h["H3_junction-limited"]
    h["H3"] = dict(confirmed=bool(wl["ci"][1] <= H3_LO and jl["ci"][0] >= H3_HI))
    print(f"  H3 (all pairs, margin of R0): ratio {wl['ratio']:.2f} [{wl['ci'][0]:.2f}, {wl['ci'][1]:.2f}] wire-limited (upper bound <= {H3_LO}); "
          f"{jl['ratio']:.2f} [{jl['ci'][0]:.2f}, {jl['ci'][1]:.2f}] junction-limited (lower bound >= {H3_HI}) -> {'CONFIRMED' if h['H3']['confirmed'] else 'not confirmed'}")
    out["hypotheses"] = h
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
