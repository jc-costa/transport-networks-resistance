"""
Pre-registered analysis of E2b (PREREGISTRATION_JUNCTION_STRATEGIES_MORE.md). Everything that decides the outcome
is fixed in that file and in the constants below; nothing here may change after the run.

It joins the new readings (run_junction_strategies_more_task.py) with the films of the strategies run (same
geometry, network and film index; the terminal resistance and the junction ratio of each film are checked) and
repeats the registered scenario of the strategies run: R0, a and N known within 10 % (relative errors drawn
uniformly, stored in the strategies run), the junction unknown (rho log-uniform 0.03 to 100), a tip contact of
one wire resistance R0 l per tip, 5 % reading noise, a quadratic ridge regression of log10 rho with WHOLE
GEOMETRIES held out. Films whose all-pairs profile has an empty shell are left out, as in the strategies run.

Usage: python junction_strategies_more_analysis.py <E2 partial.jsonl> <E2b partial.jsonl> [--boot 2000] [--out summary.json]
"""
import sys, json
import numpy as np
import junction_strategies_analysis as sa
import junction_pairwise_review_checks as rc

SEED, BOOT = 0, 2000
MARGIN, KAPPA, NOISE = 0.10, 1.0, 0.05
KINDS = ("one", "two", "four")
RANDOM_ATT = {"9x9": 81, "9x27": 243, "9x81": 729, "9x243": 2187}
WIDE_ATT = {"9x9": 81, "9x27": 243, "9x81": 729}
F1_MARGIN, F2_MARGIN, F3_MARGIN = 0.02, 0.01, 0.01
TERMINAL, ALLPAIRS = None, None          # read from the strategies run in main()


def load_join(p_orig, p_new):
    orig = {}
    for line in open(p_orig):
        r = json.loads(line)
        for n in r["networks"]:
            for j, f in enumerate(n["films"]):
                orig[(r["id"], n["net"], j)] = dict(a=r["a"], N=r["N"], rho=f["rho"], S=f["S"], P=f["P"],
                                                    uc=f["u_c"], ua=f["u_a"], uN=f["u_N"])
    films, miss, bad = [], 0, 0
    for line in open(p_new):
        r = json.loads(line)
        for n in r["networks"]:
            for f in n["films"]:
                o = orig.get((r["id"], n["net"], f["j"]))
                if o is None:
                    miss += 1; continue
                if abs(o["rho"] - f["rho"]) > 1e-9 * o["rho"] or abs(o["S"] - f["S"]) > 1e-4 * o["S"]:
                    bad += 1; continue
                if any(v is None for v in o["P"]):
                    continue
                films.append(dict(o, cell=r["id"], net=n["net"], j=f["j"], strat=f["strat"]))
    if miss or bad:
        raise ValueError(f"{miss} films of the new run are not in the strategies run and {bad} differ from it: the films are not the same")
    return films


class Data:
    def __init__(self, films):
        self.films = films; self.n = len(films)
        g = lambda k: np.array([f[k] for f in films])
        self.a, self.N, self.rho, self.S = g("a"), g("N").astype(float), g("rho"), g("S")
        self.uc, self.ua, self.uN, self.cell = g("uc"), g("ua"), g("uN"), g("cell")
        self.y = np.log10(self.rho)
        nid = {}; self.net = np.array([nid.setdefault((f["cell"], f["net"]), len(nid)) for f in films]); self.nnet = len(nid)
        fm = rc.folds_for(list(self.cell)); self.fold = np.array([fm[c] for c in self.cell])
        self._cache = {}

    def reading(self, kind, mode, design, stat):
        key = (kind, mode, design, stat)
        if key not in self._cache:
            val = np.full((self.n, 9), np.nan); cnt = np.zeros((self.n, 9))
            for i, f in enumerate(self.films):
                d = f["strat"][kind][mode][design]
                val[i] = [np.nan if v is None else v for v in d[stat]]; cnt[i] = d["cnt"]
            self._cache[key] = (val, cnt)
        return self._cache[key]

    def predict(self, kind, mode, design, stat="mean", kappa=KAPPA, kappa_rel=None, margin=MARGIN):
        rng = np.random.default_rng(SEED)
        c = 1 + margin * self.uc
        Sobs = c * self.S * np.exp(NOISE * rng.standard_normal(self.n))
        prior = np.column_stack([np.log10(self.a * (1 + margin * self.ua)), np.log10(self.N * (1 + margin * self.uN))])
        val, cnt = self.reading(kind, mode, design, stat)
        tip = {"one": 1.0, "two": 2.0, "four": 0.0}[kind]
        contact = tip * (kappa_rel * (c * self.S) if kappa_rel is not None else kappa * self.a)
        v = (c[:, None] * val + contact[:, None]) * np.exp(NOISE / np.sqrt(np.maximum(cnt, 1)) * rng.standard_normal(val.shape))
        lv = np.where(cnt > 0, np.log10(np.where(v > 0, v, np.nan)), np.nan)
        X = np.hstack([np.log10(Sobs)[:, None], sa.impute(lv, np.log10(Sobs)), prior])
        p = np.zeros(self.n)
        for f in range(5):
            tr, te = self.fold != f, self.fold == f
            p[te] = rc.fit_predict(X[tr], self.y[tr], X[te])
        succ = float(cnt.sum() / (self.n * 9 * int(design.split("x")[1])))
        return p, succ


def rmse(e, w=None):
    return float(np.sqrt(np.mean(e ** 2))) if w is None else float(np.sqrt(np.sum(w * e ** 2) / w.sum()))


def boot_weights(D, boot, rng):
    net_cell = {}
    for i in range(D.n):
        net_cell.setdefault(D.cell[i], set()).add(D.net[i])
    W = []
    for _ in range(boot):
        w = np.zeros(D.nnet)
        for ids in net_cell.values():
            np.add.at(w, rng.choice(sorted(ids), len(ids)), 1)
        W.append(w[D.net])
    return np.array(W)


def diff_ci(D, W, p1, p2):
    e1, e2 = p1 - D.y, p2 - D.y
    d = np.array([rmse(e1, w) - rmse(e2, w) for w in W])
    return rmse(e1) - rmse(e2), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    films = load_join(sys.argv[1], sys.argv[2])
    boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else BOOT
    D = Data(films)
    print(f"{D.n} films from {D.nnet} networks in {len(set(D.cell))} geometries (joined with the strategies run, margin {MARGIN:.0%}, kappa {KAPPA:g}, noise {NOISE:.0%})")
    # references of the strategies run, on the same films
    orig = sa.Data([dict(cell=f["cell"], net=f["net"], a=f["a"], N=f["N"], rho=f["rho"], S=f["S"], P=f["P"], uc=f["uc"], ua=f["ua"], uN=f["uN"], strat=None) for f in films])
    ref_t = sa.evaluate(orig, "terminal", MARGIN)["rmse"]; ref_a = sa.evaluate(orig, "allpairs", MARGIN)["rmse"]
    print(f"terminal resistance alone {ref_t:.3f}; all pairs {ref_a:.3f}")
    out = {"terminal": ref_t, "allpairs": ref_a, "random": {}, "chosen_vs_wide": {}, "estimators": {}, "hypotheses": {}}
    pred = {}
    print("\nMore readings (random landing, base configuration): RMSE of log10 rho [share of attempts that land]")
    print(f"  {'attempts':>8s}  " + "  ".join(f"{kd:>16s}" for kd in KINDS))
    for dz, att in RANDOM_ATT.items():
        row = []
        for kd in KINDS:
            p, sc = D.predict(kd, "random", dz); pred[(kd, "random", dz, "mean")] = p
            out["random"].setdefault(kd, {})[str(att)] = dict(rmse=rmse(p - D.y), success=sc,
                                                              gap_closed=(ref_t - rmse(p - D.y)) / (ref_t - ref_a))
            row.append(f"{rmse(p - D.y):.3f} [{sc:.2f}]")
        print(f"  {att:8d}  " + "  ".join(f"{r:>16s}" for r in row))
    print("  fraction of the gap to all pairs closed, two tips: " + ", ".join(f"{att}: {out['random']['two'][str(att)]['gap_closed']:.2f}" for att in RANDOM_ATT.values()))

    rng = np.random.default_rng(SEED); W = boot_weights(D, boot, rng)
    print("\nChosen sites against wide random landing (footprint 0.05 for both): RMSE of log10 rho [share that land]")
    print(f"  {'attempts':>8s}  " + "  ".join(f"{kd + ' wide / chosen':>26s}" for kd in KINDS))
    for dz, att in WIDE_ATT.items():
        row = []
        for kd in KINDS:
            pw, sw = D.predict(kd, "wide", dz); pc, sc = D.predict(kd, "chosen", dz)
            pred[(kd, "wide", dz, "mean")] = pw; pred[(kd, "chosen", dz, "mean")] = pc
            d, ci = diff_ci(D, W, pw, pc)
            out["chosen_vs_wide"].setdefault(kd, {})[str(att)] = dict(wide=rmse(pw - D.y), chosen=rmse(pc - D.y), diff=d, ci=ci, success_wide=sw, success_chosen=sc)
            row.append(f"{rmse(pw - D.y):.3f}/{rmse(pc - D.y):.3f} [{sw:.2f}/{sc:.2f}]")
        print(f"  {att:8d}  " + "  ".join(f"{r:>26s}" for r in row))

    print("\nMean against median of the readings of a position (random landing): RMSE of log10 rho")
    print(f"  {'attempts':>8s}  " + "  ".join(f"{kd + ' mean / median':>22s}" for kd in KINDS))
    for dz, att in RANDOM_ATT.items():
        row = []
        for kd in KINDS:
            pm, _ = D.predict(kd, "random", dz, "median"); pred[(kd, "random", dz, "median")] = pm
            row.append(f"{rmse(pred[(kd, 'random', dz, 'mean')] - D.y):.3f}/{rmse(pm - D.y):.3f}")
            out["estimators"].setdefault(kd, {})[str(att)] = dict(mean=rmse(pred[(kd, 'random', dz, 'mean')] - D.y), median=rmse(pm - D.y))
        print(f"  {att:8d}  " + "  ".join(f"{r:>22s}" for r in row))

    print("\nContact resistance relative to the film (random landing, two tips), RMSE at 243 and 2187 attempts")
    for kr in (0.1, 0.5, 1.0):
        r243 = rmse(D.predict("two", "random", "9x27", kappa_rel=kr)[0] - D.y); r2187 = rmse(D.predict("two", "random", "9x243", kappa_rel=kr)[0] - D.y)
        out["random"].setdefault("two_kappa_rel", {})[str(kr)] = dict(r243=r243, r2187=r2187)
        print(f"  contact = {kr} x S: {r243:.3f}, {r2187:.3f}")

    h = out["hypotheses"]
    d, ci = diff_ci(D, W, pred[("two", "random", "9x27", "mean")], pred[("two", "random", "9x243", "mean")])
    h["F1"] = dict(diff=d, ci=ci, confirmed=bool(ci[0] > F1_MARGIN))
    d2, ci2 = diff_ci(D, W, pred[("two", "wide", "9x27", "mean")], pred[("two", "chosen", "9x27", "mean")])
    h["F2"] = dict(diff=d2, ci=ci2, confirmed=bool(ci2[0] > F2_MARGIN))
    d3, ci3 = diff_ci(D, W, pred[("two", "random", "9x27", "mean")], pred[("two", "random", "9x27", "median")])
    h["F3"] = dict(diff=d3, ci=ci3, confirmed=bool(ci3[0] > F3_MARGIN))
    print("\nHypotheses (two tips, kappa 1 wire resistance, margin 10 %)")
    print(f"  F1 more readings: RMSE(243) - RMSE(2187) = {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] (lower bound > {F1_MARGIN}) -> {'CONFIRMED' if h['F1']['confirmed'] else 'not confirmed'}")
    print(f"  F2 chosen sites:  RMSE(wide) - RMSE(chosen) at 243 = {d2:+.3f} [{ci2[0]:+.3f}, {ci2[1]:+.3f}] (lower bound > {F2_MARGIN}) -> {'CONFIRMED' if h['F2']['confirmed'] else 'not confirmed'}")
    print(f"  F3 estimator:     RMSE(mean) - RMSE(median) at 243 = {d3:+.3f} [{ci3[0]:+.3f}, {ci3[1]:+.3f}] (lower bound > {F3_MARGIN}) -> {'CONFIRMED' if h['F3']['confirmed'] else 'not confirmed'}")
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
