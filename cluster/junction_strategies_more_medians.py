"""
Post hoc (E2b): median factor errors and the share of the gap to the all-pairs profile that a protocol closes,
for the numbers quoted in the paper. Uses junction_strategies_more_analysis.py on the joined films.
Usage: python junction_strategies_more_medians.py <E2 partial.jsonl> <E2b partial.jsonl> [--out summary.json]
"""
import sys, json
import numpy as np
import junction_strategies_more_analysis as ma
import junction_strategies_analysis as sa

films = ma.load_join(sys.argv[1], sys.argv[2]); D = ma.Data(films)
orig = sa.Data([dict(cell=f["cell"], net=f["net"], a=f["a"], N=f["N"], rho=f["rho"], S=f["S"], P=f["P"], uc=f["uc"], ua=f["ua"], uN=f["uN"], strat=None) for f in films])
rt, ra = sa.evaluate(orig, "terminal", ma.MARGIN), sa.evaluate(orig, "allpairs", ma.MARGIN)
fac = lambda p: float(10 ** np.median(np.abs(p - D.y)))
out = {"terminal": dict(rmse=rt["rmse"], median_factor=rt["median_factor"]), "allpairs": dict(rmse=ra["rmse"], median_factor=ra["median_factor"]), "kappa_wire": {}, "kappa_rel": {}}
gap = lambda r: (rt["rmse"] - r) / (rt["rmse"] - ra["rmse"])
print(f"terminal {rt['rmse']:.3f} ({rt['median_factor']:.2f}); all pairs {ra['rmse']:.3f} ({ra['median_factor']:.2f})")
print("contact of one wire resistance per tip: RMSE (median factor) [gap closed]")
for kd in ma.KINDS:
    for dz, att in (("9x27", 243), ("9x243", 2187)):
        p, sc = D.predict(kd, "random", dz); r = ma.rmse(p - D.y)
        out["kappa_wire"][f"{kd}|{att}"] = dict(rmse=r, median_factor=fac(p), gap_closed=gap(r), success=sc)
        print(f"  {kd:4s} {att:5d}: {r:.3f} ({fac(p):.2f}) [{gap(r):.2f}]")
print("two tips, contact relative to the film's terminal resistance: RMSE (median factor) [gap closed]")
for kr in (0.1, 0.5, 1.0):
    for dz, att in (("9x27", 243), ("9x243", 2187)):
        p, _ = D.predict("two", "random", dz, kappa_rel=kr); r = ma.rmse(p - D.y)
        out["kappa_rel"][f"{kr}|{att}"] = dict(rmse=r, median_factor=fac(p), gap_closed=gap(r))
        print(f"  {kr:3.1f} x S, {att:5d}: {r:.3f} ({fac(p):.2f}) [{gap(r):.2f}]")
for kd in ("one",):
    for kr in (0.1, 0.5, 1.0):
        p, _ = D.predict(kd, "random", "9x243", kappa_rel=kr); r = ma.rmse(p - D.y)
        out["kappa_rel"][f"{kd}|{kr}|2187"] = dict(rmse=r, median_factor=fac(p), gap_closed=gap(r))
        print(f"  one tip, {kr:3.1f} x S, 2187: {r:.3f} ({fac(p):.2f}) [{gap(r):.2f}]")
if "--out" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
