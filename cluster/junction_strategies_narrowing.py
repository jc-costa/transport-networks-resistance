"""
EXPLORATORY (post hoc): does a measurement narrow the margin of R0, a and N, as that margin grows?
The E2 records, margins of 10 %, 30 % and 50 % (the pre-registered analysis used 10 %). Narrowing is the RMSE
of the error of a parameter after the measurement divided by its prior spread; below 1, the measurement narrows
the margin. Whole geometries held out. Usage: python junction_strategies_narrowing.py <partial.jsonl>
"""
import sys
import numpy as np
import junction_strategies_analysis as sa

films = [f for f in sa.load(sys.argv[1]) if all(v is not None for v in f["P"])]
D = sa.Data(films)
cases = [("terminal only", ("terminal",)), ("two tips 9x27", ("readings", "two", "9x27")),
         ("one tip 9x27", ("readings", "one", "9x27")), ("all pairs", ("allpairs",))]
print("Narrowing (RMSE after / prior spread; below 1 narrows). Columns: R0, a, N")
print(f"{'margin':>7s}  {'information':16s}  {'log R0':>7s} {'log a':>7s} {'log N':>7s}   {'RMSE log rho':>12s}")
for m in (0.10, 0.30, 0.50):
    for lab, c in cases:
        r = sa.evaluate(D, "readings", m, c[1], "base", c[2]) if c[0] == "readings" else sa.evaluate(D, c[0], m)
        n = r["narrowing"]
        print(f"{m:7.0%}  {lab:16s}  {n['log10 R0']['all']:7.2f} {n['log10 a']['all']:7.2f} {n['log10 N']['all']:7.2f}   {r['rmse']:12.3f}")
