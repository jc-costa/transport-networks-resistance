"""
EXPLORATORY (post hoc): where does the gap between a few tip readings and the all-pairs profile come from?
Applied to the E2 records. Not pre-registered. Usage: python junction_strategies_decomposition.py <partial.jsonl>
"""
import sys
import numpy as np
import junction_strategies_analysis as sa

films = [f for f in sa.load(sys.argv[1]) if all(v is not None for v in f["P"])]
D = sa.Data(films)


def r(kind, dz, noise, kappa):
    sa.NOISE = noise
    return sa.evaluate(D, "readings", 0.10, kind, "base", dz, kappa)["rmse"]


print("RMSE of log10 rho at margin 10 %: where the information is lost")
print(f"{'':42s} {'one 9x9':>8s} {'two 9x9':>8s} {'four 9x9':>9s} {'two 9x27':>9s}")
for lab, nz, kp in (("registered (noise 5 %, kappa 1)", 0.05, 1.0), ("no reading noise", 0.0, 1.0),
                    ("no tip contact resistance", 0.05, 0.0), ("neither", 0.0, 0.0)):
    print(f"{lab:42s} {r('one', '9x9', nz, kp):8.3f} {r('two', '9x9', nz, kp):8.3f} {r('four', '9x9', nz, kp):9.3f} {r('two', '9x27', nz, kp):9.3f}")
sa.NOISE = 0.05
for kind in sa.KINDS:
    mean, cnt = D.reading(kind, "base", "9x9"); s = cnt.sum(1) / 81
    print(f"success rate {kind:4s} at 9x9 by stick length a: " + "  ".join(
        f"{lo}-{hi}: {s[(D.a >= lo) & (D.a < hi)].mean():.2f}" for lo, hi in ((0.2, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 1.01))))
for kind in ("one", "two"):
    print(f"{kind} tip(s), kappa 0, RMSE by attempts:", {sa.ATTEMPTS[dz]: round(sa.evaluate(D, 'readings', 0.10, kind, 'base', dz, 0.0)['rmse'], 3) for dz in sa.DESIGNS})
