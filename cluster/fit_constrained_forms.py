"""
The three constrained rows of Table "fit-results" and the baseline of the pinned row.

fit_figures.py prints the excluded-area law, the general power law and the percolation form with
c free. This script pools the same realizations (fit_figures.pool) and fits the rest:
  * wire-limited form, a^1 fixed:        R = A a (N a^2)^-t
  * percolation form, c = 5.64 pinned:  R = A a^q (N a^2 - 5.64)^-t, on the trials above 5.64
  * dense wire limit, q = t = 1, c = 0: R = A a / (N a^2)
  * the general power law refitted on the same trials as the pinned form (its baseline)

Usage: python fit_constrained_forms.py
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
import fit_figures as ff  # noqa: E402

N, a, R = ff.pool()
x, lR = N * a ** 2, np.log10(R)

X = np.column_stack([np.ones_like(lR), -np.log10(x)])
co = np.linalg.lstsq(X, lR - np.log10(a), rcond=None)[0]
print(f"wire-limited, a^1 fixed:  R = {10 ** co[0]:.0f} a (N a^2)^-{co[1]:.3f}   R2 = {ff.r2(lR, X @ co + np.log10(a)):.3f}")

m = x > 5.64
pin = lambda pr: pr[0] + pr[1] * np.log10(a[m]) - pr[2] * np.log10(x[m] - 5.64)  # noqa: E731
res = least_squares(lambda pr: pin(pr) - lR[m], [3, 1, 1])
print(f"c = 5.64 pinned:          R = {10 ** res.x[0]:.0f} a^{res.x[1]:.3f} (N a^2 - 5.64)^-{res.x[2]:.3f}   "
      f"{m.sum()} trials   R2 = {ff.r2(lR[m], pin(res.x)):.3f}")

lA = np.mean(lR - np.log10(a) + np.log10(x))
print(f"dense wire limit:         R = {10 ** lA:.0f} a (N a^2)^-1   R2 = {ff.r2(lR, lA + np.log10(a) - np.log10(x)):.3f}")

Xg = np.column_stack([np.ones(m.sum()), np.log10(N[m]), np.log10(a[m])])
cg = np.linalg.lstsq(Xg, lR[m], rcond=None)[0]
print(f"general power law on the same {m.sum()} trials: R2 = {ff.r2(lR[m], Xg @ cg):.3f}")
