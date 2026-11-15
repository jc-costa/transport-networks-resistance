"""
EXPLORATORY (post hoc): against which target does the frozen S+P estimator, trained on equal junctions,
read the junction ratio when the junctions vary? Applied to the transfer run, full range of rho.
Targets: the base value rho, the geometric mean of the junction resistance, its arithmetic mean
(log-normal: geometric exp(-sigma^2/2), arithmetic 1; bimodal: geometric 100^(4/9), arithmetic 45).
Usage: python junction_pairwise_transfer_estimands.py <partial.jsonl> <frozen.json>
"""
import sys, json
import numpy as np
import junction_pairwise_transfer_analysis as ta
import junction_pairwise_review_checks as rc

path, fz = sys.argv[1], sys.argv[2]
frozen = json.load(open(fz))
rows = [r for r in ta.load(path) if rc.complete(r["P"])]
S = np.array([r["S"] for r in rows]); P = np.array([r["P"] for r in rows])
y = np.log10([r["rho"] for r in rows]); var = np.array([r["variant"] for r in rows])
p = ta.predict(frozen, "S+P", S, P)
targets = {"lognormal_s1.5": dict(geo=np.log10(np.exp(-1.5 ** 2 / 2)), arith=0.0),
           "bimodal_k100": dict(geo=np.log10(100 ** (4 / 9)), arith=np.log10(45.0))}
print("S+P estimate against three possible targets, full range, log10 units")
print(f"{'rule':16s} {'target':24s} {'RMSE':>6s} {'bias':>7s} {'median factor':>14s}")
for v, t in targets.items():
    m = var == v
    for name, off in (("base value rho", 0.0), ("geometric mean of R_j", t["geo"]), ("arithmetic mean of R_j", t["arith"])):
        e = p[m] - (y[m] + off)
        print(f"{v:16s} {name:24s} {np.sqrt(np.mean(e ** 2)):6.3f} {np.mean(e):+7.3f} {10 ** np.median(np.abs(e)):14.2f}")
