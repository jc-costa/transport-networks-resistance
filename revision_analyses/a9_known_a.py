"""A9 (advisor question 1): with the stick length a known, how well does ONE network's terminal
resistance determine N? Out of sample: the 22 inverse-validation candidates (fresh networks, not
used to fit the law), each network's R inverted through the general law R = 28254 N^-1.694 a^-2.459."""
import json, glob, os
import numpy as np
CL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster")
A, p, q = 28254.0, -1.694, -2.459
errs, merrs = [], []
for f in sorted(glob.glob(os.path.join(CL, "validation_results", "candidate_*.json"))):
    c = json.load(open(f)); N, a = c["N"], c["a"]; Rs = np.array(c["R_eff_values"], float)
    if len(Rs) == 0: continue
    errs += list(np.abs(np.log10((Rs / (A * a**q)) ** (1 / p) / N)))
    merrs.append(abs(np.log10((Rs.mean() / (A * a**q)) ** (1 / p) / N)))
errs, merrs = np.array(errs), np.array(merrs)
print(f"{len(merrs)} candidates, {len(errs)} networks")
print(f"single network, a known: median factor error in N {10**np.median(errs):.3f}, 90th pct {10**np.percentile(errs, 90):.3f}, within 20%: {np.mean(errs < np.log10(1.2)):.0%}")
print(f"candidate mean: median factor error in N {10**np.median(merrs):.3f}, max {10**merrs.max():.3f}")
