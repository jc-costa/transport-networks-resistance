"""
Extend inverse candidate 19 (target 890 ohm, N = 80, a = 0.2, N a^2 = 3.2) from
the 4 percolating networks of the thesis run to 30, with the thesis's own
pipeline and seed stream.

The thesis task (run_validation_task.py) seeded np.random with
stable_seed(N, round(a, 3), 'inverse_validation', 'clip') and called
generate_percolating_network(..., max_attempts=200) until it had 30 networks;
for this candidate one call failed 200 draws in a row after the 4th network and
the task gave up. A failed call consumes the RNG exactly as successful draws do,
so continuing the same stream with a larger attempt budget reproduces the 4
stored networks first and then finds the next ones. The script checks the
first 4 against validation_results/candidate_19.json before keeping anything.

Usage (PIPELINE_DIR = the ColabProject folder):
  PIPELINE_DIR=... python extend_inverse_candidate.py [--n 30] [--out FILE]

Output: figures/data/inverse_candidate19_extended.json (same schema as the
thesis candidate files, plus draws and the reproduction check).
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.environ.get("PIPELINE_DIR", os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.join(PIPE, "utils"))
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402

TASK_ID, R0, BOUNDARY = 19, 1000.0, "clip"
ATTEMPTS_PER_NETWORK = 200000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures", "data", "inverse_candidate19_extended.json"))
    args = ap.parse_args()

    stored = json.load(open(os.path.join(PIPE, "cluster", "validation_results", f"candidate_{TASK_ID:02d}.json")))
    N, a, R_t = stored["N"], stored["a"], stored["R_target"]
    print(f"candidate {TASK_ID}: target {R_t} ohm, N = {N}, a = {a}; {stored['n']} stored networks", flush=True)

    np.random.seed(stable_seed(N, round(a, 3), "inverse_validation", BOUNDARY))
    R, draws, t0 = [], 0, time.time()
    while len(R) < args.n:
        res, k = generate_percolating_network(N, a, k_uniform=R0, max_attempts=ATTEMPTS_PER_NETWORK, boundary=BOUNDARY)
        draws += k
        if res is None:
            sys.exit(f"no percolating network in {ATTEMPTS_PER_NETWORK} draws; stopping with {len(R)}")
        R.append(distance_dependent_impedance(res, R0)["impedance"])
        i = len(R) - 1
        if i < stored["n"]:
            rel = abs(R[i] - stored["R_eff_values"][i]) / stored["R_eff_values"][i]
            print(f"  network {i}: {R[i]:.6f} ohm, stored {stored['R_eff_values'][i]:.6f}, rel {rel:.1e}", flush=True)
            if rel > 1e-9:
                sys.exit("seed stream does not reproduce the stored networks; nothing written")
        else:
            print(f"  network {i}: {R[i]:.6f} ohm after {draws} draws ({time.time() - t0:.0f} s)", flush=True)

    mean = float(np.mean(R))
    out = dict(task_id=TASK_ID, R_target=R_t, N=N, a=a, tag=stored.get("tag"), n=len(R),
               achieved_mean_ohm=mean, achieved_median_ohm=float(np.median(R)),
               achieved_std_ohm=float(np.std(R)), relative_error=(mean - R_t) / R_t,
               R_eff_values=R, draws=draws, percolation_rate=len(R) / draws,
               reproduced_stored=stored["n"], seed_parts=[N, round(a, 3), "inverse_validation", BOUNDARY],
               seconds=time.time() - t0)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"done: {len(R)} networks from {draws} draws, mean {mean:.4f} ohm, error {out['relative_error']:+.1%}; wrote {args.out}")


if __name__ == "__main__":
    main()
