"""
Constant-density control for the thickness discussion of the thesis
(chapters/resultados, "Table density-control"): scalar R_eff and onion curves
for combinations that keep the stick number density N/d fixed while d grows,
so that a thicker film is a film of the same material, not a dilution.

The grid of run_effective_resistance_d_task.py holds N fixed as d varies, so
the density falls with thickness there. This script draws N_TRIALS percolating
networks for each (N, a, d) in COMBOS with N proportional to d, computes the
distance-dependent whole-face R_eff and the onion curve of the first source
node (same constants and protocol as run_onion_grid_task.py), and writes one
JSON with per-realization values plus the per-combination summary.

Each (combo, realization) is its own work unit with seed
stable_seed(N, a, d, 'density', r), so the result does not depend on how the
units are split across workers. Runs locally with a process pool (the
combinations chosen percolate freely, so 250 realizations each take minutes,
not hours); on the cluster, wrap it in a single-node job with --workers 48.

Usage:
    python density_control.py [--trials 250] [--workers K] [--out PATH]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from pipeline_utils_3d import generate_percolating_network_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402
from multimeter import run_onion_protocol_pinv  # noqa: E402

CONTACT_DISTANCE = 0.03
R0 = 1000.0
LAYER_DISTANCES = [round(0.1 * n, 2) for n in range(1, 10)]
# N/d = 1500 throughout, a = 0.2 as in the preview figure. Under the clipping
# boundary rule N is a density per unit area of the box, so N/d is the stick
# number density per unit volume.
COMBOS = [(150, 0.2, 0.10), (225, 0.2, 0.15), (300, 0.2, 0.20)]
MAX_ATTEMPTS = 20000
OUT_DEFAULT = os.path.join(HERE, "..", "figures", "partA_constant_density_control.json")


def run_unit(args):
    ci, r = args
    N, a, d = COMBOS[ci]
    np.random.seed(stable_seed(N, round(a, 3), round(d, 3), "density", r))
    res, attempts = generate_percolating_network_3d(
        N=N, a=a, contact_distance=CONTACT_DISTANCE, box_dims=(1.0, 1.0, d),
        k_uniform=R0, max_attempts=MAX_ATTEMPTS)
    if res is None:
        return {"combo_index": ci, "realization": r, "percolated": False, "attempts": attempts}
    R_eff = float(distance_dependent_impedance(res, R0)["impedance"])
    records = run_onion_protocol_pinv(res["pre_merge_intersections"], res["pre_merge_segment_to_nodes"],
                                      res["pre_merge_source_nodes"][0], layer_distances=LAYER_DISTANCES, R0=R0)
    layer_means = {}
    for layer_d in LAYER_DISTANCES:
        vals = [x["r_eff"] for x in records if x["layer_distance"] == layer_d and np.isfinite(x["r_eff"])]
        if vals:
            layer_means[str(layer_d)] = float(np.mean(vals))
    return {"combo_index": ci, "realization": r, "percolated": True, "attempts": attempts,
            "n_sticks": res.get("n_sticks"), "n_nodes": len(res["pre_merge_intersections"]),
            "R_eff": R_eff, "layer_means": layer_means}


def summarize(rows, n_trials):
    out = []
    for ci, (N, a, d) in enumerate(COMBOS):
        mine = [x for x in rows if x["combo_index"] == ci and x["percolated"]]
        R = np.array([x["R_eff"] for x in mine])
        att = sum(x["attempts"] for x in rows if x["combo_index"] == ci)
        curve = {}
        for layer_d in LAYER_DISTANCES:
            vals = [x["layer_means"][str(layer_d)] for x in mine if str(layer_d) in x["layer_means"]]
            if vals:
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                curve[str(layer_d)] = {"mean": float(np.mean(vals)), "sem": sd / np.sqrt(len(vals)), "n": len(vals)}
        out.append({"N": N, "a": a, "d": d, "N_over_d": N / d, "Na2": N * a * a, "n": int(len(R)),
                    "n_trials": n_trials, "boundary": "clip",
                    "mean_ohm": float(R.mean()) if len(R) else None,
                    "sem_ohm": float(R.std(ddof=1) / np.sqrt(len(R))) if len(R) > 1 else None,
                    "perc_rate": len(R) / att if att else None, "total_attempts": att,
                    "mean_n_sticks": float(np.mean([x["n_sticks"] for x in mine])) if mine else None,
                    "mean_n_nodes": float(np.mean([x["n_nodes"] for x in mine])) if mine else None,
                    "onion_curve": curve})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    units = [(ci, r) for r in range(args.trials) for ci in range(len(COMBOS))]
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_unit, units), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(units):
                print(f"  {i}/{len(units)} units  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda x: (x["combo_index"], x["realization"]))
    summary = summarize(rows, args.trials)
    for s in summary:
        print(f"N={s['N']:3d} a={s['a']} d={s['d']:.2f}  N/d={s['N_over_d']:.0f}  Na2={s['Na2']:.1f}  "
              f"n={s['n']}/{s['n_trials']}  rate={s['perc_rate']:.3f}  "
              f"R_eff={s['mean_ohm']:.1f} +/- {s['sem_ohm']:.1f}  "
              f"r0.1={s['onion_curve']['0.1']['mean']:.0f}+/-{s['onion_curve']['0.1']['sem']:.0f}  "
              f"r0.9={s['onion_curve']['0.9']['mean']:.0f}+/-{s['onion_curve']['0.9']['sem']:.0f}")
    with open(args.out, "w") as f:
        json.dump({"combos": [list(c) for c in COMBOS], "n_trials": args.trials, "boundary": "clip",
                   "layer_distances": LAYER_DISTANCES, "R0": R0, "contact_distance": CONTACT_DISTANCE,
                   "summary": summary, "realizations": rows}, f, indent=1)
    print(f"wrote {os.path.normpath(args.out)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
