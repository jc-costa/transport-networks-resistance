"""
Confirmatory run of the 2D pairwise test (paper item e). See PREREGISTRATION.md,
which fixes the hypotheses and the analysis before this script is run.

The thesis's E7 found, in a comparison made after its results were seen, that
the mean of sixteen onion curves (C16) and the all-pairs profile (P) assign a
network to its (N, a) candidate more often than the terminal resistance alone.
This run repeats Part A on FRESH networks, so that the comparison is tested on
data drawn after the hypothesis was stated:

  * the same 21 candidates of validation_plan.json (A19, the 890 Ohm candidate
    at a = 0.2 with only 4 networks, is left out, as in the thesis);
  * new seeds, stable_seed(N, round(a, 3), 'pairwise2d_confirm'), so no network
    of the thesis is reused;
  * 60 percolating networks per candidate (thesis: 30), same generator call,
    give-up rule and features (run_pairwise2d_task.network_features, unchanged).

Output: pairwise2d_confirm_results/partial.jsonl, one row per candidate, same
schema as pairwise2d_results/partial.jsonl; resumable.

Usage:
  python run_pairwise2d_confirm_task.py --parallel 48
  python run_pairwise2d_confirm_task.py --units A07 --trials 2 --out /tmp/pwc   # quick check
"""
import sys, os, json, time
if "--parallel" in sys.argv:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_pairwise2d_task as rp  # noqa: E402  (network_features, constants, generator imports)

TAG = "pairwise2d_confirm"
N_TRIALS = 60
EXCLUDE = {"A19"}
OUT_DEFAULT = os.path.join(HERE, "pairwise2d_confirm_results")


def run_unit(u, n_trials=None):
    t0 = time.time()
    N, a = int(u["N"]), float(u["a"])
    n_trials = n_trials or N_TRIALS
    np.random.seed(rp.stable_seed(N, round(a, 3), TAG))
    R, feats, attempts_total = [], [], 0
    while len(R) < n_trials:
        res, att = rp.generate_percolating_network(N, a, k_uniform=rp.R0, max_attempts=rp.MAX_ATTEMPTS,
                                                   boundary=rp.BOUNDARY)
        attempts_total += att
        if res is None:
            break
        k = len(R)
        rng = np.random.default_rng(rp.stable_seed(N, round(a, 3), TAG, k, "anchors"))
        R.append(float(rp.distance_dependent_impedance(res, rp.R0)["impedance"]))
        feats.append(rp.network_features(res, rng))
    row = {k: u[k] for k in u if k != "stored_R_eff_values"}
    row.update({"seed_tag": TAG, "n": len(R), "R_eff_values": R, "reproduced_stored": None,
                "attempts_total": attempts_total, "perc_rate": len(R) / attempts_total if attempts_total else 0.0,
                "layer_distances": rp.LAYERS, "tolerance": rp.TOL, "seconds": time.time() - t0})
    for key in ("n_nodes", "n_reachable", "n_source", "n_interior", "anchors", "anchor_roles",
                "anchor_curves", "C1", "C4", "C16", "P", "P_pairs"):
        row[key] = [f[key] for f in feats]
    return row


def _job(args):
    u, nt = args
    return u["id"], run_unit(u, nt)


def main():
    args = sys.argv[1:]
    plan = [u for u in json.load(open(rp.PLAN_PATH))["units"] if u["part"] == "A" and u["id"] not in EXCLUDE]
    trials = int(rp._opt(args, "--trials")) if "--trials" in args else None
    out = rp._opt(args, "--out") or OUT_DEFAULT
    if "--units" in args:
        want = rp._opt(args, "--units").split(",")
        plan = [u for u in plan if u["id"] in want]
    os.makedirs(out, exist_ok=True)
    part = os.path.join(out, "partial.jsonl")
    done = rp._resume(part)
    todo = sorted([u for u in plan if u["id"] not in done], key=lambda u: -(u["N"] * u["a"]) ** 2)
    print(f"{len(done)} candidates already in {part}; {len(todo)} to run", flush=True)
    nproc = int(rp._opt(args, "--parallel", 1))
    with open(part, "a") as fh:
        if nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as pool:
                for i, (uid, row) in enumerate(pool.imap_unordered(_job, [(u, trials) for u in todo]), 1):
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{i}/{len(todo)}] {uid} N={row['N']} a={row['a']} n={row['n']} {row['seconds']:.0f}s", flush=True)
        else:
            for i, u in enumerate(todo, 1):
                row = run_unit(u, trials)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{i}/{len(todo)}] {u['id']} N={row['N']} a={row['a']} n={row['n']} {row['seconds']:.0f}s", flush=True)
    print(f"done -> {part}")


if __name__ == "__main__":
    main()
