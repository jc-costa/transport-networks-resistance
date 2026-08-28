"""
Standalone script backing the SLURM array version of inverse_N_a_from_R_eff.ipynb's
Section 6 (validation). A notebook can't be sliced across array tasks, so the
validation loop (the only part of that notebook that draws new networks) is
duplicated here in script form -- everything else (pooling, curve fit,
solution curves) stays in the notebook, unchanged.

Two modes:
  --prepare        Pool experiments/exp_*/effective_resistance_sweep_summary*.json,
                    fit the model, build the validation candidate list, and save
                    it to validation_plan.json. Run this ONCE, before submitting
                    the array job -- every task reads the same frozen plan, so
                    results stay aligned by index regardless of execution order.
  <task_id>         Run just validation_plan.json[task_id]: draw n_trials
                    percolating networks at that candidate's (N, a), save the
                    result to validation_results/candidate_<NN>.json.

inverse_N_a_from_R_eff.ipynb's Section 6 cell checks validation_results/ for a
complete set of candidate files and loads them instead of recomputing, if
present -- see that cell's source.
"""
import sys
import os
import glob
import json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
EXPERIMENTS_DIR = os.path.join(HERE, "..", "experiments")
PLAN_PATH = os.path.join(HERE, "validation_plan.json")
RESULTS_DIR = os.path.join(HERE, "validation_results")

sys.path.insert(0, UTILS_PATH)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402

R0 = 1000.0
N_TRIALS_VAL = 30
MAX_ATTEMPTS_VAL = 200
TARGET_PERCENTILES = [5, 15, 30, 50, 70, 85, 95]
A_CANDIDATES = [(0.2, "very-compact"), (0.4, "compact-ish"), (0.7, "sparse-ish"), (1.0, "very-sparse")]


def pool_and_fit():
    """Same pooling + BOTH fits + use_general decision as inverse_N_a_from_R_eff.ipynb
    Sections 2/3/3.1 -- must replicate the use_general logic exactly (not just the
    general fit), or this script's predicted (N, a) candidates could diverge from
    what the notebook itself would compute, breaking the cache's index alignment."""
    summary_paths = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "exp_*", "effective_resistance_sweep_summary*.json")))
    merged = defaultdict(list)
    for p in summary_paths:
        with open(p) as f:
            summary = json.load(f)
        for rec in summary.get("sweep", []):
            vals = rec.get("R_eff_values") or []
            if vals:
                merged[(rec["N"], round(rec["a"], 3))].extend(vals)

    Ns_flat, as_flat, R_flat = [], [], []
    for (N, a), vals in merged.items():
        Ns_flat.extend([N] * len(vals))
        as_flat.extend([a] * len(vals))
        R_flat.extend(vals)
    Ns_flat, as_flat, R_flat = np.array(Ns_flat), np.array(as_flat), np.array(R_flat)
    logx = np.log10(Ns_flat * as_flat ** 2)
    logR = np.log10(R_flat)

    # Section 3: simple N*a^2 fit
    slope, intercept = np.polyfit(logx, logR, 1)
    A = 10 ** intercept
    pred_simple = slope * logx + intercept
    ss_tot = np.sum((logR - logR.mean()) ** 2)
    r2_simple = 1 - np.sum((logR - pred_simple) ** 2) / ss_tot

    # Section 3.1: general N^p a^q fit
    X = np.column_stack([np.ones_like(logR), np.log10(Ns_flat), np.log10(as_flat)])
    coef, *_ = np.linalg.lstsq(X, logR, rcond=None)
    log_Ap, p_exp, q_exp = coef
    Ap = 10 ** log_Ap
    pred_general = X @ coef
    r2_general = 1 - np.sum((logR - pred_general) ** 2) / ss_tot

    use_general = r2_general > r2_simple + 0.01

    R_mean_pts = np.array([np.mean(v) for v in merged.values()])
    return {"n_files": len(summary_paths), "n_points": len(merged), "n_trials": len(R_flat),
            "slope": float(slope), "intercept": float(intercept), "A": float(A), "r2_simple": float(r2_simple),
            "Ap": float(Ap), "p_exp": float(p_exp), "q_exp": float(q_exp), "r2_general": float(r2_general),
            "use_general": bool(use_general),
            "R_mean_pts": R_mean_pts.tolist()}


def predict_N(R_target, a, fit):
    if fit["use_general"]:
        return 10 ** ((np.log10(R_target / fit["Ap"]) - fit["q_exp"] * np.log10(a)) / fit["p_exp"])
    x_t = 10 ** ((np.log10(R_target) - fit["intercept"]) / fit["slope"])
    return x_t / a ** 2


def safe_candidate(N, a, node_cap=3000, N_floor=10):
    N = int(round(N))
    a = float(np.clip(a, 0.05, 1.2))
    nodes_est = 0.5 * N ** 2 * a ** 2
    if nodes_est > node_cap:
        N = int(np.sqrt(node_cap / (0.5 * a ** 2)))
    return max(N, N_floor), a


def build_plan(fit):
    R_mean_pts = np.array(fit["R_mean_pts"])
    targets_demo = [float(round(v, -1)) for v in np.percentile(R_mean_pts, TARGET_PERCENTILES)]
    plan = []
    for R_t in targets_demo:
        for a_c, tag in A_CANDIDATES:
            N_c, a_c2 = safe_candidate(predict_N(R_t, a_c, fit), a_c)
            plan.append({"R_target": R_t, "N": N_c, "a": a_c2, "tag": tag})
    return plan


def cmd_prepare():
    fit = pool_and_fit()
    plan = build_plan(fit)
    with open(PLAN_PATH, "w") as f:
        json.dump({"fit": fit, "plan": plan}, f, indent=2)
    print(f"Pooled {fit['n_points']} (N,a) points / {fit['n_trials']} trials from {fit['n_files']} files")
    print(f"N*a^2 fit:  R_eff ~= {fit['A']:.2f} * (N*a^2)^{fit['slope']:.3f}   [R^2={fit['r2_simple']:.3f}]")
    print(f"general fit: R_eff ~= {fit['Ap']:.2f} * N^{fit['p_exp']:.3f} * a^{fit['q_exp']:.3f}   [R^2={fit['r2_general']:.3f}]")
    print(f"-> using {'GENERAL' if fit['use_general'] else 'N*a^2'} fit for predict_N (matches the notebook's own decision)")
    print(f"Wrote {len(plan)}-candidate validation plan to {PLAN_PATH}")


def cmd_run_task(task_id):
    with open(PLAN_PATH) as f:
        plan = json.load(f)["plan"]

    if task_id >= len(plan):
        print(f"task_id {task_id} out of range (plan has {len(plan)} candidates) -- nothing to do")
        return

    cand = plan[task_id]
    N, a, R_t, tag = cand["N"], cand["a"], cand["R_target"], cand["tag"]
    print(f"[task {task_id}] target={R_t} ohm   N={N}  a={a}  ({tag})")

    # stable_seed(), not hash() -- hash() of a string-tagged tuple is
    # randomized per-process (PYTHONHASHSEED), so this line previously drew
    # from an unreproducible seed_base on every array task. See
    # utils/pipeline_utils.py's stable_seed() docstring.
    seed_base = stable_seed(N, round(a, 3), "inverse_validation")
    np.random.seed(seed_base)

    R_effs = []
    total_attempts = 0
    while len(R_effs) < N_TRIALS_VAL:
        res, attempts = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS_VAL)
        total_attempts += attempts
        if res is None:
            print(f"  [WARNING] gave up after {total_attempts} draws, only found {len(R_effs)}/{N_TRIALS_VAL}")
            break
        rl = distance_dependent_impedance(res, R0)
        R_effs.append(rl["impedance"])

    achieved_mean = float(np.mean(R_effs)) if R_effs else None
    rel_err = (achieved_mean - R_t) / R_t if achieved_mean is not None else None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "task_id": task_id, "R_target": R_t, "N": N, "a": a, "tag": tag,
        "n": len(R_effs), "achieved_mean_ohm": achieved_mean,
        "achieved_median_ohm": float(np.median(R_effs)) if R_effs else None,
        "achieved_std_ohm": float(np.std(R_effs)) if R_effs else None,
        "relative_error": rel_err, "R_eff_values": R_effs,
    }
    with open(os.path.join(RESULTS_DIR, f"candidate_{task_id:02d}.json"), "w") as f:
        json.dump(result, f, indent=2)

    mean_str = f"{achieved_mean:.1f}" if achieved_mean is not None else "n/a"
    print(f"[task {task_id}] done. achieved_mean={mean_str} ohm  rel_err={rel_err}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--prepare":
        cmd_prepare()
    else:
        tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
        cmd_run_task(tid)
