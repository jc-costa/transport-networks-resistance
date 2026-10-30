"""Regenerate the Part A preview onion-curve figure with error bars.

Mirrors notebooks/r_eff_ij_vs_N_a_d.ipynb Part A (cells 5, 6, 9) exactly, except:
  - N_TRIALS raised from 4 to 50 so the per-layer spread is meaningfully estimated
  - per-layer standard error plotted as error bars
  - sample std (ddof=1) instead of population std
"""
import sys, os, json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.append(os.path.abspath("utils"))
from pipeline_utils_3d import generate_percolating_network_3d
from pipeline_utils import distance_dependent_impedance, stable_seed
from multimeter import run_onion_protocol_pinv

R0 = 1000.0
CONTACT_DISTANCE = 0.03
MAX_ATTEMPTS = 500
N_TRIALS = 50
DRAW_BUDGET = 40000   # global per-combo cap; the grid script instead gives up after
                      # one failed 500-attempt block, which cannot reach 50 successes
                      # at these low percolation rates
LAYER_DISTANCES = [round(0.1 * n, 2) for n in range(1, 10)]
preview_combos = [(70, 0.2, 0.1), (70, 0.2, 0.15), (45, 0.2, 0.1)]
OUT = sys.argv[1]


def run_one_combo(N, a, d, n_trials):
    np.random.seed(stable_seed(N, round(a, 3), round(d, 3), 'ext'))
    R_effs, per_network_layer_means, total_attempts = [], [], 0
    while len(R_effs) < n_trials and total_attempts < DRAW_BUDGET:
        res, attempts = generate_percolating_network_3d(
            N=N, a=a, contact_distance=CONTACT_DISTANCE,
            box_dims=(1.0, 1.0, d), k_uniform=R0, max_attempts=MAX_ATTEMPTS)
        total_attempts += attempts
        if res is None:
            continue
        R_effs.append(distance_dependent_impedance(res, R0)['impedance'])
        ref = res['pre_merge_source_nodes'][0]
        records = run_onion_protocol_pinv(
            res['pre_merge_intersections'], res['pre_merge_segment_to_nodes'],
            ref, layer_distances=LAYER_DISTANCES, R0=R0)
        lm = {}
        for ld in LAYER_DISTANCES:
            vals = [r['r_eff'] for r in records
                    if r['layer_distance'] == ld and np.isfinite(r['r_eff'])]
            if vals:
                lm[ld] = float(np.mean(vals))
        per_network_layer_means.append(lm)

    curve = {}
    for ld in LAYER_DISTANCES:
        per_real = [lm[ld] for lm in per_network_layer_means if ld in lm]
        if per_real:
            n = len(per_real)
            s = float(np.std(per_real, ddof=1)) if n > 1 else 0.0
            curve[ld] = {'mean': float(np.mean(per_real)), 'std': s,
                         'sem': s / np.sqrt(n) if n else 0.0, 'n': n}
    return {'N': N, 'a': a, 'd': d, 'n': len(R_effs), 'attempts': total_attempts,
            'perc_rate': len(R_effs) / total_attempts if total_attempts else None,
            'mean_ohm': float(np.mean(R_effs)) if R_effs else None,
            'sem_ohm': float(np.std(R_effs, ddof=1) / np.sqrt(len(R_effs))) if len(R_effs) > 1 else None,
            'onion_curve': curve}


rows = []
for (N, a, d) in preview_combos:
    t0 = time.time()
    row = run_one_combo(N, a, d, N_TRIALS)
    rows.append(row)
    print(f"N={N:3d} a={a} d={d:.2f}  {row['n']}/{N_TRIALS} percolating "
          f"({row['attempts']} draws, rate {row['perc_rate']:.3f})  "
          f"scalar mean={row['mean_ohm']:.1f} +/- {row['sem_ohm']:.1f}  [{time.time()-t0:.0f}s]", flush=True)

fig, ax = plt.subplots(figsize=(7, 5.5))
for row in rows:
    ds = sorted(row['onion_curve'])
    means = np.array([row['onion_curve'][d]['mean'] for d in ds])
    sems = np.array([row['onion_curve'][d]['sem'] for d in ds])
    ax.errorbar(ds, means, yerr=sems, fmt='o-', capsize=3, markersize=4, linewidth=1.4,
                label=f"$N$={row['N']}, $a$={row['a']}, $d$={row['d']} ($n$={row['n']})")
ax.set_yscale('log')
ax.set_xlabel(r'Distance from reference node, $r$')
ax.set_ylabel(r'Mean $R_{\mathrm{eff},ij}$ per layer ($\Omega$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
json.dump(rows, open(OUT.replace('.png', '.json'), 'w'), indent=2)
print("wrote", OUT)
