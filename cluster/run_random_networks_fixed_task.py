"""
Robustness batch of Sec. 4.1.2 (run_random_networks_task.py), rerun with the crossing loop
covering every placed stick. Under the clipping rule generate_points places about
N (1 + 4a/pi) sticks, but utils/random_network_batch.run_full_network_case passes the density
N to identify_intersections, which then loops over the first N sticks only, so sticks with
index >= N never cross anything (run_full_pipeline passes len(points_1) and is not affected).
This wrapper patches that one argument without editing the module, keeps every other line of
the batch, and writes random_networks_results_fixed/chunk_NN.json.
Usage: python run_random_networks_fixed_task.py <task_id>   (same N_TASKS split as the original)
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils")); sys.path.insert(0, HERE)
import random_network_batch as rnb  # noqa: E402
import run_random_networks_task as orig  # noqa: E402

_identify = rnb.identify_intersections
def _identify_all(points_1, points_2, ip, l1, l2, graph, angles, a, N):
    return _identify(points_1, points_2, ip, l1, l2, graph, angles, a, len(points_1))
rnb.identify_intersections = _identify_all

RESULTS_DIR = os.environ.get("RANDOM_FIXED_OUT", os.path.join(HERE, "random_networks_results_fixed"))

def main():
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    configs = rnb.generate_diversified_configs(orig.N_CASES, orig.N_MIN, orig.N_MAX, orig.A_MIN, orig.A_MAX,
                                               param_seed=2025, case_seed_base=10_000)
    chunk = -(-orig.N_CASES // orig.N_TASKS)
    mine = configs[task_id * chunk: min((task_id + 1) * chunk, orig.N_CASES)]
    rows = [rnb.run_full_network_case(cfg, k=orig.K_OHM, plot=False, verbose=False)[0] for cfg in mine]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(rows, open(os.path.join(RESULTS_DIR, f"chunk_{task_id:02d}.json"), "w"), indent=2)
    print(f"[task {task_id}] {len(rows)} cases, {sum(r['has_percolation'] for r in rows)} percolating")

if __name__ == "__main__":
    main()
