"""
Admittance-matrix invertibility checks for the multi-node self-loop tests
(test_self_loops_multi_node.ipynb).

Adds a verification layer on top of the existing node-loop test machinery
(node_loop_test_runner.py / loop_topology_utils.py / laplacian_utils.py /
impedance_calculator.py): after a node-loop is injected and then removed,
confirm that the reduced admittance (Laplacian) matrix M for the cleaned
network is non-singular and can genuinely be inverted, and that doing so
to solve for the node voltages reproduces the voltages already computed
by ImpedanceCalculator.calculate() (which internally uses
np.linalg.solve / np.linalg.lstsq).

Nothing in laplacian_utils.py, impedance_calculator.py,
node_loop_test_runner.py or loop_topology_utils.py is modified here --
this module only calls their existing, unmodified functions
(`compute_reduced_matrix`, `run_node_loop_test`,
`check_percolation_loops`, ...).
"""
import io
import traceback
from contextlib import redirect_stdout

import numpy as np

from laplacian_utils import compute_reduced_matrix
from node_loop_test_runner import run_node_loop_test
from loop_topology_utils import check_percolation_loops, save_node_loop_experiment
from experiment_manager import setup_experiment
from node_loop_visualization import display_and_save_plot


def check_admittance_invertible(intersections, segment_to_nodes, source_node, drain_node,
                                 k=1.0, voltages_from_calc=None, label=""):
    """
    Verify that the reduced admittance (Laplacian) matrix M for a given
    network state is non-singular and invertible, and that solving
    M @ V_internal = b via *explicit matrix inversion* reproduces the
    voltages already computed by ImpedanceCalculator.calculate().

    M and its condition number come straight from `compute_reduced_matrix`
    (laplacian_utils.py, unmodified) -- the same function
    node_loop_test_runner / print_laplacian already rely on.

    Parameters
    ----------
    intersections : list of [x, y]
    segment_to_nodes : dict {edge_idx: (n1, n2)}
    source_node, drain_node : int
    k : float
    voltages_from_calc : array-like or None
        Voltages already returned by ImpedanceCalculator.calculate(), to
        cross-check the manual inversion against. If None, only the
        invertibility check is performed (no cross-check).
    label : str

    Returns
    -------
    dict with keys: label, n_nodes, n_internal, condition_number,
    is_singular (per compute_reduced_matrix's 1e10 threshold),
    invertible (hard np.linalg.inv success/failure),
    max_voltage_discrepancy (vs. voltages_from_calc, NaN if not provided).
    """
    L, M, cond_M, is_singular = compute_reduced_matrix(
        intersections, segment_to_nodes, source_node, drain_node, k=k
    )

    n_nodes = len(intersections)
    boundary_list = [source_node, drain_node]
    internal_nodes = [i for i in range(n_nodes) if i not in boundary_list]

    out = {
        'label': label,
        'n_nodes': n_nodes,
        'n_internal': len(internal_nodes),
        'condition_number': cond_M,
        'is_singular': is_singular,
    }

    if len(internal_nodes) == 0:
        # No internal nodes to solve for -- trivially invertible (0x0 matrix)
        out['invertible'] = True
        out['max_voltage_discrepancy'] = 0.0
        return out

    # Hard proof of invertibility: np.linalg.inv raises LinAlgError if M
    # is genuinely singular (unlike cond_M > 1e10, which is a threshold
    # heuristic) -- this is the actual "can it be inverted?" check.
    try:
        M_inv = np.linalg.inv(M)
        invertible = True
    except np.linalg.LinAlgError:
        M_inv = None
        invertible = False

    out['invertible'] = invertible

    if invertible and voltages_from_calc is not None:
        V_boundary = np.array([1.0, 0.0])  # [source, drain], matches boundary_list order
        L_ib = np.array([[L[i, j] for j in boundary_list] for i in internal_nodes])
        b = -L_ib @ V_boundary
        V_internal_manual = M_inv @ b

        V_manual_full = np.array(voltages_from_calc, dtype=float).copy()
        for local_i, node in enumerate(internal_nodes):
            V_manual_full[node] = V_internal_manual[local_i]

        discrepancy = float(np.max(np.abs(
            V_manual_full - np.array(voltages_from_calc, dtype=float)
        )))
        out['max_voltage_discrepancy'] = discrepancy
    else:
        out['max_voltage_discrepancy'] = float('nan')

    return out


def _adm_summary_line(tid, adm_clean):
    return ("      admittance (cleaned): {}x{} internal, cond(M)={:.3e}, "
            "singular={}, invertible={}, |ΔV| vs solve={:.2e}").format(
        adm_clean['n_internal'], adm_clean['n_internal'],
        adm_clean['condition_number'],
        'YES' if adm_clean['is_singular'] else 'no',
        'YES' if adm_clean['invertible'] else 'NO',
        adm_clean['max_voltage_discrepancy'])


def run_and_report_admittance_test(cfg, k):
    """
    Same run/inject/cleanup/assert/persist/plot flow as
    `run_and_report_node_loop_test` (node_loop_test_runner.py, reused
    unmodified via `run_node_loop_test`), PLUS an explicit check that,
    after the node-loop is removed, the reduced admittance matrix is
    non-singular and invertible, and that inverting it to solve for
    voltages reproduces ImpedanceCalculator's own result.
    """
    tid = cfg['id']
    label = cfg['label']
    inter, g, s, src, drn = cfg['net']

    kwargs = dict(host_node=cfg['host_node'],
                  n_loop_nodes=cfg['n_loop_nodes'],
                  topology=cfg['topology'])
    for kk in ('n_branch_nodes', 'branch_from', 'chord_pair'):
        if kk in cfg:
            kwargs[kk] = cfg[kk]

    try:
        with redirect_stdout(io.StringIO()):
            R, loop_ids = run_node_loop_test(
                inter, g, s, src, drn, k, label='T{}'.format(tid), **kwargs)

        R_base = R['baseline']['impedance']
        R_loop = R['looped']['impedance']
        R_clean = R['cleaned']['impedance']

        n_ring = cfg['n_loop_nodes']
        n_branch = cfg.get('n_branch_nodes', 0)
        ring_ids = loop_ids[:n_ring]
        branch_ids = loop_ids[n_ring:] if n_branch > 0 else []

        restored_ok = abs(R_clean - R_base) < 1e-6
        perc_ok = check_percolation_loops(R['looped']['seg'], src, drn)

        assert abs(R_loop - R_base) < 1e-3, (
            "R changed {:.2e} (>{:.0e})".format(R_loop - R_base, 1e-3))
        assert restored_ok, (
            "Restored R differs {:.2e} (>{:.0e})".format(R_clean - R_base, 1e-6))
        assert perc_ok, "Connectivity lost"

        # ── Admittance-matrix invertibility checks ─────────────────────
        # `remove_node_loops` only ever appends new nodes and then strips
        # them back out, so the surviving (original) node indices are
        # unchanged -- the cleaned network's coordinates are exactly
        # `inter` (the baseline intersections passed into this test).
        adm_base = check_admittance_invertible(
            inter, s, src, drn, k=k,
            voltages_from_calc=R['baseline']['voltages'], label='baseline')
        adm_clean = check_admittance_invertible(
            inter, R['cleaned']['seg'], src, drn, k=k,
            voltages_from_calc=R['cleaned']['voltages'], label='cleaned')

        assert not adm_clean['is_singular'], (
            "Admittance matrix M is singular after self-loop removal "
            "(cond={:.3e})".format(adm_clean['condition_number']))
        assert adm_clean['invertible'], (
            "Admittance matrix M could not be inverted after self-loop removal")
        assert adm_clean['max_voltage_discrepancy'] < 1e-8, (
            "Voltages from manual M^-1 solve differ from ImpedanceCalculator "
            "by {:.2e}".format(adm_clean['max_voltage_discrepancy']))
        # Sanity check: removing the loop should not have changed whether
        # M is invertible relative to the true baseline (no loop at all).
        assert adm_clean['is_singular'] == adm_base['is_singular'], (
            "Singularity of M changed after add+remove node-loop cycle")

        with redirect_stdout(io.StringIO()):
            em = setup_experiment()
            loop_desc = "T{}: {} ring, host={}".format(
                tid, cfg['topology'], cfg['host_node'])
            if cfg['topology'] == 'branched':
                loop_desc += ", +{} branch".format(n_branch)
            elif cfg['topology'] == 'chorded':
                loop_desc += ", +chord"
            save_node_loop_experiment(
                em=em, k=k,
                title='T{} - {}'.format(tid, label),
                intersections=R['looped']['intersections'],
                graph=R['looped']['graph_full'],
                seg_baseline=R['baseline']['seg'],
                seg_with_loops=R['looped']['seg'],
                seg_cleaned=R['cleaned']['seg'],
                res_baseline=R['baseline']['result'],
                res_with_loops=R['looped']['result'],
                res_cleaned=R['cleaned']['result'],
                loop_description=loop_desc,
                loop_node_ids=ring_ids + branch_ids,
                src_node=src, drn_node=drn)

        with redirect_stdout(io.StringIO()):
            display_and_save_plot(em.experiment_dir, R['looped']['intersections'],
                R['baseline']['seg'], R['looped']['seg'], R['cleaned']['seg'],
                R['baseline']['result'], R['looped']['result'], R['cleaned']['result'],
                src, drn, topology='chain',
                title='T{} - {}'.format(tid, label),
                highlight_nodes=ring_ids, branch_nodes=branch_ids)

        pct = (R_loop / R_base - 1) * 100 if R_base else float('nan')
        print("  T{:<2s} PASS  R {:>8.2f} -> {:>8.2f} -> {:>8.2f} Ohm "
              "(Delta{:+.1f}%)  restored=OK  perc=OK  | {}".format(
                  tid, R_base, R_loop, R_clean, pct, label))
        print(_adm_summary_line(tid, adm_clean))

    except AssertionError as e:
        print("  T{}: FAIL - {}".format(tid, e))
    except Exception as e:
        print("  T{}: ERROR - {}\n{}".format(tid, e, traceback.format_exc()))
