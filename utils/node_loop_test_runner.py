"""
Node-loop test runner utilities.

Orchestrates node-loop tests end to end: run baseline impedance, inject a
node-loop (add_node_loop), run impedance again, clean it back up
(remove_node_loops), assert everything is physically consistent, persist
the experiment, and render the 3-panel plot. Used by
test_self_loops_multi_node.ipynb.

All the noisy diagnostic printing that impedance_calculator /
experiment_manager / loop_topology_utils do internally is captured and
suppressed here -- each test prints exactly ONE summary line with just
enough information to sanity-check physical coherence (impedance before/
during/after the loop, whether it was restored, and whether connectivity
held), plus its plot.
"""
import io
import traceback
from contextlib import redirect_stdout

from impedance_calculator import ImpedanceCalculator
from laplacian_utils import compute_reduced_matrix
from self_loop_utils import count_self_loops, check_percolation
from loop_topology_utils import (
    add_node_loop, remove_node_loops, check_percolation_loops,
    save_node_loop_experiment,
)
from experiment_manager import setup_experiment
from node_loop_visualization import display_and_save_plot
from test_networks import k_test


def _run_imp_explicit(intersections, graph, seg, k, src, drn):
    """Run impedance with explicit source/drain nodes (silent -- returns
    the result dict, no printing)."""
    calc = ImpedanceCalculator(intersections, graph, seg)
    calc.source_node = src
    calc.drain_node = drn
    return calc.calculate(k=k, verbose=False)


def run_node_loop_test(intersections, graph, seg, src, drn, k,
                        host_node, n_loop_nodes, topology,
                        n_branch_nodes=1, branch_from=None, chord_pair=None,
                        label=""):
    'Complete test: baseline -> inject -> cleanup. Returns (R_dict, loop_node_ids).'
    res_base = _run_imp_explicit(intersections, graph, seg, k, src, drn)

    inters_loop, seg_loop, graph_loop, loop_node_ids = add_node_loop(
        intersections, seg, graph,
        host_node=host_node, n_loop_nodes=n_loop_nodes, topology=topology,
        n_branch_nodes=n_branch_nodes, branch_from=branch_from, chord_pair=chord_pair,
    )
    res_loop = _run_imp_explicit(inters_loop, graph_loop, seg_loop, k, src, drn)

    inters_clean, seg_clean, graph_clean = remove_node_loops(
        inters_loop, seg_loop, graph_loop, loop_node_ids
    )
    res_clean = _run_imp_explicit(inters_clean, graph_clean, seg_clean, k, src, drn)

    return {
        'baseline': {'result': res_base, 'seg': seg, 'graph': graph,
                     'impedance': res_base['impedance'],
                     'voltages': res_base['voltages']},
        'looped': {'result': res_loop, 'seg': seg_loop, 'graph': graph_loop,
                   'impedance': res_loop['impedance'],
                   'voltages': res_loop['voltages'],
                   'intersections': inters_loop, 'graph_full': graph_loop},
        'cleaned': {'result': res_clean, 'seg': seg_clean, 'graph': graph_clean,
                    'impedance': res_clean['impedance'],
                    'voltages': res_clean['voltages']},
    }, loop_node_ids


def _summary_line(tid, status, R_base, R_loop, R_clean, restored_ok, perc_ok,
                   n_before, n_loop_nodes_count, extra=""):
    'One compact, information-dense line per test -- everything needed to sanity-check coherence.'
    pct = (R_loop / R_base - 1) * 100 if R_base else float('nan')
    return ("  T{:<2s} {:<4s}  R {:>8.2f} -> {:>8.2f} -> {:>8.2f} Ohm "
            "(Delta{:+.1f}%)  restored={}  perc={}  nodes {}->{}{}").format(
        tid, status, R_base, R_loop, R_clean, pct,
        'OK' if restored_ok else 'FAIL',
        'OK' if perc_ok else 'FAIL',
        n_before, n_before + n_loop_nodes_count,
        (" | " + extra) if extra else "")


def run_and_report_node_loop_test(cfg, k=k_test):
    'Execute one test config: run, assert, persist, plot, print one summary line.'
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

        # Plot (kept visible -- this is the actual thing worth looking at)
        with redirect_stdout(io.StringIO()):
            display_and_save_plot(em.experiment_dir, R['looped']['intersections'],
                R['baseline']['seg'], R['looped']['seg'], R['cleaned']['seg'],
                R['baseline']['result'], R['looped']['result'], R['cleaned']['result'],
                src, drn, topology='chain',
                title='T{} - {}'.format(tid, label),
                highlight_nodes=ring_ids, branch_nodes=branch_ids)

        print(_summary_line(tid, 'PASS', R_base, R_loop, R_clean,
                             restored_ok, perc_ok, len(inter), len(loop_ids),
                             extra=label))

    except AssertionError as e:
        print("  T{}: FAIL - {}".format(tid, e))
    except Exception as e:
        print("  T{}: ERROR - {}\n{}".format(tid, e, traceback.format_exc()))


def run_and_report_dual_loop_test(tid, label, intersections, graph, seg, src, drn,
                                   loops, k=k_test, topology='chain'):
    """Execute a test with MULTIPLE node-loops injected simultaneously on
    (possibly) different hosts, e.g. Test H. ``loops`` is a list of dicts,
    each with the same keys `add_node_loop` accepts (host_node,
    n_loop_nodes, topology, and optionally n_branch_nodes/branch_from/
    chord_pair) -- one dict per loop to inject, applied in order.

    Same behavior as `run_and_report_node_loop_test`: run, assert, persist,
    plot, print one summary line.
    """
    try:
        with redirect_stdout(io.StringIO()):
            res_base = _run_imp_explicit(intersections, graph, seg, k, src, drn)

            cur_inter, cur_seg, cur_graph = intersections, seg, graph
            all_loop_ids = []
            for loop_cfg in loops:
                cur_inter, cur_seg, cur_graph, ids = add_node_loop(
                    cur_inter, cur_seg, cur_graph, **loop_cfg)
                all_loop_ids += ids

            res_loop = _run_imp_explicit(cur_inter, cur_graph, cur_seg, k, src, drn)

            inters_clean, seg_clean, graph_clean = remove_node_loops(
                cur_inter, cur_seg, cur_graph, all_loop_ids)
            res_clean = _run_imp_explicit(inters_clean, graph_clean, seg_clean, k, src, drn)

        R_base, R_loop, R_clean = (res_base['impedance'], res_loop['impedance'],
                                    res_clean['impedance'])
        restored_ok = abs(R_clean - R_base) < 1e-6
        perc_ok = check_percolation_loops(cur_seg, src, drn)

        assert abs(R_loop - R_base) < 1e-3, (
            "R changed {:.2e} (>{:.0e})".format(R_loop - R_base, 1e-3))
        assert restored_ok, (
            "Restored R differs {:.2e} (>{:.0e})".format(R_clean - R_base, 1e-6))
        assert perc_ok, "Connectivity lost"

        with redirect_stdout(io.StringIO()):
            em = setup_experiment()
            loop_desc = "T{}: {} simultaneous node-loops".format(tid, len(loops))
            save_node_loop_experiment(
                em=em, k=k, title='T{} - {}'.format(tid, label),
                intersections=cur_inter, graph=cur_graph,
                seg_baseline=seg, seg_with_loops=cur_seg, seg_cleaned=seg_clean,
                res_baseline=res_base, res_with_loops=res_loop, res_cleaned=res_clean,
                loop_description=loop_desc, loop_node_ids=all_loop_ids,
                src_node=src, drn_node=drn)

            display_and_save_plot(em.experiment_dir, cur_inter,
                seg, cur_seg, seg_clean, res_base, res_loop, res_clean,
                src, drn, topology=topology, title='T{} - {}'.format(tid, label),
                highlight_nodes=all_loop_ids)

        print(_summary_line(tid, 'PASS', R_base, R_loop, R_clean,
                             restored_ok, perc_ok, len(intersections),
                             len(all_loop_ids), extra=label))

    except AssertionError as e:
        print("  T{}: FAIL - {}".format(tid, e))
    except Exception as e:
        print("  T{}: ERROR - {}\n{}".format(tid, e, traceback.format_exc()))
