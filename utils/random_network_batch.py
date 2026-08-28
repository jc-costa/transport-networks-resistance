"""
Batch pipeline for running many random stick networks through the full
percolation -> pruning -> boundary-merge -> impedance workflow used in
PiaNo.ipynb / test_random_network_impedance.ipynb.

Extracted from test_150_random_networks.ipynb so the notebook stays a thin
driver: generate configs, run the batch, look at the table/plots.
"""
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from point_generator import generate_points
from intersection_utils import identify_intersections
from path_analysis import (
    find_path, build_connected_path, rebuild_local_graph, verify_pruned_percolation,
)
from plotting import plot_percolating_paths
from network_visualization import plot_before_after_pruning, plot_before_after_merge
from pruning import prune_dead_end_nodes, build_pruned_structures
from edge_discovery import discover_consecutive_edges, check_dead_ends
from impedance_calculator import ImpedanceCalculator, merge_boundary_nodes

MAX_INTERSECTIONS_SAFE = 4000  # defensive cap; well above what N<=90, a<=0.80 produce


def generate_diversified_configs(n_cases, n_min, n_max, a_min, a_max,
                                  param_seed=2025, case_seed_base=10_000):
    """Build `n_cases` reproducible (N, a, seed) configurations.

    `param_seed` seeds the parameter grid itself; `case_seed_base + case_id`
    seeds the network generation for each individual case.
    """
    param_rng = random.Random(param_seed)
    configs = []
    for case_id in range(n_cases):
        N_i = param_rng.randint(n_min, n_max)
        a_i = round(param_rng.uniform(a_min, a_max), 3)
        configs.append({'case_id': case_id, 'N': N_i, 'a': a_i, 'seed': case_seed_base + case_id})
    return configs


def run_full_network_case(cfg, k=1000.0, plot=True, verbose=False,
                           max_intersections_safe=MAX_INTERSECTIONS_SAFE):
    """Run one (N, a, seed) configuration through the full pipeline.

    Percolation check -> (pruning, only if dead-ends are found) ->
    boundary-node merge (source nodes collapse into one, drain nodes into
    another) -> impedance solve. Returns `(row, example)`:
    - `row`: flat summary dict, one row of the results table.
    - `example`: raw intermediate data, enough to replot/reanalyze the case
      later with `analyze_case` (or `None` if point generation failed).

    When `plot=True`, before/after-pruning and before/after-merge figures
    are shown automatically for every case that actually reaches those
    steps (pruning only when dead-ends were found; merge for every solved
    case).
    """
    case_id, N, a, seed = cfg['case_id'], cfg['N'], cfg['a'], cfg['seed']
    random.seed(seed)
    np.random.seed(seed)

    row = {
        'case_id': case_id, 'N': N, 'a': a, 'seed': seed,
        'n_intersections': 0, 'has_percolation': False,
        'needed_pruning': False, 'nodes_removed': 0, 'edges_removed': 0,
        'proceeds_after_pruning': False, 'n_final_nodes': 0, 'n_final_edges': 0,
        'effective_resistance': float('inf'), 'is_singular': None,
        'condition_number': float('nan'), 'status': 'ok',
    }

    points_1, points_2, angles, graph = [], [], [], []
    intersections_points, intersected_line_1, intersected_line_2 = [], [], []

    try:
        generate_points(N, points_1, points_2, angles, a)
    except (RuntimeError, ValueError) as exc:
        row['status'] = f'generation_failed: {exc}'
        return row, None

    intersections_points, intersected_line_1, intersected_line_2, graph = identify_intersections(
        points_1, points_2, intersections_points, intersected_line_1,
        intersected_line_2, graph, angles, a, N
    )
    row['n_intersections'] = len(intersections_points)

    example = dict(
        points_1=points_1, points_2=points_2, angles=angles,
        intersected_line_1=intersected_line_1, intersected_line_2=intersected_line_2,
        intersections_points=intersections_points,
    )

    if len(intersections_points) > max_intersections_safe:
        row['status'] = f'skipped_too_dense ({len(intersections_points)} intersections)'
        return row, example

    boundary_condition_source, boundary_condition_drain = [], []
    for i, (x, y) in enumerate(intersections_points):
        if x < 0.1:
            boundary_condition_source.append(i)
        elif x > 0.9:
            boundary_condition_drain.append(i)

    if not boundary_condition_source or not boundary_condition_drain:
        row['status'] = 'no_boundary_nodes'
        return row, example

    # Find all paths starting from each not-yet-covered source node
    # (identical logic to PiaNo.ipynb / test_new_self_loops_random_networks.ipynb)
    path_nodes, path_pairs_of_lines, path_pairs_of_edges, all_path_perc = [], [], [], []
    for i in boundary_condition_source:
        do_not_check = 0
        number_path = len(path_nodes)
        for kk in range(number_path):
            if i in path_nodes[kk]:
                do_not_check = 1
        if do_not_check == 0:
            nodes, pair_of_lines, pair_of_edges, path_perc = find_path(intersections_points, graph, i)
            path_nodes.append(nodes)
            path_pairs_of_lines.append(pair_of_lines)
            path_pairs_of_edges.append(pair_of_edges)
            all_path_perc.append(path_perc)

    example.update(path_nodes=path_nodes, path_pairs_of_edges=path_pairs_of_edges, all_path_perc=all_path_perc)

    has_percolation = any(p == 1 for p in all_path_perc)
    row['has_percolation'] = has_percolation

    if not has_percolation:
        row['status'] = 'no_percolating_path'
        return row, example

    # Plot the percolating path(s) -- same function used at the end of PiaNo.ipynb
    if plot:
        plot_percolating_paths(N, intersections_points, path_nodes, path_pairs_of_edges, all_path_perc)
        plt.close('all')  # avoid accumulating open figures in memory

    # Boundary segments (sticks touching the source/drain regions)
    source_segments, drain_segments = set(), set()
    for node_idx in boundary_condition_source:
        source_segments.update(graph[node_idx])
    for node_idx in boundary_condition_drain:
        drain_segments.update(graph[node_idx])

    # All edges of the percolating component(s) (consecutive-nodes-on-a-stick logic)
    perc_indices = [idx for idx, p in enumerate(all_path_perc) if p == 1]
    all_perc_nodes_set = set()
    for idx in perc_indices:
        all_perc_nodes_set.update(path_nodes[idx])

    segment_edge_list = discover_consecutive_edges(
        graph, intersections_points, all_perc_nodes_set, points_1, points_2
    )

    # Apply pruning / removal ONLY when dangling (dead-end) bonds are found
    dangling_intern, has_dead_ends = check_dead_ends(segment_edge_list, source_segments, drain_segments)
    row['needed_pruning'] = has_dead_ends

    if has_dead_ends:
        filtered_segment_edge_list, _, _ = prune_dead_end_nodes(
            segment_edge_list, intersections_points, verbose=verbose
        )
    else:
        filtered_segment_edge_list = segment_edge_list

    example.update(segment_edge_list=segment_edge_list, filtered_segment_edge_list=filtered_segment_edge_list)

    _, filtered_nodes, _, stats = build_pruned_structures(segment_edge_list, filtered_segment_edge_list)
    row['nodes_removed'] = len(stats['removed_nodes'])
    row['edges_removed'] = len(stats['removed_edges'])

    proceed = verify_pruned_percolation(
        filtered_segment_edge_list, intersections_points, source_segments, drain_segments
    )
    row['proceeds_after_pruning'] = proceed

    # Automatic before/after-pruning visual analysis -- only for the cases
    # that actually needed pruning (dead-end bonds were found), so the
    # notebook output stays readable; every other case can still be
    # inspected later on demand with analyze_case(case_id).
    if has_dead_ends and plot:
        fig_prune, _ = plot_before_after_pruning(
            segment_edge_list, filtered_segment_edge_list, intersections_points, N, figsize=(12, 6)
        )
        if fig_prune:
            fig_prune.suptitle(
                f"Case {case_id}: N={N}  a={a:.3f}  -- still percolates after pruning: {proceed}",
                y=1.03,
            )
            plt.show()
        plt.close('all')

    if not proceed or not filtered_segment_edge_list:
        row['status'] = 'pruned_to_disconnection'
        return row, example

    # Rebuild a compact local graph for the connected source<->drain component
    segment_to_nodes, filtered_nodes = build_connected_path(
        filtered_segment_edge_list, filtered_nodes, intersections_points
    )
    node_mapping, path_intersections, new_segment_to_nodes, path_graph = rebuild_local_graph(
        filtered_nodes, intersections_points, segment_to_nodes
    )

    source_nodes_pruned = [n for n in filtered_nodes if intersections_points[n][0] < 0.1]
    drain_nodes_pruned = [n for n in filtered_nodes if intersections_points[n][0] > 0.9]
    source_boundary_new = [node_mapping[n] for n in source_nodes_pruned if n in node_mapping]
    drain_boundary_new = [node_mapping[n] for n in drain_nodes_pruned if n in node_mapping]

    if not source_boundary_new or not drain_boundary_new:
        row['status'] = 'pruned_boundary_lost'
        return row, example

    # Merge ALL source-boundary nodes into one physical node, and ALL
    # drain-boundary nodes into another -- same merge_boundary_nodes used
    # in test_random_network_impedance.ipynb / test_pruning_examples.ipynb /
    # pipeline_utils.run_full_pipeline. This always runs once a network
    # reaches this point (percolates + survives pruning); when there is
    # already just one source/drain node it is effectively a no-op.
    (merged_intersections, merged_graph, merged_segment_to_nodes,
     merged_source_node, merged_drain_node, merge_mapping, edge_conductance) = merge_boundary_nodes(
        path_intersections, path_graph, new_segment_to_nodes, source_boundary_new, drain_boundary_new
    )

    calc = ImpedanceCalculator(
        merged_intersections, merged_graph, merged_segment_to_nodes, edge_conductance=edge_conductance
    )
    with np.errstate(all='ignore'):
        result = calc.calculate(k=k, verbose=verbose)

    row['n_final_nodes'] = len(merged_intersections)
    row['n_final_edges'] = len(merged_segment_to_nodes)
    row['effective_resistance'] = result['impedance']
    row['is_singular'] = bool(result['is_singular'])
    row['condition_number'] = float(result['condition_number'])
    row['status'] = 'ok'

    example.update(
        result=result, merged_intersections=merged_intersections,
        merged_source_node=merged_source_node, merged_drain_node=merged_drain_node,
        merged_segment_to_nodes=merged_segment_to_nodes, edge_conductance=edge_conductance,
        filtered_nodes=filtered_nodes, k=k,
        n_source_merged=len(source_boundary_new), n_drain_merged=len(drain_boundary_new),
    )

    # Automatic before/after-merge visual analysis -- runs for every case
    # that actually reaches the merge step (i.e. every solved case), showing
    # the network right after source/drain nodes were collapsed, colored by
    # node voltage and branch current.
    if plot:
        fig_merge = plot_before_after_merge(
            filtered_segment_edge_list, filtered_nodes, intersections_points,
            merged_intersections, merged_segment_to_nodes, merged_source_node,
            merged_drain_node, edge_conductance, result, k,
        )
        if fig_merge:
            fig_merge.suptitle(
                f"Case {case_id}: N={N}  a={a:.3f}  -- "
                f"{len(source_boundary_new)} source node(s) / {len(drain_boundary_new)} drain node(s) merged  "
                f"R={result['impedance']:.3f} Ohm",
                y=1.03,
            )
            plt.show()
        plt.close('all')

    return row, example


def run_batch(configs, k=1000.0, plot=True, verbose=False):
    """Run `run_full_network_case` for every config; return (results, examples)."""
    results = []
    examples = {}
    for cfg in configs:
        row, example = run_full_network_case(cfg, k=k, plot=plot, verbose=verbose)
        results.append(row)
        if example is not None:
            examples[cfg['case_id']] = example

        singular_label = 'N/A' if row['is_singular'] is None else ('YES' if row['is_singular'] else 'no')
        print(
            f"Case {row['case_id']:3d}: N={row['N']:3d} a={row['a']:.3f}  "
            f"intersections={row['n_intersections']:5d}  percolates={str(row['has_percolation']):5s}  "
            f"pruned={str(row['needed_pruning']):5s}  R={row['effective_resistance']:>10.3f} Ohm  "
            f"singular={singular_label:3s}  [{row['status']}]"
        )

    print(f"\nFinished {len(results)} networks.")
    return results, examples


def build_results_dataframe(results):
    """Turn the list of `row` dicts from `run_batch` into a display-ready DataFrame."""
    df = pd.DataFrame(results)
    df['singular_label'] = df['is_singular'].map({True: 'YES', False: 'no'}).fillna('N/A')
    return df


def summarize_batch(df):
    """Print the aggregate percolation/pruning/singularity summary for a batch.

    Returns `(stats, singular_df)`: `stats` is a dict of the printed numbers,
    `singular_df` is the (possibly empty) slice of `df` with singular `M` --
    call `display(singular_df)` in the notebook if it's non-empty.
    """
    n_total = len(df)
    n_percolating = int(df['has_percolation'].sum())
    n_pruned = int(df['needed_pruning'].sum())
    n_solved = int((df['status'] == 'ok').sum())
    n_singular = int((df['is_singular'] == True).sum())
    n_nonsingular = int((df['is_singular'] == False).sum())
    n_not_applicable = n_total - n_solved

    print("=" * 70)
    print(f"SUMMARY OVER {n_total} RANDOM NETWORKS")
    print("=" * 70)
    print(f"Total networks generated:                  {n_total}")
    print(f"Percolating (source-drain path exists):     {n_percolating} ({100 * n_percolating / n_total:.1f}%)")
    print(f"Required pruning/removal of dead-ends:      {n_pruned} ({100 * n_pruned / n_total:.1f}%)")
    print(f"Successfully solved for V and R:             {n_solved} ({100 * n_solved / n_total:.1f}%)")
    print()
    print("Singularity of the reduced Laplacian matrix M (answered for every example):")
    print(f"  Singular   (cond(M) > 1e10):    {n_singular}")
    print(f"  Non-singular (invertible):      {n_nonsingular}")
    print(f"  N/A (no linear system solved):  {n_not_applicable}")

    stats = {
        'n_total': n_total, 'n_percolating': n_percolating, 'n_pruned': n_pruned,
        'n_solved': n_solved, 'n_singular': n_singular, 'n_nonsingular': n_nonsingular,
        'n_not_applicable': n_not_applicable,
    }

    solved = df[df['status'] == 'ok']
    if len(solved):
        print()
        print("Effective resistance statistics (solved cases only):")
        print(f"  min    = {solved['effective_resistance'].min():.3f} Ohm")
        print(f"  median = {solved['effective_resistance'].median():.3f} Ohm")
        print(f"  max    = {solved['effective_resistance'].max():.3f} Ohm")
        stats.update(
            r_min=float(solved['effective_resistance'].min()),
            r_median=float(solved['effective_resistance'].median()),
            r_max=float(solved['effective_resistance'].max()),
        )

    if n_singular:
        print("\nSingular-matrix cases:")
        singular_df = df.loc[df['is_singular'] == True,
                              ['case_id', 'N', 'a', 'n_intersections', 'condition_number', 'status']]
    else:
        print("\nNo singular matrices were found among the solved networks.")
        singular_df = df.iloc[0:0]

    return stats, singular_df


def plot_percolation_overview(df, k_ohm, figsize=(13, 5)):
    """Two-panel overview: which (N, a) percolated, and R vs. a for solved cases."""
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    colors = df['has_percolation'].map({True: 'seagreen', False: 'lightgray'})
    ax[0].scatter(df['a'], df['N'], c=colors, s=40, edgecolors='black', linewidths=0.3)
    ax[0].set_xlabel('a (stick length)')
    ax[0].set_ylabel('N (number of sticks)')
    ax[0].set_title('Percolation over the (N, a) configurations\n(green = percolates, gray = does not)')

    solved = df[df['status'] == 'ok']
    singular_mask = solved['is_singular'] == True
    ax[1].scatter(solved.loc[~singular_mask, 'a'], solved.loc[~singular_mask, 'effective_resistance'],
                  c='steelblue', s=30, alpha=0.7, label='non-singular M')
    ax[1].scatter(solved.loc[singular_mask, 'a'], solved.loc[singular_mask, 'effective_resistance'],
                  c='crimson', marker='x', s=70, label='singular M')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('a (stick length)')
    ax[1].set_ylabel('Effective resistance R (Ohm, log scale)')
    ax[1].set_title(f'Effective resistance vs. a\n(solved cases; k = {k_ohm:.0f} Ohm/segment)')
    ax[1].legend()

    plt.tight_layout()
    return fig


def analyze_case(case_id, results, examples, plot_pruning=True, plot_circuit=True):
    """Re-inspect any case from `run_batch` without re-running the pipeline.

    Prints the percolation/pruning/singularity summary, and re-plots the
    before/after-pruning and before/after-merge figures from data already
    stored in `examples[case_id]` (whether or not they were shown
    automatically during the original `run_batch` call).
    """
    row = next((r for r in results if r['case_id'] == case_id), None)
    example = examples.get(case_id)

    if row is None or example is None:
        print(f"Case {case_id}: no data available (status is likely 'generation_failed').")
        return

    singular_label = 'N/A' if row['is_singular'] is None else ('YES' if row['is_singular'] else 'no')

    print(f"Case {row['case_id']}: N={row['N']}  a={row['a']:.3f}  seed={row['seed']}")
    print(f"  status:                     {row['status']}")
    print(f"  percolates (raw):           {row['has_percolation']}")
    print(f"  needed pruning:             {row['needed_pruning']}  "
          f"(nodes removed={row['nodes_removed']}, edges removed={row['edges_removed']})")
    print(f"  percolates after pruning:   {row['proceeds_after_pruning']}")
    print(f"  singular M:                 {singular_label}"
          + (f"  (cond={row['condition_number']:.3e})" if row['is_singular'] is not None else ""))
    if row['status'] == 'ok':
        print(f"  effective resistance:       {row['effective_resistance']:.4f} Ohm")

    if not row['has_percolation']:
        print("  -> No percolating path for this (N, a, seed); nothing to plot.")
        return

    if plot_pruning and 'filtered_segment_edge_list' in example:
        fig_prune, _ = plot_before_after_pruning(
            example['segment_edge_list'], example['filtered_segment_edge_list'],
            example['intersections_points'], row['N'], figsize=(12, 6)
        )
        if fig_prune:
            fig_prune.suptitle(
                f"Case {case_id}: pruning (needed={row['needed_pruning']}, "
                f"still percolates={row['proceeds_after_pruning']})",
                y=1.03,
            )
            plt.show()
        plt.close('all')

    if plot_circuit:
        if row['status'] == 'ok':
            fig_merge = plot_before_after_merge(
                example['filtered_segment_edge_list'], example['filtered_nodes'],
                example['intersections_points'], example['merged_intersections'],
                example['merged_segment_to_nodes'], example['merged_source_node'],
                example['merged_drain_node'], example['edge_conductance'],
                example['result'], example['k'],
            )
            if fig_merge:
                plt.show()
            plt.close('all')
        else:
            print(f"  -> No solved circuit to visualize (status='{row['status']}').")
