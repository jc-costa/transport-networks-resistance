# ============================================================================
# RESULT EXPORTER
# Functions for building detailed export data structures
# ============================================================================
import numpy as np
from boundary_utils import validate_strict_boundary_nodes


def _to_serializable(obj):
    """Convert numpy types and tuples to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {_to_serializable(k): _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    return obj


def build_nodes_detailed(filtered_nodes, intersections_points, voltages=None):
    """
    Build detailed node information list.
    
    Args:
        filtered_nodes: List of node indices after filtering
        intersections_points: List of all intersection points [(x, y), ...]
        voltages: Optional voltage array from impedance calculation
    
    Returns:
        List of node detail dictionaries
    """
    nodes_detailed = []
    
    for idx, node in enumerate(filtered_nodes):
        x, y = intersections_points[node]
        node_type = "source" if x < 0.1 else ("drain" if x > 0.9 else "internal")
        voltage = float(voltages[idx]) if voltages is not None and idx < len(voltages) else None
        
        nodes_detailed.append({
            "local_index": idx,
            "global_index": int(node),
            "x": float(x),
            "y": float(y),
            "type": node_type,
            "voltage_V": voltage
        })
    
    return nodes_detailed


def build_edges_detailed(filtered_segment_edge_list, edge_conductance, 
                        voltages=None, currents=None):
    """
    Build detailed edge information list.
    
    Args:
        filtered_segment_edge_list: List of (segment, (n1, n2)) after pruning
        edge_conductance: Dictionary mapping edge keys to conductance values
        voltages: Optional voltage array
        currents: Optional current array
    
    Returns:
        List of edge detail dictionaries
    """
    edges_detailed = []
    
    for idx, (seg, (n1, n2)) in enumerate(filtered_segment_edge_list):
        edge_key = (min(n1, n2), max(n1, n2))
        conductance = edge_conductance.get(edge_key, 1.0)
        current_val = float(currents[idx]) if currents is not None and idx < len(currents) else None
        
        # Calculate voltage drop
        voltage_drop = None
        if voltages is not None:
            v1 = voltages[n1] if n1 < len(voltages) else 0
            v2 = voltages[n2] if n2 < len(voltages) else 0
            voltage_drop = abs(float(v1) - float(v2))
        
        edges_detailed.append({
            "edge_index": idx,
            "segment": int(seg),
            "node1": int(n1),
            "node2": int(n2),
            "conductance_S": float(conductance),
            "resistance_ohms": 1.0 / conductance if conductance > 0 else float('inf'),
            "current_A": current_val,
            "voltage_drop_V": voltage_drop
        })
    
    return edges_detailed


def build_percolating_paths_detailed(all_path_perc, path_nodes, path_pairs_of_edges):
    """
    Build detailed percolating paths information.
    
    Args:
        all_path_perc: List of percolation status (1=percolating)
        path_nodes: List of node lists for each path
        path_pairs_of_edges: List of edge lists for each path
    
    Returns:
        List of percolating path detail dictionaries
    """
    percolating_paths = []
    
    for k, perc in enumerate(all_path_perc):
        if perc == 1:
            path_info = {
                "path_index": k,
                "nodes": [int(n) for n in path_nodes[k]] if k < len(path_nodes) else [],
                "edges": [[int(e[0]), int(e[1])] for e in path_pairs_of_edges[k]] if k < len(path_pairs_of_edges) else [],
                "num_nodes": len(path_nodes[k]) if k < len(path_nodes) else 0,
                "num_edges": len(path_pairs_of_edges[k]) if k < len(path_pairs_of_edges) else 0,
            }
            percolating_paths.append(path_info)
    
    return percolating_paths


def build_lines_info(points_1, points_2, angles=None):
    """
    Build detailed line generation information.
    
    Args:
        points_1: List of start points
        points_2: List of end points
        angles: Optional list of angles
    
    Returns:
        List of line info dictionaries
    """
    lines_info = []
    
    for i in range(len(points_1)):
        lines_info.append({
            "line_index": i,
            "start_point": [float(points_1[i][0]), float(points_1[i][1])],
            "end_point": [float(points_2[i][0]), float(points_2[i][1])],
            "angle_rad": float(angles[i]) if angles is not None and i < len(angles) else None
        })
    
    return lines_info


def calculate_impedance_vs_k(ImpedanceCalculator, merged_intersections, merged_graph, 
                             merged_segment_to_nodes, edge_conductance, 
                             k_values=None):
    """
    Calculate impedance for different k (resistance per segment) values.
    
    Args:
        ImpedanceCalculator: The calculator class
        merged_intersections: Node positions after merge
        merged_graph: Edge list after merge
        merged_segment_to_nodes: Segment to nodes mapping
        edge_conductance: Edge conductance dictionary
        k_values: List of k values to test (default: [1, 10, 100, 1000, 10000])
    
    Returns:
        Dictionary mapping k values to impedance results
    """
    if k_values is None:
        k_values = [1, 10, 100, 1000, 10000]
    
    impedance_vs_k = {}
    
    for k_value in k_values:
        calc_k = ImpedanceCalculator(
            merged_intersections, merged_graph, merged_segment_to_nodes,
            edge_conductance=edge_conductance
        )
        result_k = calc_k.calculate(k=k_value, verbose=False)
        
        impedance_vs_k[k_value] = {
            "k_ohms": k_value,
            "R_equivalent_ohms": float(result_k['impedance']),
            "current_A": float(result_k.get('current', 0)),
        }
    
    return impedance_vs_k


def build_experiment_summary(
    # Basic parameters
    N, a, timestamp, in_colab,
    # Network generation
    intersections_points, lines_info,
    # Percolation
    has_percolation, n_perc, percolating_paths_detailed,
    # Boundary nodes
    source_node_set, drain_node_set,
    # Pruning
    nodes_before, nodes_after, removed_nodes,
    edges_before_set, edges_after_set, removed_edges,
    segments_before=None, segments_after=None, segments_removed=None,
    dead_end_nodes=None,
    # Network after pruning
    nodes_detailed=None, edges_detailed=None,
    # Merge operation
    merged_source_node=None, merged_drain_node=None,
    nodes_merged_to_source=None, nodes_merged_to_drain=None,
    merge_mapping=None, merged_graph=None, merged_intersections=None,
    # Impedance calculation
    R_calc=None, I=None, total_source_current=None,
    impedance_vs_k=None,
    # Electrical solution
    voltages=None, currents=None,
    # Conductance info
    edge_conductance=None, merged_segment_to_nodes=None,
    enforce_strict_boundary=True,
):
    """
    Build comprehensive experiment summary dictionary.
    
    Returns:
        Dictionary with all experiment data
    """
    if enforce_strict_boundary:
        validate_strict_boundary_nodes(
            list(source_node_set) if source_node_set is not None else [],
            list(drain_node_set) if drain_node_set is not None else [],
            context="build_experiment_summary"
        )

    # Build merged graph detailed (use segment_to_nodes which maps edge -> (n1, n2))
    merged_graph_detailed = []
    if merged_segment_to_nodes is not None:
        for seg_idx, (n1, n2) in merged_segment_to_nodes.items():
            merged_graph_detailed.append({
                "node1": int(n1),
                "node2": int(n2)
            })
    
    max_current = None
    if currents is not None and len(currents) > 0:
        max_current = float(np.max(np.abs(currents)))
    
    summary = {
        "timestamp": timestamp,
        "environment": "Google Colab" if in_colab else "Local",
        
        # Parameters
        "parameters": {
            "N": int(N),
            "a": float(a),
            "stick_length": float(a),
            "domain_size": [1.0, 1.0],
            "source_region": "x < 0.1",
            "drain_region": "x > 0.9",
        },
        
        # Network generation stats
        "network_generation": {
            "total_lines": int(N),
            "total_intersections": len(intersections_points) if intersections_points else 0,
            "lines": lines_info or [],
        },
        
        # Percolation analysis
        "percolation_analysis": {
            "has_percolation": has_percolation,
            "num_percolating_paths": int(n_perc) if n_perc else 0,
            "percolating_paths": percolating_paths_detailed or [],
        },
        
        # Boundary nodes (before pruning)
        "boundary_nodes_original": {
            "source_nodes": [int(n) for n in source_node_set] if source_node_set else [],
            "drain_nodes": [int(n) for n in drain_node_set] if drain_node_set else [],
            "num_source_nodes": len(source_node_set) if source_node_set else 0,
            "num_drain_nodes": len(drain_node_set) if drain_node_set else 0,
        },
        
        # Pruning results
        "pruning": {
            "nodes_before": len(nodes_before) if nodes_before else 0,
            "nodes_after": len(nodes_after) if nodes_after else 0,
            "nodes_removed": len(removed_nodes) if removed_nodes else 0,
            "edges_before": len(edges_before_set) if edges_before_set else 0,
            "edges_after": len(edges_after_set) if edges_after_set else 0,
            "edges_removed": len(removed_edges) if removed_edges else 0,
            "segments_before": len(segments_before) if segments_before else 0,
            "segments_after": len(segments_after) if segments_after else 0,
            "segments_removed": list(segments_removed) if segments_removed else [],
            "removed_nodes_list": [int(n) for n in removed_nodes] if removed_nodes else [],
            "removed_edges_list": [[int(e[0]), int(e[1])] for e in removed_edges] if removed_edges else [],
            "dead_end_nodes": [int(n) for n in dead_end_nodes] if dead_end_nodes else [],
        },
        
        # Network after pruning (detailed)
        "network_pruned": {
            "nodes": nodes_detailed or [],
            "edges": edges_detailed or [],
            "num_nodes": len(nodes_detailed) if nodes_detailed else 0,
            "num_edges": len(edges_detailed) if edges_detailed else 0,
        },
        
        # Merge operation
        "merge_operation": {
            "merged_source_node": int(merged_source_node) if merged_source_node is not None else None,
            "merged_drain_node": int(merged_drain_node) if merged_drain_node is not None else None,
            "nodes_merged_to_source": [_to_serializable(n) for n in nodes_merged_to_source] if nodes_merged_to_source else [],
            "nodes_merged_to_drain": [_to_serializable(n) for n in nodes_merged_to_drain] if nodes_merged_to_drain else [],
            "merge_mapping": {str(k): int(v) for k, v in merge_mapping.items()} if merge_mapping else {},
            "merged_graph": merged_graph_detailed,
            "merged_intersections": _to_serializable(merged_intersections) if merged_intersections else [],
        },
        
        # Impedance calculation results
        "impedance_calculation": {
            "k_used_ohms": 1,
            "calculated_impedance_ohms": float(R_calc) if R_calc is not None else None,
            "applied_voltage_V": 1.0,
            "total_current_A": float(I) if I is not None else None,
            "total_source_current_A": float(total_source_current) if total_source_current is not None else None,
        },
        
        # Impedance for different k values
        "impedance_vs_k": impedance_vs_k or {},
        
        # Electrical solution
        "electrical_solution": {
            "voltages": voltages.tolist() if voltages is not None else [],
            "currents": currents.tolist() if currents is not None else [],
            "max_current_A": max_current,
        },
        
        # Conductance matrix info
        "conductance_info": {
            "edge_conductance": {str(k): float(v) for k, v in edge_conductance.items()} if edge_conductance else {},
            "merged_segment_to_nodes": {str(k): [int(n) for n in v] for k, v in merged_segment_to_nodes.items()} if merged_segment_to_nodes else {},
        },
    }
    
    return summary


def print_impedance_summary(impedance_vs_k):
    """Print formatted impedance results for different k values."""
    print(f"\n{'-'*60}")
    print("IMPEDANCE RESULTS FOR DIFFERENT k VALUES:")
    print(f"{'-'*60}")
    for k_val, data in impedance_vs_k.items():
        print(f"  k = {k_val:5d} Ohm  ->  R_equivalent = {data['R_equivalent_ohms']:.4f} Ohm")
    print(f"{'-'*60}")


def print_export_summary(experiment_dir, files_saved):
    """Print formatted export completion summary."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE - DETAILED EXPORT")
    print(f"{'='*60}")
    print(f"Results directory: {experiment_dir}")
    print(f"\nFiles saved:")
    for f in files_saved:
        print(f"  - {f}")
    print(f"{'='*60}")
