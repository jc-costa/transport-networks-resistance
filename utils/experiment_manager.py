# ============================================================================
# EXPERIMENT MANAGER
# Handles experiment configuration, directory management, and file saving
# Works both locally and on Google Colab
# ============================================================================
import os
import re
import json
import numpy as np
from datetime import datetime


class ExperimentManager:
    """
    Manages experiment directories and file saving.
    Automatically creates numbered experiment folders (exp_1, exp_2, etc.)
    """
    
    def __init__(self, base_dir=None, colab_base_dir="/content/drive/MyDrive/ColabProject/experiments"):
        """
        Initialize experiment manager.
        
        Args:
            base_dir: Base directory for local experiments. If None, auto-detect.
            colab_base_dir: Base directory for Colab experiments.
        """
        self.in_colab = self._detect_colab()
        
        if self.in_colab:
            self.base_dir = colab_base_dir
        elif base_dir:
            self.base_dir = base_dir
        else:
            # Default to experiments folder relative to this file
            self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Get next experiment number and create folder
        self.experiment_number = self._get_next_experiment_number()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(self.base_dir, f"exp_{self.experiment_number}")
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def _detect_colab(self):
        """Detect if running on Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _get_next_experiment_number(self):
        """Find the highest experiment number and return the next one."""
        existing = os.listdir(self.base_dir) if os.path.exists(self.base_dir) else []
        max_num = 0
        pattern = re.compile(r'^exp_(\d+)$')
        for folder in existing:
            match = pattern.match(folder)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        return max_num + 1
    
    def save_figure(self, fig, filename, dpi=150):
        """
        Save a matplotlib figure to the experiment directory.
        
        Args:
            fig: Matplotlib figure object
            filename: Name for the saved file
            dpi: Resolution (default 150)
        
        Returns:
            Full filepath of saved figure
        """
        filepath = os.path.join(self.experiment_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"  [OK] Figure saved: {filename}")
        return filepath
    
    def save_json(self, data, filename):
        """
        Save data as JSON to the experiment directory.
        
        Args:
            data: Data to save (dict or list)
            filename: Name for the JSON file
        
        Returns:
            Full filepath of saved file
        """
        filepath = os.path.join(self.experiment_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=self._to_serializable)
        print(f"  [OK] JSON saved: {filename}")

        # Keep PDF reports in sync with completed experiments.
        if filename == "experiment_summary.json":
            try:
                from pdf_generator import generate_all_pdfs
                generate_all_pdfs(self.experiment_dir, self.base_dir)
            except Exception as e:
                print(f"  [WARNING] PDF auto-generation failed: {e}")

        return filepath
    
    @staticmethod
    def _to_serializable(obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)
    
    def print_config(self):
        """Print experiment configuration summary."""
        print(f"{'='*60}")
        print("EXPERIMENT CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Environment: {'Google Colab' if self.in_colab else 'Local'}")
        print(f"  Experiment #: {self.experiment_number}")
        print(f"  Timestamp: {self.timestamp}")
        print(f"  Results directory: {self.experiment_dir}")
        print(f"{'='*60}\n")
    
    def get_info(self):
        """Return experiment info as dictionary."""
        return {
            "experiment_number": self.experiment_number,
            "timestamp": self.timestamp,
            "environment": "Google Colab" if self.in_colab else "Local",
            "experiment_dir": self.experiment_dir,
            "base_dir": self.base_dir,
        }

    def save_network(self, N, a, points_1, points_2, angles, filename="network_data.json"):
        """
        Save the generated network data so it can be reloaded later.

        Args:
            N: Number of sticks
            a: Aspect ratio (stick length)
            points_1: List of start points, 2D `(x, y)` or 3D `(x, y, z)`
            points_2: List of end points, same shape as points_1
            angles: List of orientations, one per stick -- a scalar angle
                (radians) for 2D sticks, or a direction vector (e.g. `(dx,
                dy, dz)`) for 3D sticks
            filename: Name for the saved file (default "network_data.json",
                the name load_network() expects; pass a custom name when
                saving more than one network in the same experiment folder)

        Returns:
            Full filepath of saved file
        """
        network_data = {
            "N": int(N),
            "a": float(a),
            "lines": []
        }
        for i in range(len(points_1)):
            line = {
                "start_point": [float(c) for c in points_1[i]],
                "end_point": [float(c) for c in points_2[i]],
            }
            orientation = angles[i]
            if np.isscalar(orientation):
                line["angle_rad"] = float(orientation)
            else:
                line["direction"] = [float(c) for c in orientation]
            network_data["lines"].append(line)
        return self.save_json(network_data, filename)


def load_network(experiment_number, base_dir=None,
                 colab_base_dir="/content/drive/MyDrive/ColabProject/experiments"):
    """
    Load a previously saved network from a given experiment number.

    Args:
        experiment_number: Integer experiment number (e.g. 5 loads exp_5/)
        base_dir: Base experiments directory (auto-detected if None)
        colab_base_dir: Base directory used on Google Colab

    Returns:
        dict with keys: N, a, points_1, points_2, angles, experiment_dir
    """
    # Detect environment
    try:
        import google.colab  # noqa: F401
        resolved_base = colab_base_dir
    except ImportError:
        if base_dir:
            resolved_base = base_dir
        else:
            resolved_base = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "experiments"
            )

    exp_dir = os.path.join(resolved_base, f"exp_{experiment_number}")
    network_file = os.path.join(exp_dir, "network_data.json")

    if not os.path.isfile(network_file):
        raise FileNotFoundError(
            f"Network data not found at {network_file}. "
            f"Experiment {experiment_number} may not have saved network data."
        )

    with open(network_file, 'r') as f:
        data = json.load(f)

    N = data["N"]
    a = data["a"]
    points_1 = [tuple(line["start_point"]) for line in data["lines"]]
    points_2 = [tuple(line["end_point"]) for line in data["lines"]]
    angles = [line["angle_rad"] if "angle_rad" in line else tuple(line["direction"])
              for line in data["lines"]]

    print(f"{'='*60}")
    print(f"LOADED NETWORK FROM EXPERIMENT {experiment_number}")
    print(f"{'='*60}")
    print(f"  Directory: {exp_dir}")
    print(f"  N = {N}, a = {a}")
    print(f"  {len(points_1)} segments loaded")
    print(f"{'='*60}\n")

    return {
        "N": N,
        "a": a,
        "points_1": points_1,
        "points_2": points_2,
        "angles": angles,
        "experiment_dir": exp_dir,
    }


# Convenience function for quick setup
def setup_experiment(base_dir=None):
    """
    Quick setup for experiment management.
    
    Returns:
        ExperimentManager instance
    """
    exp = ExperimentManager(base_dir=base_dir)
    exp.print_config()
    return exp
