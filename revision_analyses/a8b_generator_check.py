"""A8b: the fragment table of A8 recomputed with the pipeline's own generator (utils/point_generator_3d.generate_points_3d,
clipping rule) for a few (a, d) cells. Prints mean kept length / a and the fraction of sticks kept whole."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from point_generator_3d import generate_points_3d
np.random.seed(2026)
for a, d in [(0.2, 0.1), (0.5, 0.1), (1.1, 0.1), (0.2, 0.3), (0.5, 1.0)]:
    p1, p2, dirs = [], [], []
    N = int(40000 / ((1 + a) ** 2 * (d + a) / d))
    generate_points_3d(N, p1, p2, dirs, a, box_dims=(1.0, 1.0, d))
    L = np.linalg.norm(np.asarray(p2) - np.asarray(p1), axis=1) / a
    print(f"a={a} d={d}: n={len(L)} mean kept/a={L.mean():.2f} whole={np.mean(L > 1 - 1e-9):.2f}")
