"""A7 (exact case): deterministic network through the full 2D pipeline, solvable by hand.
m horizontal sticks of length 1 at y_k = (k - 1/2)/m, and two slanted sticks, one inside each
electrode strip: x_s(y) = 0.03 + 0.04 y and x_d(y) = 0.97 - 0.04 y. Every horizontal stick is
then a single wire of length L_k = 0.94 - 0.08 y_k between the two terminals, the slanted
sticks are shorted inside the strips, and the m wires are parallel edges that the condensation
collapses into one. Exact values: R = k/m (uniform model), R = 1 / sum_k 1/(R0 L_k) (distance model).
(Exactly perpendicular sticks are avoided: the crossing routine returns no intersection for a
relative angle of exactly pi/2, which random orientations never produce.)"""
import sys, os, io, contextlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from pipeline_utils import run_full_pipeline, distance_dependent_impedance
R0 = k = 1000.0
for m in (10, 25):
    ys = (np.arange(m) + 0.5) / m
    p1 = [(0.0, y) for y in ys]; p2 = [(1.0, y) for y in ys]; ang = [0.0] * m
    for (x0, x1) in ((0.03, 0.07), (0.97, 0.93)):
        p1.append((x0, 0.0)); p2.append((x1, 1.0)); ang.append(float(np.arctan2(1.0, x1 - x0)))
    a_nom = max(np.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(p1, p2))
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_full_pipeline(p1, p2, ang, len(p1), a_nom, k=k, verbose=False)
    if not res.get("has_percolation"):
        print(f"m={m}: no percolation found"); continue
    L = 0.94 - 0.08 * ys
    exact_uni, exact_dd = k / m, 1.0 / np.sum(1.0 / (R0 * L))
    R_uni = float(res["impedance"]); R_dd = float(distance_dependent_impedance(res, R0)["impedance"])
    print(f"m={m}: crossings={len(res['intersections_points'])} (expected {2*m}); "
          f"uniform R={R_uni:.10g} exact {exact_uni:.10g} rel.err {abs(R_uni-exact_uni)/exact_uni:.1e}; "
          f"distance-dependent R={R_dd:.10g} exact {exact_dd:.10g} rel.err {abs(R_dd-exact_dd)/exact_dd:.1e}")
