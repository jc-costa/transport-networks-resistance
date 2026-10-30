"""All fits of Tables fit-results / fit-critical on the deduplicated pool, and the effect on the inversion candidates."""
import sys, os, io, contextlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster"))
import fit_figures as ff
with contextlib.redirect_stdout(io.StringIO()):
    N, a, R = ff.pool()
dup = {(30, 0.6), (30, 0.8), (45, 0.6), (45, 0.8), (60, 0.8)}
keep = np.ones(len(R), bool)
for k in dup:
    idx = np.flatnonzero((N == k[0]) & (np.round(a, 3) == k[1]))
    vals, seen = R[idx], {}
    for i, v in zip(idx, vals):
        key = float(f"{v:.9g}")
        if key in seen: keep[i] = False
        seen[key] = 1
print("kept", keep.sum(), "of", len(R))
for lab, m in (("with duplicates", np.ones(len(R), bool)), ("deduplicated", keep)):
    print("==", lab); f = ff.fits(N[m], a[m], R[m])
    x = N[m] * a[m] ** 2; sel = x > ff.C_INF
    from scipy.optimize import least_squares
    lR = np.log10(R[m])[sel]; aa = a[m][sel]; xx = x[sel]
    res = least_squares(lambda p: p[0] + p[1] * np.log10(aa) - p[2] * np.log10(xx - ff.C_INF) - lR, [3, 1, 1])
    pred = res.x[0] + res.x[1] * np.log10(aa) - res.x[2] * np.log10(xx - ff.C_INF)
    print(f"pinned c=5.64 on {sel.sum()} trials: A={10**res.x[0]:.0f} q={res.x[1]:.3f} t={res.x[2]:.3f} R2={ff.r2(lR, pred):.3f}")
    X = np.c_[np.ones(m.sum()), np.log10(x)]; c, *_ = np.linalg.lstsq(X, np.log10(R[m]) - np.log10(a[m]), rcond=None)
    pr = X @ c + np.log10(a[m]); print(f"dense limit check skipped; a^1-fixed: A={10**c[0]:.0f} s={c[1]:.3f} R2={ff.r2(np.log10(R[m]), pr):.4f}")
    if lab == "deduplicated":
        g = f["general"]
        print("candidate N with the deduplicated general law (old -> new):")
        for T in (26, 31, 40, 73, 150, 460, 890):
            row = []
            for aa_ in (0.2, 0.4, 0.7, 1.0):
                old = round((28254 * aa_ ** -2.459 / T) ** (1 / 1.694)); new = round((g["A"] * aa_ ** g["q"] / T) ** (-1 / g["p"]))
                row.append(f"a={aa_}: {old}->{new}")
            print(f"  T={T}: " + ", ".join(row))
