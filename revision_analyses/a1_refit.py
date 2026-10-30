"""A1: dimensionally consistent refit of the pooled 2D sweeps (revision plan, block H1),
plus the C3 duplicate check. Pools exactly as cluster/fit_figures.py."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster"))
import fit_figures as ff

N, a, R = ff.pool()
y = np.log10(R); x = N * a**2
# --- C3: duplicated trials (baseline cell values that are a prefix of the extended cell) ---
import glob, re
paths = sorted(glob.glob(os.path.join(ff.EXPERIMENTS_DIR, "exp_*", "effective_resistance_sweep_summary*.json")))
cells = {}
for p in paths:
    if int(re.search(r"exp_(\d+)", p).group(1)) < ff.MERGE_FIX_FIRST_EXPERIMENT: continue
    s = json.load(open(p))
    if s.get("config", {}).get("boundary") != "clip": continue
    for rec in s.get("sweep", []):
        v = rec.get("R_eff_values") or []
        if v: cells.setdefault((rec["N"], round(rec["a"], 3)), []).append((os.path.basename(os.path.dirname(p)), len(v), v))
dups = {}
for k, lst in cells.items():
    if len(lst) > 1:
        lst = sorted(lst, key=lambda t: t[1])
        short, long_ = lst[0][2], lst[-1][2]
        if len(short) < len(long_) and all(abs(u - w) <= 1e-9 * abs(w) for u, w in zip(short, long_)):
            dups[k] = short
print("cells pooled from more than one summary:", {k: [(e, n) for e, n, _ in v] for k, v in cells.items() if len(v) > 1})
print("cells whose shorter list repeats the start of the longer one:", sorted(dups), "->", sum(len(v) for v in dups.values()), "duplicated trials")
keep = np.ones(len(R), bool)
for k, short in dups.items():
    for val in short:
        idx = np.flatnonzero((N == k[0]) & (np.round(a, 3) == k[1]) & np.isclose(R, val, rtol=1e-12))
        if len(idx) >= 2: keep[idx[-1]] = False   # drop one copy
print("trials after removing duplicates:", keep.sum(), "of", len(R))

def ols(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
    n, k = X.shape; sse = r @ r
    return c, 1 - sse / np.sum((y - y.mean())**2), n * np.log(sse / n) + 2 * k, n * np.log(sse / n) + k * np.log(n)

def all_fits(N, a, y, label):
    x = N * a**2; one = np.ones_like(y)
    c_col, r2c, aic_c, bic_c = ols(np.c_[one, np.log10(x)], y)
    c_gen, r2g, aic_g, bic_g = ols(np.c_[one, np.log10(N), np.log10(a)], y)
    c_a1, r2a, aic_a, bic_a = ols(np.c_[one, np.log10(x)], y - np.log10(a))
    r2a = 1 - np.sum((y - np.log10(a) - np.c_[one, np.log10(x)] @ c_a1)**2) / np.sum((y - y.mean())**2)
    p, q = c_gen[1], c_gen[2]
    print(f"[{label}] n={len(y)}")
    print(f"  collapse   R ~ (Na^2)^{c_col[1]:.3f}                 R2={r2c:.4f}  AIC={aic_c:.0f}  BIC={bic_c:.0f}")
    print(f"  a^1 fixed  R ~ a (Na^2)^{c_a1[1]:.3f}  A={10**c_a1[0]:.0f}   R2={r2a:.4f}  AIC={aic_a:.0f}  BIC={bic_a:.0f}")
    print(f"  general    R ~ N^{p:.3f} a^{q:.3f}  A={10**c_gen[0]:.0f}  R2={r2g:.4f}  AIC={aic_g:.0f}  BIC={bic_g:.0f}   q-2p={q-2*p:.3f}")
    return q - 2 * p

k0 = all_fits(N, a, y, "pooled as in the thesis")
k1 = all_fits(N[keep], a[keep], y[keep], "duplicates removed")
# bootstrap over (N, a) cells for q-2p
rng = np.random.default_rng(2026)
Nk, ak, yk = N[keep], a[keep], y[keep]
keys = np.array([f"{n}_{aa:.3f}" for n, aa in zip(Nk, ak)]); uk = np.unique(keys)
groups = [np.flatnonzero(keys == u) for u in uk]
kap = []
for _ in range(2000):
    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    X = np.c_[np.ones(len(idx)), np.log10(Nk[idx]), np.log10(ak[idx])]
    c, *_ = np.linalg.lstsq(X, yk[idx], rcond=None); kap.append(c[2] - 2 * c[1])
lo, hi = np.percentile(kap, [2.5, 97.5])
print(f"q-2p (duplicates removed) = {k1:.3f}, 95% bootstrap over {len(uk)} cells: [{lo:.3f}, {hi:.3f}]")
