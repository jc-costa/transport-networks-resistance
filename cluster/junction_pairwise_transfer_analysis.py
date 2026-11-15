"""
Pre-registered analysis of the transfer run (PREREGISTRATION_JUNCTION_TRANSFER.md). Everything that
decides the outcome is fixed in that file and in the constants below; nothing here may change after the run.

The estimators are those frozen from the dense grid (figures/data/junction_estimator_frozen.json); they
are applied unchanged to the new films. Rows are (geometry, network, junction rule, rho); the four values of
rho of a network are drawn at random and are the same for the three rules.

Usage: python junction_pairwise_transfer_analysis.py <partial.jsonl> <frozen.json> [--boot 2000] [--out summary.json]
"""
import sys, json
import numpy as np
import junction_pairwise_review_checks as rc
import junction_pairwise_regime_analysis as rg

SEED, BOOT = 0, 2000
FULL, PHYS = (0.03, 100.0), (1.0, 30.0)
RMSE_MARGIN, GAIN_MARGIN = 0.35, 0.03
COVER_LO, COVER_HI = 0.80, 0.95
R0_SIGMA, R0_RMSE_MARGIN = 0.30, 0.40
ESTS = ("S", "S+level", "S+P")


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        for k, net in enumerate(r["networks"]):
            for b in net["by_cond"]:
                if b["rho"] > 0:
                    rows.append(dict(cell=r["id"], net=k, a=r["a"], x=r["x"], variant=b["variant"], rho=b["rho"],
                                     S=b["S"], P=b["P"]))
    return rows


def predict(frozen, name, S, P):
    e = frozen["estimators"][name]
    X = np.log10(S)[:, None]
    if name == "S+level":
        X = np.hstack([X, np.log10(P).mean(1, keepdims=True)])
    elif name == "S+P":
        X = np.hstack([X, np.log10(P)])
    Z = (X - np.array(e["mu"])) / np.array(e["sd"])
    return rg.design_like(Z) @ np.array(e["w"])


def stats(y, p, w, q90):
    err = p - y
    ww = w / w.sum()
    rmse = float(np.sqrt(np.sum(ww * err ** 2)))
    return dict(rmse=rmse, bias=float(np.sum(ww * err)), cover=float(np.sum(ww * (np.abs(err) <= q90))))


def wmedian(v, w):
    o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


def main():
    path, fz = sys.argv[1], sys.argv[2]
    boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else BOOT
    frozen = json.load(open(fz)); rows = load(path)
    bad_early = [r for r in rows if any(v is not None and (not np.isfinite(v) or v <= 0) for v in r["P"])]
    if bad_early:
        raise ValueError(f"{len(bad_early)} rows have a non-positive or non-finite pairwise value (first: {bad_early[0]['cell']} "
                         f"network {bad_early[0]['net']}); the run is not analysed until the fault is found")
    ok = [r for r in rows if rc.complete(r["P"])]
    nnet = len({(r["cell"], r["net"]) for r in rows}); nok = len({(r["cell"], r["net"]) for r in ok})
    print(f"{len(rows)} rows from {nnet} networks; {len(ok)} rows from {nok} networks have a complete all-pairs profile "
          f"({len({r['cell'] for r in ok})} geometries)")
    rows = ok
    S = np.array([r["S"] for r in rows]); P = np.array([r["P"] for r in rows])
    rho = np.array([r["rho"] for r in rows]); y = np.log10(rho)
    var = np.array([r["variant"] for r in rows]); cell = np.array([r["cell"] for r in rows])
    nid = {}; sn = np.array([nid.setdefault((r["cell"], r["net"]), len(nid)) for r in rows])
    net_cell = {}
    for (c, k), i in nid.items():
        net_cell.setdefault(c, []).append(i)
    rng = np.random.default_rng(SEED)
    W = []
    for _ in range(boot):
        w = np.zeros(len(nid))
        for ids in net_cell.values():
            np.add.at(w, rng.choice(ids, len(ids)), 1)
        W.append(w[sn])
    W = np.array(W); w1 = np.ones(len(y))
    pred = {n: predict(frozen, n, S, P) for n in ESTS}
    # scale error on R0 (common to S and every shell), one factor per network, applied to the test films only
    z = np.exp(R0_SIGMA * np.random.default_rng(SEED + 1).standard_normal(len(nid)))[sn]
    pred_r0 = predict(frozen, "S+P", S * z, P * z[:, None])
    q = {n: frozen["estimators"][n]["q90"] for n in ESTS}

    def block(mask, label):
        res = {}
        for n in ESTS:
            res[n] = stats(y[mask], pred[n][mask], w1[mask], q[n])
            res[n]["median_factor"] = float(10 ** wmedian(np.abs(pred[n][mask] - y[mask]), w1[mask]))
            res[n]["p90_factor"] = float(10 ** np.percentile(np.abs(pred[n][mask] - y[mask]), 90))
            b = np.array([np.sqrt(np.sum(Wb[mask] * (pred[n][mask] - y[mask]) ** 2) / Wb[mask].sum()) for Wb in W])
            res[n]["rmse_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        d = np.array([np.sqrt(np.sum(Wb[mask] * (pred["S+level"][mask] - y[mask]) ** 2) / Wb[mask].sum()) -
                      np.sqrt(np.sum(Wb[mask] * (pred["S+P"][mask] - y[mask]) ** 2) / Wb[mask].sum()) for Wb in W])
        res["gain_level_minus_P"] = dict(point=res["S+level"]["rmse"] - res["S+P"]["rmse"], ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        # calibration slope: regression of the prediction on the true log10 rho
        res["slope_S+P"] = float(np.polyfit(y[mask], pred["S+P"][mask], 1)[0])
        return res

    inr = lambda lo, hi: (rho >= lo) & (rho <= hi)
    uni = var == "uniform"
    out = {}
    out["uniform_full"] = block(uni & inr(*FULL), "uniform, full range")
    out["uniform_phys"] = block(uni & inr(*PHYS), "uniform, physical range")
    for v in sorted(set(var) - {"uniform"}):
        out[f"{v}_full"] = block((var == v) & inr(*FULL), v)

    u = out["uniform_full"]
    h = {}
    h["T1"] = dict(rmse=u["S+P"]["rmse"], ci=u["S+P"]["rmse_ci"], confirmed=bool(u["S+P"]["rmse_ci"][1] <= RMSE_MARGIN))
    h["T2"] = dict(gain=u["gain_level_minus_P"]["point"], ci=u["gain_level_minus_P"]["ci"], confirmed=bool(u["gain_level_minus_P"]["ci"][0] > GAIN_MARGIN))
    h["T3"] = dict(coverage=u["S+P"]["cover"], confirmed=bool(COVER_LO <= u["S+P"]["cover"] <= COVER_HI))
    ph = out["uniform_phys"]
    h["T4"] = dict(rmse=ph["S+P"]["rmse"], ci=ph["S+P"]["rmse_ci"], confirmed=bool(ph["S+P"]["rmse_ci"][1] <= RMSE_MARGIN))
    m = uni & inr(*FULL); err = pred_r0[m] - y[m]
    b = np.array([np.sqrt(np.sum(Wb[m] * (pred_r0[m] - y[m]) ** 2) / Wb[m].sum()) for Wb in W])
    h["T5"] = dict(rmse=float(np.sqrt(np.mean(err ** 2))), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    h["T5"]["confirmed"] = bool(h["T5"]["ci"][1] <= R0_RMSE_MARGIN)
    print("\nT1 uniform, full range, RMSE of S+P: %.3f [%.3f, %.3f]  (upper bound <= %.2f) -> %s" % (h["T1"]["rmse"], *h["T1"]["ci"], RMSE_MARGIN, "CONFIRMED" if h["T1"]["confirmed"] else "not confirmed"))
    print("T2 gain in RMSE of S+P over S+level: %+.3f [%+.3f, %+.3f]  (lower bound > %.2f) -> %s" % (h["T2"]["gain"], *h["T2"]["ci"], GAIN_MARGIN, "CONFIRMED" if h["T2"]["confirmed"] else "not confirmed"))
    print("T3 coverage of the 90%% interval of S+P: %.3f  (between %.2f and %.2f) -> %s" % (h["T3"]["coverage"], COVER_LO, COVER_HI, "CONFIRMED" if h["T3"]["confirmed"] else "not confirmed"))
    print("T4 uniform, rho in [1, 30], RMSE of S+P: %.3f [%.3f, %.3f]  (upper bound <= %.2f) -> %s" % (h["T4"]["rmse"], *h["T4"]["ci"], RMSE_MARGIN, "CONFIRMED" if h["T4"]["confirmed"] else "not confirmed"))
    print("T5 R0 error sigma %.2f, RMSE of S+P: %.3f [%.3f, %.3f]  (upper bound <= %.2f) -> %s" % (R0_SIGMA, h["T5"]["rmse"], *h["T5"]["ci"], R0_RMSE_MARGIN, "CONFIRMED" if h["T5"]["confirmed"] else "not confirmed"))
    print("\nDescriptive, RMSE / median factor error / bias (log10) / coverage of the 90% interval, full range")
    for key, res in out.items():
        for n in ESTS:
            r = res[n]
            print(f"  {key:22s} {n:8s} RMSE {r['rmse']:.3f}  median factor {r['median_factor']:.2f}  p90 factor {r['p90_factor']:.2f}  bias {r['bias']:+.3f}  coverage {r['cover']:.2f}")
        print(f"  {key:22s} calibration slope of S+P on true log10 rho: {res['slope_S+P']:.2f}")
    out["hypotheses"] = h
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
