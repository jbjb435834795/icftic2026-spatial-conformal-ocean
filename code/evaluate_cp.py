# Main CP evaluation matrix — per-family execution for parallelism.
# Usage: python evaluate_cp.py --split dev_2020 --family convlstm
# Families: convlstm | fno2d | direct_resnet | persistence | climatology | quantile_cqr
# Writes: eval/partial_{family}_{split}.json + partial arrays npz.
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import cp_lib as cp

CAMP = Path(__file__).resolve().parent.parent
PRED = CAMP / "predictions"
EVAL = CAMP / "eval"
EVAL.mkdir(exist_ok=True)

SEEDS = [42, 7, 123]
VAR_NAMES = ["zos", "uo", "vo", "thetao", "so"]
LEADS = list(range(12))
LEVELS = [0.9, 0.95, 0.8]
ALPHA = {0.9: 0.1, 0.95: 0.05, 0.8: 0.2}
ACI_GAMMA = 0.005
TAU_DOY = 30.0
METHODS = ["split", "normalized", "gaussian", "online", "aci", "weighted"]


def season_of(doy):
    m = doy // 100
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[m]


def load_family(family, split):
    if family in ("persistence", "climatology"):
        d = np.load(PRED / f"{family}_{split}.npz")
        return {"pred": d["pred"].astype(np.float32),
                "scale": None,
                "targets": d["targets"].astype(np.float32),
                "doy": d["start_doy"].astype(int)}
    preds = [np.load(PRED / f"{family}_s{s}_{split}.npz")["pred"].astype(np.float32)
             for s in SEEDS]
    stack = np.stack(preds)
    d = np.load(PRED / f"{family}_s{SEEDS[0]}_{split}.npz")
    return {"pred": stack.mean(0), "scale": stack.std(0),
            "targets": d["targets"].astype(np.float32),
            "doy": d["start_doy"].astype(int)}


def load_quantile(split):
    qs = [np.load(PRED / f"quantile_s{s}_{split}.npz") for s in SEEDS]
    return {"qlo": np.stack([q["q05"] for q in qs]).mean(0).astype(np.float32),
            "q50": np.stack([q["q50"] for q in qs]).mean(0).astype(np.float32),
            "qhi": np.stack([q["q95"] for q in qs]).mean(0).astype(np.float32),
            "targets": qs[0]["targets"].astype(np.float32),
            "doy": qs[0]["start_doy"].astype(int)}


def stream_radii(err_cal, err_eval, alpha, gamma=None):
    """Online quantile stream over a preallocated buffer (no copies).
    gamma=None -> fixed-alpha refresh; gamma set -> per-cell ACI."""
    n0, cells = err_cal.shape
    nt = err_eval.shape[0]
    hist = np.empty((n0 + nt, cells), dtype=np.float32)
    hist[:n0] = np.abs(err_cal)
    abs_eval = np.abs(err_eval)
    alpha_t = np.full(cells, alpha, dtype=np.float64)
    radii = np.empty((nt, cells), dtype=np.float32)
    for t in range(nt):
        n = n0 + t
        if gamma is None:
            k = int(math.ceil((n + 1) * (1 - alpha)))
            k = min(max(k, 1), n)
            part = np.partition(hist[:n], k - 1, axis=0)
            radii[t] = part[k - 1]
        else:
            k = np.ceil((n + 1) * (1 - alpha_t)).astype(int)
            k = np.clip(k, 1, n)
            for kv in np.unique(k):
                cols = np.where(k == kv)[0]
                part = np.partition(hist[:n][:, cols], kv - 1, axis=0)
                radii[t, cols] = part[kv - 1, np.arange(len(cols))]
            exceed = (abs_eval[t] > radii[t]).astype(np.float64)
            alpha_t = np.clip(alpha_t + gamma * (alpha - exceed), 1e-4, 0.5)
        hist[n] = abs_eval[t]
    return radii


def weighted_radii(err_cal, doy_cal, doy_eval, alpha, tau=TAU_DOY):
    scores = np.abs(err_cal)
    order = np.argsort(scores, axis=0)
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    radii = np.empty((len(doy_eval), scores.shape[1]), dtype=np.float32)
    for t, td in enumerate(doy_eval):
        d = np.abs(doy_cal - td) % 365
        d = np.minimum(d, 365 - d)
        w = np.exp(-0.5 * (d / tau) ** 2).astype(np.float32)
        w_sorted = w[order]
        cw = np.cumsum(w_sorted, axis=0)
        cw /= cw[-1:]
        hit = cw >= (1 - alpha)
        idx = np.argmax(hit, axis=0)
        radii[t] = sorted_scores[idx, np.arange(scores.shape[1])]
    return radii


def summarize(y, lo, hi, doy, alpha):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    wink = cp.winkler(y, lo, hi, alpha)
    out = {
        "overall": {"coverage": float(inside.mean()), "width": float(width.mean()),
                    "winkler": float(wink.mean())},
        "by_variable": {}, "by_lead": {}, "by_season": {},
        "perwin_cover": inside.mean(axis=(3, 4)), "perwin_width": width.mean(axis=(3, 4)),
    }
    for v in range(5):
        out["by_variable"][VAR_NAMES[v]] = {
            "coverage": float(inside[:, :, v].mean()),
            "width": float(width[:, :, v].mean()),
            "winkler": float(wink[:, :, v].mean())}
    for h in range(12):
        out["by_lead"][h + 1] = {
            "coverage": float(inside[:, h].mean()),
            "width": float(width[:, h].mean()),
            "winkler": float(wink[:, h].mean())}
    seasons = np.array([season_of(d) for d in doy])
    for s in ["DJF", "MAM", "JJA", "SON"]:
        m = seasons == s
        out["by_season"][s] = {"coverage": float(inside[m].mean()),
                               "width": float(width[m].mean()),
                               "winkler": float(wink[m].mean()),
                               "windows": int(m.sum())}
    return out


def run_family(family, split):
    t0 = time.time()
    spatial, perwin = {}, {}
    fam_json = {"family": family, "split": split, "levels": {}}

    if family == "quantile_cqr":
        cal_q, ev_q = load_quantile("cal_2019"), load_quantile(split)
        fam_json["point_rmse_by_lead_var"] = np.sqrt(
            ((ev_q["targets"] - ev_q["q50"]) ** 2).mean(axis=(0, 3, 4))).tolist()
        for level in LEVELS:
            alpha = ALPHA[level]
            lo_all = np.empty_like(ev_q["qlo"])
            hi_all = np.empty_like(ev_q["qhi"])
            for v in range(5):
                for h in LEADS:
                    sl = (slice(None), h, v)
                    q = cp.cqr_conformal(
                        cal_q["qlo"][sl].reshape(-1), cal_q["qhi"][sl].reshape(-1),
                        cal_q["targets"][sl].reshape(-1), alpha)
                    lo_all[sl] = ev_q["qlo"][sl] - q
                    hi_all[sl] = ev_q["qhi"][sl] + q
            s = summarize(ev_q["targets"], lo_all, hi_all, ev_q["doy"], alpha)
            fam_json["levels"][str(level)] = {"cqr": {k: v2 for k, v2 in s.items()
                                                      if not k.startswith("perwin")}}
            perwin[f"quantile_cqr_cqr_L{int(level*100)}_cover"] = s["perwin_cover"]
            perwin[f"quantile_cqr_cqr_L{int(level*100)}_width"] = s["perwin_width"]
            spatial[f"quantile_cqr_cqr_L{int(level*100)}"] = (
                (ev_q["targets"] >= lo_all) & (ev_q["targets"] <= hi_all)).mean(0)
            print(f"[{split}] {family} L{level} done {time.time()-t0:.0f}s", flush=True)
    else:
        c, e = load_family(family, "cal_2019"), load_family(family, split)
        err_cal = c["targets"] - c["pred"]
        err_ev = e["targets"] - e["pred"]
        n_ev = err_ev.shape[0]
        fam_json["point_rmse_by_lead_var"] = np.sqrt(
            (err_ev ** 2).mean(axis=(0, 3, 4))).tolist()
        for level in LEVELS:
            alpha = ALPHA[level]
            methods = {}
            for method in METHODS:
                if method == "normalized" and family in ("persistence", "climatology"):
                    continue
                lo_all = np.empty_like(err_ev)
                hi_all = np.empty_like(err_ev)
                for v in range(5):
                    for h in LEADS:
                        ec = err_cal[:, h, v].reshape(len(err_cal), -1)
                        ee = err_ev[:, h, v].reshape(n_ev, -1)
                        if method == "split":
                            q = cp.split_conformal(ec.ravel(), alpha)
                            r = np.full_like(ee, q)
                        elif method == "normalized":
                            sc = c["scale"][:, h, v].reshape(len(err_cal), -1)
                            se = e["scale"][:, h, v].reshape(n_ev, -1)
                            r = cp.normalized_conformal(ec.ravel(), sc.ravel(), se, alpha).reshape(se.shape)
                        elif method == "gaussian":
                            r = np.full_like(ee, cp.gaussian_interval(ec.ravel(), alpha))
                        elif method == "online":
                            r = stream_radii(ec, ee, alpha, gamma=None)
                        elif method == "aci":
                            r = stream_radii(ec, ee, alpha, gamma=ACI_GAMMA)
                        elif method == "weighted":
                            r = weighted_radii(ec, c["doy"], e["doy"], alpha)
                        pe = e["pred"][:, h, v].reshape(n_ev, -1)
                        lo_all[:, h, v] = (pe - r).reshape(n_ev, 32, 64)
                        hi_all[:, h, v] = (pe + r).reshape(n_ev, 32, 64)
                s = summarize(e["targets"], lo_all, hi_all, e["doy"], alpha)
                methods[method] = {k: v2 for k, v2 in s.items() if not k.startswith("perwin")}
                perwin[f"{family}_{method}_L{int(level*100)}_cover"] = s["perwin_cover"]
                perwin[f"{family}_{method}_L{int(level*100)}_width"] = s["perwin_width"]
                spatial[f"{family}_{method}_L{int(level*100)}"] = (
                    (e["targets"] >= lo_all) & (e["targets"] <= hi_all)).mean(0)
                print(f"[{split}] {family} {method} L{level} done {time.time()-t0:.0f}s",
                      flush=True)
            fam_json["levels"][str(level)] = methods

    (EVAL / f"partial_{family}_{split}.json").write_text(json.dumps(fam_json))
    np.savez_compressed(EVAL / f"partial_{family}_{split}_arrays.npz",
                        **spatial, **perwin)
    print(f"[{split}] {family} COMPLETE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev_2020", "final_2021"], required=True)
    ap.add_argument("--family", required=True,
                    choices=["convlstm", "fno2d", "direct_resnet",
                             "persistence", "climatology", "quantile_cqr"])
    a = ap.parse_args()
    run_family(a.family, a.split)
