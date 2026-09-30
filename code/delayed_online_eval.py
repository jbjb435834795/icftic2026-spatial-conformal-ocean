# Delayed-feedback re-evaluation of online refresh and ACI, plus per-window
# Winkler arrays for paired block-bootstrap CIs.
#
# Timing model (fixes the review issue "online feedback chronology"):
#   Window t (0-based) uses input days t..t+2 and is ISSUED at the end of day
#   t+2. The lead-L error of window t' targets day t'+2+L, so it can enter the
#   update history only when processing a window t with t >= t' + L.
#   Calibration-year (2019) errors are fully available before deployment.
#   ACI's alpha_t updates when each delayed error arrives, using the radius
#   that was issued for that window.
#
# Also recomputes split and SACP (sigma_s=1) intervals to emit per-window,
# per-variable Winkler scores for {split, sacp, online, aci} (delayed online).
#
# Usage: python delayed_online_eval.py --split final_2021 --family convlstm
# Writes: eval/delayed_partial_{family}_{split}.json
#         eval/delayed_partial_{family}_{split}_arrays.npz
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

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
SACP_SIGMA = 1.0
H, W = 32, 64


def season_of(doy):
    m = doy // 100
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[m]


def load_family(family, split):
    if family in ("persistence", "climatology"):
        d = np.load(PRED / f"{family}_{split}.npz")
        return {"pred": d["pred"].astype(np.float32),
                "targets": d["targets"].astype(np.float32),
                "doy": d["start_doy"].astype(int)}
    preds = [np.load(PRED / f"{family}_s{s}_{split}.npz")["pred"].astype(np.float32)
             for s in SEEDS]
    stack = np.stack(preds)
    d = np.load(PRED / f"{family}_s{SEEDS[0]}_{split}.npz")
    return {"pred": stack.mean(0), "targets": d["targets"].astype(np.float32),
            "doy": d["start_doy"].astype(int)}


def sacp_radius_fields(scores, sigma_s, levels):
    """Gaussian-kernel neighborhood weighted quantile radius fields (H, W).
    Identical to sacp_eval.sacp_radius_fields; replicated for standalone runs."""
    nc = scores.shape[0]
    S = np.sort(scores, axis=0).transpose(1, 2, 0)  # (H, W, nc)
    out = {lv: np.empty((H, W), dtype=np.float32) for lv in levels}
    R = int(np.ceil(2 * sigma_s))
    oi, oj = np.mgrid[-R:R + 1, -R:R + 1]
    kern = np.exp(-0.5 * (oi ** 2 + oj ** 2) / sigma_s ** 2).astype(np.float64)
    for i in range(H):
        i0, i1 = max(0, i - R), min(H, i + R + 1)
        for j in range(W):
            j0, j1 = max(0, j - R), min(W, j + R + 1)
            w = kern[i0 - (i - R):i1 - (i - R), j0 - (j - R):j1 - (j - R)].ravel()
            v = S[i0:i1, j0:j1].reshape(-1, nc).ravel()
            wv = np.repeat(w, nc)
            order = np.argsort(v, kind="stable")
            cw = np.cumsum(wv[order])
            cw /= cw[-1]
            vs = v[order]
            for lv in levels:
                idx = int(np.searchsorted(cw, 1 - ALPHA[lv], side="left"))
                out[lv][i, j] = vs[min(idx, len(vs) - 1)]
    return out


def delayed_stream_radii(err_cal, err_eval, alpha, lead, gamma=None):
    """Observation-aware delayed quantile stream for one (var, lead) slice.
    gamma=None -> fixed-alpha online refresh; gamma set -> delayed ACI."""
    nc, cells = err_cal.shape
    N = err_eval.shape[0]
    hist = np.empty((nc + N, cells), dtype=np.float32)
    hist[:nc] = np.abs(err_cal)
    n_hist = nc
    alpha_t = np.full(cells, float(alpha), dtype=np.float64)
    radii = np.empty((N, cells), dtype=np.float32)
    pending = deque()
    abs_ev = np.abs(err_eval)
    for t in range(N):
        while pending and pending[0] + lead <= t:
            tp = pending.popleft()
            e = abs_ev[tp]
            hist[n_hist] = e
            n_hist += 1
            if gamma is not None:
                exceed = (e > radii[tp]).astype(np.float64)
                alpha_t = np.clip(alpha_t + gamma * (alpha - exceed), 1e-4, 0.5)
        if gamma is None:
            k = int(math.ceil((n_hist + 1) * (1 - alpha)))
            k = min(max(k, 1), n_hist)
            radii[t] = np.partition(hist[:n_hist], k - 1, axis=0)[k - 1]
        else:
            k = np.ceil((n_hist + 1) * (1 - alpha_t)).astype(int)
            k = np.clip(k, 1, n_hist)
            for kv in np.unique(k):
                cols = np.where(k == kv)[0]
                part = np.partition(hist[:n_hist][:, cols], kv - 1, axis=0)
                radii[t, cols] = part[kv - 1, np.arange(len(cols))]
        pending.append(t)
    return radii


def split_radius(err_cal, alpha):
    scores = np.abs(err_cal).ravel()
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float(scores.max())
    return float(np.sort(scores)[k - 1])


def winkler(y, lo, hi, alpha):
    w = hi - lo
    w = w + (2.0 / alpha) * (lo - y) * (y < lo)
    w = w + (2.0 / alpha) * (y - hi) * (y > hi)
    return w


def summarize(y, lo, hi, doy, alpha):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    wink = winkler(y, lo, hi, alpha)
    out = {"overall": {"coverage": float(inside.mean()), "width": float(width.mean()),
                       "winkler": float(wink.mean())},
           "by_variable": {}, "by_lead": {}, "by_season": {},
           "perwin_cover": inside.mean(axis=(3, 4)),
           "perwin_width": width.mean(axis=(3, 4))}
    for v in range(5):
        out["by_variable"][VAR_NAMES[v]] = {
            "coverage": float(inside[:, :, v].mean()),
            "width": float(width[:, :, v].mean()),
            "winkler": float(wink[:, :, v].mean())}
    for h in range(12):
        out["by_lead"][h + 1] = {"coverage": float(inside[:, h].mean()),
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


def run(family, split):
    t0 = time.time()
    cal, ev = load_family(family, "cal_2019"), load_family(family, split)
    err_cal = cal["targets"] - cal["pred"]
    err_ev = ev["targets"] - ev["pred"]
    score_cal = np.abs(err_cal)
    n_ev = err_ev.shape[0]

    # SACP radius fields, computed once per (v, h), shared across levels
    sacp_cache = {}
    for v in range(5):
        for h in LEADS:
            sacp_cache[(v, h)] = sacp_radius_fields(
                score_cal[:, h, v], SACP_SIGMA, LEVELS)
    print(f"[{split}] {family} sacp fields cached {time.time()-t0:.0f}s", flush=True)

    fam_json = {"family": family, "split": split,
                "feedback": ("delayed: the lead-L error of window t enters the "
                             "history only when processing window t+L; ACI "
                             "alpha updates on arrival with the issued radius"),
                "levels": {}}
    spatial_all, perwin_all = {}, {}

    for level in LEVELS:
        alpha = ALPHA[level]
        lo_onl = np.empty_like(err_ev)
        hi_onl = np.empty_like(err_ev)
        lo_aci = np.empty_like(err_ev)
        hi_aci = np.empty_like(err_ev)
        wink_pw = {m: np.zeros((n_ev, 5), dtype=np.float64)
                   for m in ["split", "sacp", "online", "aci"]}
        for v in range(5):
            for h in LEADS:
                lead = h + 1
                ec = err_cal[:, h, v].reshape(len(err_cal), -1)
                ee = err_ev[:, h, v].reshape(n_ev, -1)
                pe = ev["pred"][:, h, v].reshape(n_ev, -1)
                y = ev["targets"][:, h, v].reshape(n_ev, -1)
                r_split = np.full_like(ee, split_radius(ec, alpha))
                r_sacp = np.broadcast_to(
                    sacp_cache[(v, h)][level].reshape(1, -1), (n_ev, H * W))
                r_onl = delayed_stream_radii(ec, ee, alpha, lead, gamma=None)
                r_aci = delayed_stream_radii(ec, ee, alpha, lead, gamma=ACI_GAMMA)
                for m, r in [("split", r_split), ("sacp", r_sacp),
                             ("online", r_onl), ("aci", r_aci)]:
                    lo = pe - r
                    hi = pe + r
                    wink_pw[m][:, v] += winkler(y, lo, hi, alpha).mean(axis=1)
                    if m == "online":
                        lo_onl[:, h, v] = lo.reshape(n_ev, H, W)
                        hi_onl[:, h, v] = hi.reshape(n_ev, H, W)
                    elif m == "aci":
                        lo_aci[:, h, v] = lo.reshape(n_ev, H, W)
                        hi_aci[:, h, v] = hi.reshape(n_ev, H, W)
            print(f"[{split}] {family} L{level} v{v} done {time.time()-t0:.0f}s",
                  flush=True)
        for m in wink_pw:
            wink_pw[m] /= 12.0

        methods_json = {}
        for m, lo_m, hi_m in [("online", lo_onl, hi_onl), ("aci", lo_aci, hi_aci)]:
            s = summarize(ev["targets"], lo_m, hi_m, ev["doy"], alpha)
            methods_json[m] = {k: v2 for k, v2 in s.items()
                               if not k.startswith("perwin")}
            tag = f"{family}_{m}_L{int(level*100)}"
            spatial_all[tag] = ((ev["targets"] >= lo_m) &
                                (ev["targets"] <= hi_m)).mean(0)
            perwin_all[f"{tag}_cover"] = s["perwin_cover"]
            perwin_all[f"{tag}_width"] = s["perwin_width"]
        fam_json["levels"][str(level)] = methods_json
        for m in wink_pw:
            perwin_all[f"{family}_{m}_L{int(level*100)}_winklerpw"] = wink_pw[m]
        print(f"[{split}] {family} L{level} COMPLETE "
              f"online_cov={methods_json['online']['overall']['coverage']:.3f} "
              f"w={methods_json['online']['overall']['width']:.4f} | "
              f"aci_cov={methods_json['aci']['overall']['coverage']:.3f} "
              f"w={methods_json['aci']['overall']['width']:.4f} "
              f"{time.time()-t0:.0f}s", flush=True)

    (EVAL / f"delayed_partial_{family}_{split}.json").write_text(json.dumps(fam_json))
    np.savez_compressed(EVAL / f"delayed_partial_{family}_{split}_arrays.npz",
                        **spatial_all, **perwin_all)
    print(f"[{split}] {family} DELAYED COMPLETE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev_2020", "final_2021"], required=True)
    ap.add_argument("--family", required=True,
                    choices=["convlstm", "fno2d", "direct_resnet",
                             "persistence", "climatology"])
    a = ap.parse_args()
    run(a.family, a.split)
