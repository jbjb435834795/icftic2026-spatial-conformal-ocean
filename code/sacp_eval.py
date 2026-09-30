# Spatially-Adaptive Conformal Prediction (SACP) for regional ocean forecasting.
#
# Idea: split CP pools calibration scores GLOBALLY over the domain -> one radius
# for every grid cell -> severe conditional undercoverage in hard regions
# (equatorial cold tongue). SACP computes, per (variable, lead), a LOCAL radius
# field: at each grid cell, the (1-alpha) quantile of calibration scores pooled
# over a Gaussian spatial neighborhood (kernel scale sigma_s in grid cells,
# truncated at 2*sigma_s, weights renormalized at domain edges).
#   sigma_s -> inf : recovers global split CP (uniform radius)
#   sigma_s = 0    : per-cell local split CP (exact finite-sample per cell)
# Works with any conformity score: |y - mu| (mean forecasters) or the CQR score
# max(qlo - y, y - qhi) (quantile model) -> "SACP" / "SACP-CQR".
#
# Usage:
#   python sacp_eval.py --split dev_2020  --family convlstm --mode sweep
#   python sacp_eval.py --split final_2021 --family convlstm --mode final --sigma 2.0
# Sweep (sigma in {0,1,2,4}) writes eval/sacp_sweep_{family}_{split}.json
# Final writes eval/partial_{family}_sacp_{split}.json (+ _arrays.npz),
# same schema as evaluate_cp.py partial files.
from __future__ import annotations

import argparse
import json
import time
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


def season_of(doy):
    m = doy // 100
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[m]


def load_family(family, split):
    if family in ("persistence", "climatology"):
        d = np.load(PRED / f"{family}_{split}.npz")
        return {"pred": d["pred"].astype(np.float32), "targets": d["targets"].astype(np.float32),
                "doy": d["start_doy"].astype(int)}
    preds = [np.load(PRED / f"{family}_s{s}_{split}.npz")["pred"].astype(np.float32)
             for s in SEEDS]
    stack = np.stack(preds)
    d = np.load(PRED / f"{family}_s{SEEDS[0]}_{split}.npz")
    return {"pred": stack.mean(0), "targets": d["targets"].astype(np.float32),
            "doy": d["start_doy"].astype(int)}


def load_quantile(split):
    qs = [np.load(PRED / f"quantile_s{s}_{split}.npz") for s in SEEDS]
    return {"qlo": np.stack([q["q05"] for q in qs]).mean(0).astype(np.float32),
            "q50": np.stack([q["q50"] for q in qs]).mean(0).astype(np.float32),
            "qhi": np.stack([q["q95"] for q in qs]).mean(0).astype(np.float32),
            "targets": qs[0]["targets"].astype(np.float32),
            "doy": qs[0]["start_doy"].astype(int)}


def sacp_radius_fields(scores, sigma_s):
    """scores: (nc, H, W) calibration conformity scores for one (var, lead).
    Returns {level: radius field (H, W)} for all levels at once."""
    nc, H, W = scores.shape
    out = {lv: np.empty((H, W), dtype=np.float32) for lv in LEVELS}
    # per-cell sorted scores, layout (H, W, nc) for fast window gather
    S = np.sort(scores, axis=0).transpose(1, 2, 0)  # (H, W, nc)

    if sigma_s <= 0:  # per-cell local split CP (exact order statistic)
        for lv in LEVELS:
            k = int(np.ceil((nc + 1) * (1 - ALPHA[lv])))
            k = min(max(k, 1), nc)
            out[lv] = S[:, :, k - 1].copy()
        return out

    R = int(np.ceil(2 * sigma_s))
    oi, oj = np.mgrid[-R:R + 1, -R:R + 1]
    kern = np.exp(-0.5 * (oi ** 2 + oj ** 2) / sigma_s ** 2).astype(np.float64)
    targets_1m = {lv: 1 - ALPHA[lv] for lv in LEVELS}
    for i in range(H):
        i0, i1 = max(0, i - R), min(H, i + R + 1)
        for j in range(W):
            j0, j1 = max(0, j - R), min(W, j + R + 1)
            w = kern[i0 - (i - R):i1 - (i - R), j0 - (j - R):j1 - (j - R)].ravel()
            v = S[i0:i1, j0:j1].reshape(-1, nc).ravel()  # cell-major, rows sorted
            wv = np.repeat(w, nc)
            order = np.argsort(v, kind="stable")
            cw = np.cumsum(wv[order])
            cw /= cw[-1]
            vs = v[order]
            for lv in LEVELS:
                idx = int(np.searchsorted(cw, targets_1m[lv], side="left"))
                idx = min(idx, len(vs) - 1)
                out[lv][i, j] = vs[idx]
    return out


def summarize(y, lo, hi, doy, alpha):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    wink = winkler_score(y, lo, hi, alpha)
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


def winkler_score(y, lo, hi, alpha):
    w = hi - lo
    below = y < lo
    above = y > hi
    w = w + (2.0 / alpha) * (lo - y) * below
    w = w + (2.0 / alpha) * (y - hi) * above
    return w


def cell_stats(spatial_map, level_name):
    """spatial_map: (12, 5, 32, 64) -> stats of per-cell coverage."""
    cell = spatial_map.mean(axis=(0, 1))
    return {"level": level_name, "min": float(cell.min()), "mean": float(cell.mean()),
            "std": float(cell.std()), "frac_below_085": float((cell < 0.85).mean()),
            "frac_below_08": float((cell < 0.80).mean())}


def run(family, split, mode, sigma):
    t0 = time.time()
    if family == "quantile_cqr":
        cal, ev = load_quantile("cal_2019"), load_quantile(split)
        score_cal = np.maximum(cal["qlo"] - cal["targets"], cal["targets"] - cal["qhi"])
        centers = None  # intervals from qlo/qhi + radius
    else:
        cal, ev = load_family(family, "cal_2019"), load_family(family, split)
        score_cal = np.abs(cal["targets"] - cal["pred"])

    sigmas = [0.0, 1.0, 2.0, 4.0] if mode == "sweep" else [sigma]
    report = {"family": family, "split": split, "mode": mode, "sigma_results": {}}
    spatial_all, perwin_all = {}, {}

    for sg in sigmas:
        radii = {}  # (v,h) -> {level: (32,64)}
        for v in range(5):
            for h in LEADS:
                radii[(v, h)] = sacp_radius_fields(score_cal[:, h, v], sg)
        print(f"[{split}] {family} sigma={sg} radii done {time.time()-t0:.0f}s", flush=True)

        for level in LEVELS:
            alpha = ALPHA[level]
            if family == "quantile_cqr":
                lo_all = np.empty_like(ev["qlo"])
                hi_all = np.empty_like(ev["qhi"])
                for v in range(5):
                    for h in LEADS:
                        r = radii[(v, h)][level]
                        lo_all[:, h, v] = ev["qlo"][:, h, v] - r[None]
                        hi_all[:, h, v] = ev["qhi"][:, h, v] + r[None]
            else:
                lo_all = np.empty_like(ev["pred"])
                hi_all = np.empty_like(ev["pred"])
                for v in range(5):
                    for h in LEADS:
                        r = radii[(v, h)][level]
                        lo_all[:, h, v] = ev["pred"][:, h, v] - r[None]
                        hi_all[:, h, v] = ev["pred"][:, h, v] + r[None]
            s = summarize(ev["targets"], lo_all, hi_all, ev["doy"], alpha)
            smap = ((ev["targets"] >= lo_all) & (ev["targets"] <= hi_all)).mean(0)
            entry = {k: v2 for k, v2 in s.items() if not k.startswith("perwin")}
            entry["cell_stats"] = cell_stats(smap, level)
            report["sigma_results"][f"{sg:g}"] = report["sigma_results"].get(f"{sg:g}", {})
            report["sigma_results"][f"{sg:g}"][str(level)] = entry
            if mode == "final":
                tag = f"{family}_sacp_L{int(level*100)}"
                spatial_all[tag] = smap
                perwin_all[f"{tag}_cover"] = s["perwin_cover"]
                perwin_all[f"{tag}_width"] = s["perwin_width"]
            print(f"[{split}] {family} sigma={sg} L{level} "
                  f"cov={entry['overall']['coverage']:.3f} "
                  f"w={entry['overall']['width']:.4f} "
                  f"cellmin={entry['cell_stats']['min']:.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    if mode == "sweep":
        (EVAL / f"sacp_sweep_{family}_{split}.json").write_text(json.dumps(report))
    else:
        fam_json = {"family": family, "split": split, "sigma_s": sigma,
                    "levels": report["sigma_results"][f"{sigma:g}"]}
        (EVAL / f"partial_{family}_sacp_{split}.json").write_text(json.dumps(fam_json))
        np.savez_compressed(EVAL / f"partial_{family}_sacp_{split}_arrays.npz",
                            **spatial_all, **perwin_all)
    print(f"[{split}] {family} {mode} COMPLETE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev_2020", "final_2021"], required=True)
    ap.add_argument("--family", required=True,
                    choices=["convlstm", "fno2d", "direct_resnet",
                             "persistence", "climatology", "quantile_cqr"])
    ap.add_argument("--mode", choices=["sweep", "final"], required=True)
    ap.add_argument("--sigma", type=float, default=2.0)
    a = ap.parse_args()
    run(a.family, a.split, a.mode, a.sigma)
