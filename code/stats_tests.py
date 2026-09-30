# Statistical tests for CP calibration on the final split.
# Moving-block bootstrap (block=14 windows ~ overlap horizon) over
# window-level mean coverage -> two-sided percentile p-values for
# H0: coverage == nominal; Holm correction across the tested family.
import json
from pathlib import Path

import numpy as np

CAMP = Path(__file__).resolve().parent.parent
EVAL = CAMP / "eval"

VAR_NAMES = ["zos", "uo", "vo", "thetao", "so"]
BLOCK = 14
NBOOT = 10000
LEVEL = 0.9
ALPHA = 0.1


def moving_block_pvalue(x, nominal, rng, block=BLOCK, nboot=NBOOT):
    """Two-sided bootstrap p-value for mean(x) == nominal."""
    n = len(x)
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(nboot, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(nboot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    p_le = float(np.mean(means <= nominal))
    p_ge = float(np.mean(means >= nominal))
    return 2.0 * min(p_le, p_ge)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def main(split="final_2021"):
    pw = np.load(EVAL / f"perwindow_{split}.npz")
    rng = np.random.default_rng(20260915)
    keys = sorted(k for k in pw.files if k.endswith("_cover") and "_L90_" in k)
    records = []
    for k in keys:
        cover = pw[k]  # (N, 12, 5)
        name = k[: -len("_cover")]
        for v in range(5):
            for h in range(12):
                x = cover[:, h, v]
                p = moving_block_pvalue(x, LEVEL, rng)
                records.append({
                    "series": name, "variable": VAR_NAMES[v], "lead": h + 1,
                    "coverage": float(x.mean()), "p_raw": p,
                })
    padj = holm([r["p_raw"] for r in records])
    for r, a in zip(records, padj):
        r["p_holm"] = float(a)
        r["reject_005"] = bool(a < 0.05)
    n_reject = sum(r["reject_005"] for r in records)
    by_series = {}
    for r in records:
        s = by_series.setdefault(r["series"], {"n": 0, "reject": 0,
                                               "min_cov": 1.0, "max_cov": 0.0})
        s["n"] += 1
        s["reject"] += int(r["reject_005"])
        s["min_cov"] = min(s["min_cov"], r["coverage"])
        s["max_cov"] = max(s["max_cov"], r["coverage"])
    out = {
        "split": split, "level": LEVEL, "block": BLOCK, "nboot": NBOOT,
        "n_tests": len(records), "n_reject_holm005": n_reject,
        "by_series": by_series,
        "records": records,
    }
    (EVAL / f"stats_{split}.json").write_text(json.dumps(out, indent=1))
    print(f"tests={len(records)} rejects(Holm<0.05)={n_reject}")
    for s, d in by_series.items():
        print(f"  {s:36s} n={d['n']:3d} reject={d['reject']:3d} "
              f"cov=[{d['min_cov']:.3f},{d['max_cov']:.3f}]")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "final_2021")
