# Paired moving-block bootstrap CI for the dimensionless macro Winkler skill
# (paper Eq. 2) on the 2021 evaluation, per dynamic forecaster and method
# {sacp, online, aci} (delayed feedback), relative to split CP.
#
# Skill S = 100 [1 - (1/5) sum_v mean_w W_{m,v,w} / mean_w W_{split,v,w}]
# Windows are resampled with a paired moving-block bootstrap (block=14,
# B=10000), i.e. the same block indices are applied to both methods, so the
# split-CP denominator is resampled jointly with the numerator.
# Usage: python skill_ci.py final_2021
import json
import sys
from pathlib import Path

import numpy as np

CAMP = Path(__file__).resolve().parent.parent
EVAL = CAMP / "eval"
DYN = ["convlstm", "fno2d", "direct_resnet"]
METHODS = ["sacp", "online", "aci"]
BLOCK = 14
NBOOT = 10000
LEVEL = "L90"
RNG_SEED = 20260922


def skill(w_m, w_split):
    """w_*: (N, 5) per-window per-variable Winkler; returns macro skill %."""
    return 100.0 * (1.0 - np.mean(np.mean(w_m, axis=0) / np.mean(w_split, axis=0)))


def main(split="final_2021"):
    rng = np.random.default_rng(RNG_SEED)
    out = {"split": split, "level": "0.9", "block": BLOCK, "nboot": NBOOT,
           "paired": "same block indices applied to method and split arrays",
           "results": {}}
    for fam in DYN:
        z = np.load(EVAL / f"delayed_partial_{fam}_{split}_arrays.npz")
        w_split = z[f"{fam}_split_{LEVEL}_winklerpw"]  # (N, 5)
        N = w_split.shape[0]
        # precompute bootstrap block index matrix
        nblocks = int(np.ceil(N / BLOCK))
        starts = rng.integers(0, N, size=(NBOOT, nblocks))
        idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % N
        idx = idx.reshape(NBOOT, -1)[:, :N]
        fam_out = {"n_windows": int(N)}
        for m in METHODS:
            w_m = z[f"{fam}_{m}_{LEVEL}_winklerpw"]
            s_point = float(skill(w_m, w_split))
            boots = np.empty(NBOOT)
            for b in range(NBOOT):
                ii = idx[b]
                boots[b] = skill(w_m[ii], w_split[ii])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            fam_out[m] = {"skill_pct": round(s_point, 2),
                          "ci95": [round(float(lo), 2), round(float(hi), 2)]}
            print(f"{fam:14s} {m:7s} skill={s_point:5.2f}%  "
                  f"CI95=[{lo:.2f}, {hi:.2f}]")
        out["results"][fam] = fam_out
    (EVAL / f"skill_ci_{split}.json").write_text(json.dumps(out, indent=1))
    print("wrote", EVAL / f"skill_ci_{split}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final_2021")
