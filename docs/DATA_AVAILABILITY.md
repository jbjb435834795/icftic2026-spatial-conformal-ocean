# Data availability

## 1. Input reanalysis: GLORYS12V1

The ocean state fields used for training, calibration, and evaluation come from
**GLORYS12V1**, the CMEMS global ocean reanalysis.

- Product page: Copernicus Marine Service — `GLOBAL_MULTIYEAR_PHY_001_030`
- Access requires a free Copernicus Marine account; the data are not redistributed
  here.
- Subset used: 10°S–10°N, 180°W–120°W, daily, regridded to a 32×64 grid.
- Variables: SSH, zonal current *u*, meridional current *v*, potential temperature
  (SST), salinity (SSS).

Any use of GLORYS12 must acknowledge the Copernicus Marine Service in accordance
with its licence terms.

## 2. Per-window prediction arrays (≈1.4 GB) — not distributed here

The frozen per-window prediction arrays produced by the trained forecasters
(ConvLSTM, FNO, ResNet, quantile ConvLSTM, persistence, climatology; calibration
2019, development 2020, evaluation 2021) are **not included in this repository**.
They are held in lab archival storage.

What this means concretely:

- Everything in `evidence/`, `results/`, and `logs/` was derived from those arrays
  and is sufficient to reproduce **every number in the paper's tables** and all
  reported statistical tests.
- The scripts that *consume* the raw arrays (`code/evaluate_cp.py`,
  `code/sacp_eval.py`, `code/delayed_online_eval.py`, `code/delayed_integrate.py`)
  will not run without them.
- The scripts that consume only the aggregates — `code/stats_tests.py`,
  `code/skill_ci.py`, `code/robust_bandwidth_analysis*`, and everything in
  `analysis/` — do run against the committed evidence files.

### Planned archival deposit

The arrays are intended for deposit in a persistent data repository so this work can
be fully re-executed end to end.

- Zenodo DOI: **TODO — add once the deposit is published**
- Deposit contents (planned): `predictions/*.npz` for 2019/2020/2021, plus the
  `evidence/final_2021/*.npz` frozen arrays excluded from git by `.gitignore`.

Until the deposit is live, the arrays can be requested from the authors.

## 3. Integrity checking

`docs/SHA256SUMS_v3.txt` lists SHA-256 checksums for the original experiment
package, including the `.npz` arrays that git does not carry. After obtaining the
arrays, verify them with:

```bash
sha256sum -c docs/SHA256SUMS_v3.txt
```

Mismatches indicate a corrupted or altered copy; do not publish results derived
from an unverified archive.
