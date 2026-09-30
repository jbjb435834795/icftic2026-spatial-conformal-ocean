# Spatially Localized Conformal Calibration for Regional Ocean Forecasting

Reproducibility package for the manuscript *Spatially Localized Conformal Calibration
for Regional Ocean Forecasting* (Tao Li, Wenjie Li — China University of Petroleum
(East China)), submitted to ICFTIC 2026.

This repository contains the **conformal calibration code** and the **frozen result
files** from which every number in the paper's tables is derived. It is deliberately
minimal: no retraining pipeline, no raw reanalysis fields, and no manuscript PDF.

> **Status of the manuscript.** The paper is under review at the time of this release.
> This repository publishes the method code and frozen evidence only. Peer-review
> correspondence and the submitted/revised manuscript PDF are **not** included and
> are not intended for redistribution.

## What the paper does

Regional neural ocean forecasters produce skillful point predictions but poorly
calibrated uncertainty. This work compares conformal calibration strategies for
12-day forecasts of five ocean variables over the equatorial Pacific:

| Method | Idea |
|---|---|
| split CP | one global radius per (variable, lead), pooled over the whole domain |
| **SACP** | per-cell radius: Gaussian-weighted spatial neighbourhood of calibration scores |
| normalized / seasonal | scale- and season-aware variants |
| CQR | conformalized quantile regression from a quantile ConvLSTM |
| online refresh / ACI | adaptive, evaluated under **observation-aware feedback delay** |

The headline finding: on the 2021 evaluation period, SACP keeps nearly the same
marginal coverage as split CP while improving the dimensionless macro Winkler skill
by **6.5–6.8 %** across ConvLSTM, FNO, and ResNet, with paired block-bootstrap 95 %
confidence intervals excluding zero. SACP also sharply reduces severe cellwise
undercoverage in the equatorial cold tongue.

## Data and protocol

- **Source**: GLORYS12V1 reanalysis, 10°S–10°N, 180°W–120°W, daily 32×64 grid.
  GLORYS12 is produced by the Copernicus Marine Service and requires a free
  account; see [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).
- **Variables**: SSH, zonal current *u*, meridional current *v*, potential
  temperature (SST), salinity (SSS).
- **Windows**: 3 input days → 12 forecast days, unit stride, all calendar days kept
  (351 windows in a common year, 352 in leap year 2020).

| Period | Role |
|---|---|
| 2009–2017 | model training |
| 2018 | early stopping |
| 2019 | conformal calibration |
| 2020 | method/bandwidth development |
| 2021 | evaluation (frozen) |

**Important caveat.** SACP was designed *after* the baseline 2021 results had been
inspected. The 2021 SACP result is therefore an **exploratory** evaluation with the
bandwidth fixed on 2020, not an independent confirmatory test. See
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Repository layout

```
code/       conformal calibration library and evaluation scripts
analysis/   secondary analyses (spatial comparison, tile check, figure redraw)
evidence/   frozen JSON results — coverage, width, Winkler, seasonal split, tests
results/    derived CSV summaries (bandwidth margins, LOO transfer, macro skill)
logs/       run logs from the delayed-feedback and robustness executions
docs/       data availability, reproduction steps, number provenance, limitations
```

## Quick start

```bash
pip install -r requirements.txt          # numpy (required), matplotlib (figures only)
bash run.sh                              # rebuild the 2020 bandwidth robustness CSVs
```

`run.sh` executes `code/robust_bandwidth_analysis_portable.py`, which reads only the
frozen 2020 aggregate sweep JSONs in `evidence/` and writes to `results_rerun/`
(git-ignored). It does **not** reconstruct 2021 intervals and makes no independent
temporal-validation claim.

To recompute the supplementary comparisons:

```bash
python analysis/reviewer_analysis.py .    # global vs cellwise vs Gaussian bandwidth, -> analysis/
python analysis/small_spatial_check.py .  # fixed four-tile within-domain coverage check
python analysis/redraw_figures.py .       # redraw figures from frozen arrays (matplotlib)
```

Full end-to-end reproduction, including the delayed-feedback online/ACI chain that
requires the per-window prediction arrays, is documented in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Provenance of every table number

[`docs/NUMBERS_TRACE.md`](docs/NUMBERS_TRACE.md) maps each table entry of the paper to
the exact source file, JSON field, and generating script. The integrity of the frozen
arrays described there is verifiable against
[`docs/SHA256SUMS_v3.txt`](docs/SHA256SUMS_v3.txt).

## Known limitations

- Single-region case study (equatorial Pacific). No cross-region, second-product,
  or in-situ observational validation was performed.
- The 2021 SACP assessment is exploratory, not confirmatory.
- Autumn (SON) undercoverage persists for split CP, SACP, and the online methods.
- SACP is an empirical localizer; it does not inherit an exact distribution-free
  local validity guarantee from split CP.

## License

- **Code** (`code/`, `analysis/`): MIT — see [`LICENSE`](LICENSE).
- **Text and documentation** (`README.md`, `docs/`): CC BY 4.0 — see
  [`LICENSE-TEXT`](LICENSE-TEXT).
- **Frozen result data** (`evidence/`, `results/`, `logs/`): CC BY 4.0.

Cite this work using the metadata in [`CITATION.cff`](CITATION.cff).
