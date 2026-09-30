# Reproduction guide

Three tiers of reproducibility are supported, depending on how much data you have.

## Tier 1 — Verify the reported summaries (no extra data needed)

Uses only the frozen JSON/CSV files committed to this repository.

```bash
pip install -r requirements.txt
bash run.sh                              # 2020 bandwidth robustness -> results_rerun/
python analysis/reviewer_analysis.py .   # global / cellwise / Gaussian bandwidth comparison
python analysis/small_spatial_check.py . # fixed four-tile within-domain coverage check
```

`run.sh` runs `code/robust_bandwidth_analysis_portable.py`, which reads the four
2020 aggregate sweep JSONs in `evidence/` and writes `bandwidth_margin.csv`,
`loo_combo_transfer.csv`, `loo_combo_summary.csv`, `leave_one_model_out.csv`,
`leave_one_level_out.csv`, and `ROBUSTNESS_SUMMARY.md` into `results_rerun/`.
Compare against the committed `results/` — they should match.

## Tier 2 — Recompute statistics from frozen arrays

Requires the per-window arrays (see [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md)).

```bash
python code/stats_tests.py final_2021    # moving-block bootstrap + Holm correction
python code/skill_ci.py final_2021       # paired block-bootstrap CI for Winkler skill
```

- `stats_tests.py` reads `perwindow_final_2021.npz` and writes the 2,100 Holm-corrected
  test records.
- `skill_ci.py` reads the per-window Winkler arrays
  (`delayed_partial_{family}_final_2021_arrays.npz`, keys `*_winklerpw`) and
  reproduces `evidence/final_2021/skill_ci_final_2021.json` (block = 14, B = 10,000,
  paired: the same block indices are applied to the method and to the split-CP
  denominator).

## Tier 3 — Full re-execution from raw predictions

Requires the ~1.4 GB per-window prediction arrays placed under `predictions/`.

```
predictions/*.npz
  |- code/evaluate_cp.py          -> eval/partial_{family}_{split}.json
  |                                  (split / normalized / cqr / weighted / gaussian
  |                                   + instant-feedback online and ACI)
  |- code/sacp_eval.py            -> eval/partial_{family}_sacp_{split}.json
  |- code/delayed_online_eval.py  -> eval/delayed_partial_{family}_{split}.json
  |                                  + *_arrays.npz with per-window Winkler
  |- code/delayed_integrate.py    -> merges delayed results; backs up instant-feedback
  |                                  values to instantfb_backup_{split}.npz
  |- code/stats_tests.py          -> eval/stats_{split}.json
  |- code/skill_ci.py             -> eval/skill_ci_{split}.json
```

Note the directory convention: the scripts resolve paths relative to their own
location (`Path(__file__).resolve().parents[1]`), so keep the `predictions/`,
`eval/`, `evidence/`, and `code/` layout intact.

## The delayed-feedback timing model

The online refresh and ACI results in the paper use an **observation-aware feedback
delay**, implemented in `code/delayed_online_eval.py`:

- Window *t* (0-based) uses input days *t* … *t+2* and is issued at the end of day
  *t+2*.
- The lead-*L* error of window *t′* targets day *t′+2+L*, so it enters the update
  history only when processing a window *t* with *t ≥ t′ + L*.
- Calibration-year (2019) errors are fully available before deployment, so no
  leakage occurs there.
- ACI's α updates when each delayed error arrives, using the radius that was issued
  for that window.

The earlier instant-feedback values are preserved in
`instantfb_backup_final_2021.npz` and in the `*_instantfb` entries of the
`cp_results` JSONs.

## Reproducing figures

```bash
python analysis/redraw_figures.py .
```

Redraws the two manuscript figures from frozen arrays without recomputing any
experiment, and writes a trace JSON recording the source hashes.
