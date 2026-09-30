# Limitations

These are the honest boundaries of what this package supports. They are recorded
here so that no downstream use overstates the evidence.

## 1. Exploratory, not confirmatory

SACP was designed **after** the baseline 2021 results had been inspected. The 2021
SACP result is therefore an exploratory evaluation whose bandwidth (σ_s = 1) was
fixed using 2020 development data. It is **not** an independent confirmatory test,
and the reported skill gains should not be read as a pre-registered effect size.

## 2. Single region, single product

All results come from one regional case study: the equatorial Pacific
(10°S–10°N, 180°W–120°W) forced by GLORYS12V1 reanalysis. No experiment covers:

- another ocean region,
- a second observational or reanalysis data product,
- validation against in-situ observations.

Nothing here supports a cross-region generalization claim.

## 3. Within-domain checks are not external validation

`analysis/small_spatial_check.py` splits the *same* domain into four fixed 16×32
index-space tiles and re-measures empirical coverage against the frozen maps. It is a
post-hoc stratification of one domain. Its results (12/12 model × tile summaries
improving; 713 of 720 model × tile × variable × lead comparisons improving, 7
worsening with a maximum degradation of 0.35 pp, all 720 retained unfiltered) must
**not** be described as successful cross-region validation.

## 4. Autumn undercoverage is unresolved

Seasonal miscalibration persists. In SON, coverage is approximately 0.866 for ACI
and 0.867 for split CP — both below the 0.90 nominal level. Neither SACP nor the
adaptive methods fix this.

## 5. No exact local validity guarantee

SACP is an empirical spatially localized calibration rule. Assigning Gaussian
distance weights does not automatically inherit the exact finite-sample marginal
guarantee of split conformal prediction. Coverage figures reported for SACP are
*observed* coverage, not a distribution-free local validity claim.

## 6. Bandwidth selection is development-period only

The leave-one-configuration-out bandwidth result (matching the within-case optimum
in 91 % of architecture × level × variable settings) holds **within the 2020
development-period candidate set**. Efficiency differences between neighbouring
bandwidths are small, so the selection is not evidence of a sharply optimal scale.

## 7. Not a head-to-head comparison with LSCP or STACI

LSCP and STACI are discussed as related work, but no numerical comparison against
them was run. The numerical comparison covers global, cellwise, and
Gaussian-neighbourhood calibration only. Note also that the endpoint quantile rules
differ between the per-cell and positive-bandwidth variants, so the 0.33–0.39 pp
differences cannot be attributed entirely to neighbourhood pooling.

## 8. Statistical scope of the tests

The Holm correction is applied jointly across all 2,100 tests in the reported
family. Aggregate maps averaged over all variables and lead times are **not**
interchangeable with day-12 SST maps; mixing those two views of the results is a
misreading, not a finding.
