# Conformal prediction methods for regional ocean forecasting.
# All methods operate per (variable, lead) on PHYSICAL-unit errors.
# Array conventions:
#   cal_err, test_err : (Nc,) / (Nt,) signed errors  (y - yhat)
#   interval output   : (lo, hi) arrays of shape (Nt,)
import math

import numpy as np


def split_conformal(cal_err, alpha):
    """Marginal split CP. Returns radius q (scalar)."""
    n = len(cal_err)
    scores = np.abs(cal_err)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float(np.max(scores))  # conservative bound
    q = np.sort(scores)[k - 1]
    return float(q)


def apply_interval(center, q):
    return center - q, center + q


def normalized_conformal(cal_err, cal_scale, test_scale, alpha, eps=1e-8):
    """Score s = |err| / sigma. Returns per-test-point radii."""
    n = len(cal_err)
    scores = np.abs(cal_err) / np.maximum(cal_scale, eps)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        q = float(np.max(scores))
    else:
        q = float(np.sort(scores)[k - 1])
    return q * np.maximum(test_scale, eps)


def cqr_conformal(cal_qlo, cal_qhi, cal_y, alpha):
    """CQR score = max(qlo - y, y - qhi). Returns additive correction q."""
    scores = np.maximum(cal_qlo - cal_y, cal_y - cal_qhi)
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float(np.max(scores))
    return float(np.sort(scores)[k - 1])


def cqr_apply(qlo, qhi, q):
    return qlo - q, qhi + q


def gaussian_interval(cal_err, alpha):
    """Normal-assumption baseline: z * sigma_hat."""
    sigma = float(np.std(cal_err, ddof=1))
    z = {0.10: 1.6449, 0.05: 1.9600, 0.20: 1.2816}[round(alpha, 2)]
    return z * sigma


def aci_stream(cal_err, test_err, alpha, gamma=0.005, alpha_init=None):
    """Adaptive conformal inference (Gibbs & Candes 2021), run per point.
    alpha_t updates online; radius = empirical (1-alpha_t) quantile of
    |cal_err| + past |test_err|. Returns radii array (Nt,)."""
    alpha_t = alpha if alpha_init is None else alpha_init
    history = np.abs(np.asarray(cal_err))
    radii = []
    for e in test_err:
        k = math.ceil((len(history) + 1) * (1 - alpha_t))
        k = min(max(k, 1), len(history))
        r = float(np.sort(history)[k - 1])
        radii.append(r)
        err = float(abs(e))
        alpha_t = alpha_t + gamma * (alpha - (1.0 if err > r else 0.0))
        alpha_t = min(max(alpha_t, 1e-4), 0.5)
        history = np.append(history, err)
    return np.array(radii)


def circular_doy_dist(a, b):
    """Circular day-of-year distance in [0, 182.5]."""
    d = abs(a - b) % 365
    return np.minimum(d, 365 - d)


def weighted_conformal(cal_err, cal_doy, test_doy, alpha, tau=30.0):
    """Seasonal weighted quantile CP (covariate-shift correction).
    Weights ~ exp(-0.5 * (doy_dist / tau)^2). Returns per-test radii."""
    cal_err = np.asarray(cal_err)
    scores = np.abs(cal_err)
    order = np.argsort(scores)
    s_sorted = scores[order]
    radii = []
    for td in np.atleast_1d(test_doy):
        dist = circular_doy_dist(cal_doy, td)
        w = np.exp(-0.5 * (dist / tau) ** 2)
        w_sorted = w[order]
        cw = np.cumsum(w_sorted) / np.sum(w_sorted)
        idx = np.searchsorted(cw, 1 - alpha)
        idx = min(idx, len(s_sorted) - 1)
        radii.append(float(s_sorted[idx]))
    return np.array(radii)


def enbpi(cal_err_per_seed, test_err, alpha, B=None):
    """Ensemble-batch prediction interval (simplified EnbPI):
    radius = (1-alpha) quantile of bootstrap-aggregated |residuals| where
    residuals combine each ensemble member's calibration error with the
    ensemble-mean test error at each step (online residual refresh)."""
    # per-seed calibration errors: list of (Nc,)
    n = len(test_err)
    ens_cal = np.mean(np.abs(cal_err_per_seed), axis=0)  # mean |err| over seeds
    history = ens_cal.copy()
    radii = []
    for e in test_err:
        k = math.ceil((len(history) + 1) * (1 - alpha))
        k = min(max(k, 1), len(history))
        r = float(np.sort(history)[k - 1])
        radii.append(r)
        history = np.append(history, abs(float(e)))
    return np.array(radii)


# ---------------- metrics ----------------

def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi):
    return float(np.mean(hi - lo))


def winkler(y, lo, hi, alpha):
    """Winkler score for symmetric intervals."""
    w = hi - lo
    below = y < lo
    above = y > hi
    w = w + (2.0 / alpha) * (lo - y) * below
    w = w + (2.0 / alpha) * (y - hi) * above
    return w
