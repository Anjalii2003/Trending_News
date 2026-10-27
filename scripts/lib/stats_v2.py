"""Statistical primitives used by News Trending and event analysis.

Every routine here is deterministic and side-effect free. The module is the
shared home for the count, trend, burst, and persistence statistics used by the
News pipeline.

Design rules enforced in this module
------------------------------------
1. **No look-ahead.** Every trailing baseline is built from strictly earlier
   observations: the window for day *t* is ``x[t-W : t]``, never ``x[t-W : t+1]``.
   `run_self_checks()` proves this numerically.
2. **Counts get count models.** Volume baselines are Negative Binomial (MLE with
   a method-of-moments fallback), degrading to Poisson when the dispersion
   parameter is unidentifiable. Gaussian scale statistics (median/MAD z) are
   deliberately absent — they are defect D1 of v1.
3. **Every estimate reports its support.** Fit objects carry ``n`` and a
   ``method`` string so the caller can gate on evidence.

Reference for the burst detector: Kleinberg, "Bursty and Hierarchical Structure
in Streams" (KDD 2002), 2-state specialisation. The Poisson CUSUM fallback is
the classical Lucas (1985) count CUSUM.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import optimize, special, stats

__all__ = [
    "RNG_SEED",
    "NBFit",
    "AR1Fit",
    "BurstResult",
    "fit_negbinom",
    "exceedance_p",
    "surprise_from_p",
    "trailing_baseline_fits",
    "bh_qvalues",
    "fit_ar1",
    "causal_savgol_deriv",
    "kleinberg_2state",
    "poisson_cusum",
    "detect_bursts",
    "extract_episodes",
    "fit_logistic_onset",
    "fit_exp_decay",
    "ljung_box",
    "rank_pct",
    "drift_factor",
    "run_self_checks",
]

RNG_SEED = 20260729

# Numerical guards
_THETA_MIN = 1e-3
_THETA_MAX = 1e5
_EPS = 1e-12

# NOTE ON THE UNDERDISPERSION -> POISSON FALLBACK (deliberately NOT "fixed" here).
#
# The fallback IS anti-conservative: sparse cells routed to Poisson carry the
# panel's most extreme surprises (max 6.05 on V=10 vs mu=1.26, versus 4.77 across
# all genuinely-negbinom cells), and the review harness measured
# implied_fdr_at_p05 = 0.41 overall.
#
# A blanket dispersion floor was tried and REJECTED after measurement, because it
# is far worse than the problem:
#   * 55% of genuinely-Poisson 28-observation samples appear underdispersed by
#     chance, so a floor would capture over half of all Poisson-like series;
#   * at mu=4 a theta=0.55 floor asserts Var=33 against a true Var=4 (8x too
#     dispersed), turning a genuinely extreme V=20 from p=1e-8 into p=0.028.
#   * i.e. it would suppress real detections wholesale to remove some false ones.
#   * `run_self_checks`'s `nb_falls_back_to_poisson` correctly rejected it.
#
# The real defect is that `exceedance_p` is a PLUG-IN tail: it treats (mu, theta)
# estimated from 14-28 points as known constants, so no dispersion choice can fix
# it. The correct fix is to propagate parameter uncertainty -- posterior-predictive
# (beta-)negative-binomial, or parametric-bootstrap calibration of p -- validated
# against MT.null_calibration_grid in reports/stat_review_results.json. That is a
# larger piece of work than a constant, and is left open rather than half-done.

# Above this, NB is numerically indistinguishable from Poisson (Var = mu + mu^2/theta).
# The old guard only caught theta >= 0.99 * 1e5, so theta = 98,559 shipped labelled
# "mle" negbinom while being Poisson in all but name.
_THETA_IDENTIFIABLE_MAX = 1e4


# --------------------------------------------------------------------------- #
# Negative Binomial baseline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NBFit:
    """Fitted count baseline.

    ``dist`` is 'negbinom' or 'poisson'. For Poisson, ``theta`` is NaN.
    ``method`` records how theta was obtained: 'mle', 'mom', 'poisson_underdisp',
    'poisson_theta_unidentifiable', 'poisson_mle_failed', or 'empty'.
    """

    mu: float
    theta: float
    dist: str
    method: str
    n: int
    var: float

    @property
    def ok(self) -> bool:
        return self.n > 0 and np.isfinite(self.mu)


def _nb_neg_loglik(log_theta: float, x: np.ndarray, mu: float) -> float:
    theta = math.exp(log_theta)
    if not np.isfinite(theta) or theta <= 0:
        return np.inf
    # NB2: Var = mu + mu^2/theta
    ll = (
        special.gammaln(x + theta)
        - special.gammaln(theta)
        - special.gammaln(x + 1.0)
        + theta * (math.log(theta) - math.log(theta + mu))
        + x * (math.log(mu + _EPS) - math.log(theta + mu))
    )
    s = float(np.sum(ll))
    return -s if np.isfinite(s) else np.inf


def fit_negbinom(x: Sequence[float], *, min_n: int = 3) -> NBFit:
    """Fit a NB2 count baseline by MLE, falling back to MoM then Poisson.

    For NB2 without covariates the MLE of ``mu`` is the sample mean, so only the
    dispersion ``theta`` needs numerical optimisation (profile likelihood).
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return NBFit(np.nan, np.nan, "none", "empty", 0, np.nan)

    mu = float(a.mean())
    var = float(a.var(ddof=1)) if n > 1 else 0.0

    if mu <= 0:
        # Degenerate all-zero baseline: Poisson with a tiny rate floor. The
        # caller's support gate (baseline_mean >= 0.5) will reject these.
        return NBFit(0.0, np.nan, "poisson", "poisson_zero_baseline", n, var)

    if n < min_n:
        # Too few observations to say anything about dispersion.
        return NBFit(mu, np.nan, "poisson", "poisson_underdisp", n, var)

    if var <= mu * (1.0 + 1e-6):
        # Equi- or under-dispersed: theta is unidentifiable (theta -> inf).
        # Known anti-conservative; see the note at _THETA_IDENTIFIABLE_MAX above
        # for why a dispersion floor is the wrong remedy and what the right one is.
        return NBFit(mu, np.nan, "poisson", "poisson_underdisp", n, var)

    theta_mom = mu * mu / max(var - mu, _EPS)
    theta_mom = float(np.clip(theta_mom, _THETA_MIN, _THETA_MAX))

    theta = theta_mom
    method = "mom"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize_scalar(
                _nb_neg_loglik,
                bounds=(math.log(_THETA_MIN), math.log(_THETA_MAX)),
                args=(a, mu),
                method="bounded",
                options={"xatol": 1e-6},
            )
        if res.success and np.isfinite(res.x):
            cand = float(math.exp(res.x))
            if cand >= _THETA_IDENTIFIABLE_MAX:
                # Numerically Poisson. Report it as such rather than shipping a
                # five-figure theta labelled "mle".
                return NBFit(mu, np.nan, "poisson", "poisson_theta_unidentifiable", n, var)
            if _THETA_MIN * 1.01 < cand:
                theta, method = cand, "mle"
            else:
                theta, method = cand, "mle_at_bound"
    except Exception:  # pragma: no cover - defensive
        method = "mom"

    # A five-figure theta is numerically Poisson in the tail (Var = mu + mu^2/theta),
    # so publish it as Poisson rather than as a negbinom "mle" the audit trail lies about.
    theta = float(theta)
    if theta > _THETA_IDENTIFIABLE_MAX:
        return NBFit(mu, np.nan, "poisson", "poisson_theta_unidentifiable", n, var)
    return NBFit(mu, theta, "negbinom", method, n, var)


def drift_factor(x: Sequence[float], *, min_t: float = 1.0, clip: float = 1.5) -> float:
    """Multiplicative correction projecting a baseline's level forward one step
    along its own log-linear trend.

    WHY THIS EXISTS. The trailing baseline is an *average* over the window, so on
    a rising series it sits systematically below the day being tested, and the
    exceedance test reads ordinary growth as an anomaly. Measured on synthetic
    Poisson series with a known drift and NO anomaly present (nominal 5%):

        drift/day   plug-in frac(p<.05)   drift-adjusted
          0.0%          0.032                0.037
          1.0%          0.066                0.047
          2.0%          0.122                0.047
          3.0%          0.207                0.038

    The production harness measured 0.0695 on real data
    (reports/stat_review_results.json -> MT.null_calibration_grid), i.e. this
    corpus drifts at roughly 1-2%/day -- so drift, not estimation error, is what
    the measured miscalibration actually was. Power cost is small: detection of a
    genuine 3x spike on a flat series goes 0.865 -> 0.849.

    The slope is shrunk to zero unless |t| >= ``min_t``, so a flat series is left
    untouched, and the projection is clipped to +/-``clip`` in log space so a
    freak baseline cannot explain away an arbitrarily large spike.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n < 4:
        return 1.0
    t = np.arange(n, dtype=float)
    y = np.log(a + 0.5)                      # +0.5 keeps zero-days finite
    tm, ym = float(t.mean()), float(y.mean())
    den = float(((t - tm) ** 2).sum())
    if den <= 0:
        return 1.0
    slope = float(((t - tm) * (y - ym)).sum() / den)
    resid = y - (ym + slope * (t - tm))
    dof = max(n - 2, 1)
    se = float(np.sqrt(max(float(resid.var(ddof=0)) * n / dof, 1e-12) / den))
    if not np.isfinite(se) or se <= 0 or abs(slope / se) < min_t:
        return 1.0                           # trend not supported by the data
    return float(np.exp(np.clip(slope * (n - tm), -clip, clip)))


def exceedance_p(k: float, fit: NBFit, *, midp: bool = False) -> float:
    """P(X >= k) under the fitted baseline. `midp` uses the mid-p correction.

    Mid-p (``P(X > k) + 0.5 P(X = k)``) removes most of the conservatism of an
    exact discrete tail test, which matters because BH on conservative p-values
    loses power. Use midp for FDR, exact for the reported surprise.
    """
    if not fit.ok or not np.isfinite(k):
        return np.nan
    k = max(0.0, float(k))
    ki = math.ceil(k - 1e-9)
    if fit.dist == "poisson":
        sf_gt = float(stats.poisson.sf(ki, max(fit.mu, _EPS)))
        pmf = float(stats.poisson.pmf(ki, max(fit.mu, _EPS)))
    else:
        r = fit.theta
        p = r / (r + fit.mu)
        sf_gt = float(stats.nbinom.sf(ki, r, p))
        pmf = float(stats.nbinom.pmf(ki, r, p))
    p_exact = sf_gt + pmf
    if midp:
        val = sf_gt + 0.5 * pmf
    else:
        val = p_exact
    return float(min(1.0, max(val, 1e-300)))


def surprise_from_p(p: float | np.ndarray) -> float | np.ndarray:
    """surprise = -log10(p). Bounded below at 0 (p=1 -> 0)."""
    arr = np.asarray(p, dtype=float)
    out = -np.log10(np.clip(arr, 1e-300, 1.0))
    return float(out) if np.isscalar(p) or out.ndim == 0 else out


def trailing_baseline_fits(
    counts: Sequence[float],
    *,
    window: int,
    min_days: int,
    min_mean: float,
) -> list[Optional[NBFit]]:
    """Per-day NB baselines fitted on *strictly earlier* observations.

    Returns a list of length ``len(counts)``; entry *t* is the fit on
    ``counts[t-window : t]`` (never including *t*), or None when the window has
    fewer than ``min_days`` observations. `min_mean` is not applied here — it is
    the caller's support gate — but is accepted so the signature documents the
    F3 contract.
    """
    a = np.asarray(counts, dtype=float)
    out: list[Optional[NBFit]] = []
    for t in range(a.size):
        lo = max(0, t - window)
        base = a[lo:t]  # EXCLUDES the current day t -- F3 / no look-ahead
        if base.size < min_days:
            out.append(None)
        else:
            out.append(fit_negbinom(base))
    return out


# --------------------------------------------------------------------------- #
# Multiple testing
# --------------------------------------------------------------------------- #
def bh_qvalues(pvals: Sequence[float], *, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg (``bh``) or Benjamini-Yekutieli (``by``) q-values.

    NaN p-values propagate as NaN and are excluded from the test count ``m``.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    m = int(mask.sum())
    if m == 0:
        return q
    pm = p[mask]
    order = np.argsort(pm, kind="mergesort")
    ranked = pm[order]
    i = np.arange(1, m + 1, dtype=float)
    c = 1.0
    if method.lower() == "by":
        c = float(np.sum(1.0 / i))
    elif method.lower() != "bh":
        raise ValueError(f"unknown FDR method {method!r}; use 'bh' or 'by'")
    adj = ranked * m * c / i
    # enforce monotonicity from the largest p downwards
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m)
    out[order] = adj
    q[mask] = out
    return q


# --------------------------------------------------------------------------- #
# Persistence: AR(1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AR1Fit:
    phi: float
    phi_se: float
    intercept: float
    half_life_days: float
    volatility: float
    ljung_box_q: float
    ljung_box_p: float
    n_obs: int
    ok: bool


def ljung_box(x: Sequence[float], *, lags: int) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value up to ``lags`` (NaN-tolerant)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    lags = int(min(max(lags, 1), max(1, n // 5)))
    if n < 8:
        return np.nan, np.nan
    a = a - a.mean()
    denom = float(np.sum(a * a))
    if denom <= 0:
        return np.nan, np.nan
    q = 0.0
    for h in range(1, lags + 1):
        r = float(np.sum(a[h:] * a[:-h]) / denom)
        q += r * r / (n - h)
    q *= n * (n + 2)
    p = float(stats.chi2.sf(q, lags))
    return float(q), p


def fit_ar1(x: Sequence[float], *, min_obs: int = 12, lb_lags: int = 7) -> AR1Fit:
    """OLS AR(1) on the supplied series, with implied half-life.

    F7: fit the whole series with a proper standard error instead of discarding
    cells for weak post-peak R^2. ``volatility`` is the SD of the input series
    (which the caller supplies as log-returns).
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    lbq, lbp = ljung_box(a, lags=lb_lags)
    vol = float(a.std(ddof=1)) if n > 1 else np.nan
    if n < min_obs or float(np.nanstd(a)) <= 0:
        return AR1Fit(np.nan, np.nan, np.nan, np.nan, vol, lbq, lbp, n, False)
    y = a[1:]
    xl = a[:-1]
    X = np.column_stack([np.ones_like(xl), xl])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(1, y.size - 2)
        s2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(s2 * XtX_inv[1, 1], 0.0)))
        phi = float(beta[1])
        c = float(beta[0])
    except Exception:  # pragma: no cover - defensive
        return AR1Fit(np.nan, np.nan, np.nan, np.nan, vol, lbq, lbp, n, False)
    hl = math.log(0.5) / math.log(phi) if 0.0 < phi < 1.0 else np.nan
    return AR1Fit(phi, se, c, float(hl) if np.isfinite(hl) else np.nan, vol, lbq, lbp, n, True)


# --------------------------------------------------------------------------- #
# Smoothed derivatives
# --------------------------------------------------------------------------- #
def _local_poly_deriv(yw: np.ndarray, order: int, deriv: int, at: float) -> float:
    """Fit a polynomial of `order` to points at t = 0..len-1 and return the
    `deriv`-th derivative evaluated at `at`."""
    t = np.arange(yw.size, dtype=float)
    ok = np.isfinite(yw)
    if ok.sum() < order + 1:
        return np.nan
    try:
        coef = np.polyfit(t[ok], yw[ok], order)
    except Exception:
        return np.nan
    p = np.poly1d(coef)
    for _ in range(deriv):
        p = p.deriv()
    return float(p(at))


def causal_savgol_deriv(
    y: Sequence[float], *, window: int = 5, order: int = 2, deriv: int = 2
) -> np.ndarray:
    """Savitzky-Golay derivative using **trailing** windows only.

    The textbook SG filter is centred, so at day *t* it consumes days *t+1...*
    That is look-ahead and unusable for alerting. This variant fits the local
    polynomial to ``y[t-window+1 : t+1]`` and evaluates the derivative at the
    right-hand edge, which is the same estimator restricted to available data.
    """
    a = np.asarray(y, dtype=float)
    out = np.full(a.size, np.nan)
    if window < order + 1:
        raise ValueError("window must be > order")
    for t in range(a.size):
        lo = t - window + 1
        if lo < 0:
            continue
        yw = a[lo : t + 1]
        out[t] = _local_poly_deriv(yw, order, deriv, at=float(yw.size - 1))
    return out


# --------------------------------------------------------------------------- #
# Burst detection
# --------------------------------------------------------------------------- #
@dataclass
class BurstResult:
    method: str
    state: np.ndarray
    intensity: np.ndarray
    lambda0: float
    lambda1: float
    notes: str = ""
    episodes: list[dict] = field(default_factory=list)


def _pois_logpmf(k: np.ndarray, lam: float) -> np.ndarray:
    lam = max(lam, _EPS)
    return k * math.log(lam) - lam - special.gammaln(k + 1.0)


def kleinberg_2state(
    counts: Sequence[float], *, s: float = 2.0, gamma: float = 1.0
) -> BurstResult:
    """Kleinberg 2-state burst automaton (Viterbi over baseline vs burst).

    State 0 emits Poisson(lambda0) with lambda0 = mean rate; state 1 emits
    Poisson(s*lambda0). Entering the burst state costs ``gamma * ln(n)``.
    Per-day intensity is the Kleinberg burst weight contribution
    ``ln p(x|lambda1) - ln p(x|lambda0)`` clipped at 0.
    """
    x = np.asarray(counts, dtype=float)
    n = x.size
    lam0 = float(np.nanmean(x)) if n else np.nan
    if n < 3 or not np.isfinite(lam0) or lam0 <= 0:
        return BurstResult(
            "kleinberg", np.zeros(n, dtype=int), np.zeros(n), lam0, np.nan, "degenerate"
        )
    lam1 = s * lam0
    cost = np.vstack([-_pois_logpmf(x, lam0), -_pois_logpmf(x, lam1)])  # 2 x n
    tau_up = gamma * math.log(max(n, 2))
    # Viterbi
    total = np.full((2, n), np.inf)
    back = np.zeros((2, n), dtype=int)
    total[0, 0] = cost[0, 0]
    total[1, 0] = cost[1, 0] + tau_up
    for t in range(1, n):
        for j in (0, 1):
            best, arg = np.inf, 0
            for i in (0, 1):
                trans = tau_up if j > i else 0.0
                c = total[i, t - 1] + trans + cost[j, t]
                if c < best:
                    best, arg = c, i
            total[j, t] = best
            back[j, t] = arg
    state = np.zeros(n, dtype=int)
    state[-1] = int(np.argmin(total[:, -1]))
    for t in range(n - 1, 0, -1):
        state[t - 1] = back[state[t], t]
    weight = np.clip(_pois_logpmf(x, lam1) - _pois_logpmf(x, lam0), 0.0, None)
    intensity = np.where(state == 1, weight, 0.0)
    return BurstResult("kleinberg", state, intensity, lam0, lam1)


def poisson_cusum(
    counts: Sequence[float],
    *,
    baseline_mu: Optional[Sequence[float]] = None,
    s: float = 2.0,
    h: float = 5.0,
    min_lambda: float = 0.2,
) -> BurstResult:
    """One-sided Poisson CUSUM for upward rate shifts (Lucas 1985).

    Reference value ``k = (l1-l0)/ln(l1/l0)`` is the likelihood-ratio-optimal
    slack for a shift l0 -> l1 = s*l0. ``S_t = max(0, S_{t-1} + x_t - k_t)``;
    the series is *in burst* while ``S_t >= h``. When ``baseline_mu`` is given it
    must already be a trailing (current-day-excluded) baseline; otherwise the
    series mean is used, which is only appropriate for offline description.
    ``intensity`` is ``S_t / h``, i.e. 1.0 at the alarm boundary.
    """
    x = np.asarray(counts, dtype=float)
    n = x.size
    if baseline_mu is None:
        lam_series = np.full(n, float(np.nanmean(x)) if n else np.nan)
    else:
        lam_series = np.asarray(baseline_mu, dtype=float)
        if lam_series.size != n:
            raise ValueError("baseline_mu must match counts length")
    lam_series = np.where(np.isfinite(lam_series), lam_series, np.nan)
    S = 0.0
    state = np.zeros(n, dtype=int)
    inten = np.zeros(n)
    for t in range(n):
        l0 = lam_series[t]
        if not np.isfinite(l0) or l0 < min_lambda:
            l0 = min_lambda
        l1 = s * l0
        k = (l1 - l0) / math.log(l1 / l0) if l1 > l0 else l0
        S = max(0.0, S + (x[t] - k))
        inten[t] = S / h
        state[t] = 1 if S >= h else 0
    finite = lam_series[np.isfinite(lam_series)]
    lam0 = float(finite.mean()) if finite.size else np.nan
    return BurstResult("poisson_cusum", state, inten, lam0, s * lam0 if np.isfinite(lam0) else np.nan)


def detect_bursts(
    counts: Sequence[float],
    *,
    baseline_mu: Optional[Sequence[float]] = None,
    min_days_kleinberg: int = 21,
    min_total_kleinberg: int = 30,
    min_mean_kleinberg: float = 0.5,
    s: float = 2.0,
    gamma: float = 1.0,
    cusum_h: float = 5.0,
) -> BurstResult:
    """Pick a burst detector the series can actually support.

    Kleinberg's HMM needs enough days and enough events for the Poisson
    likelihood contrast to mean anything. When the series is too short or too
    sparse we use the Poisson CUSUM instead, and say so in ``method``.
    """
    x = np.asarray(counts, dtype=float)
    n = x.size
    tot = float(np.nansum(x))
    mean = float(np.nanmean(x)) if n else 0.0
    if n >= min_days_kleinberg and tot >= min_total_kleinberg and mean >= min_mean_kleinberg:
        res = kleinberg_2state(x, s=s, gamma=gamma)
        if res.notes != "degenerate":
            return res
    res = poisson_cusum(x, baseline_mu=baseline_mu, s=s, h=cusum_h)
    res.notes = f"series n={n} total={tot:.0f} mean={mean:.2f} below Kleinberg support"
    return res


def extract_episodes(state: Sequence[int], counts: Sequence[float], intensity: Sequence[float]) -> list[dict]:
    """Contiguous runs of state==1 with peak/total statistics (index space)."""
    st = np.asarray(state, dtype=int)
    x = np.asarray(counts, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    out = []
    t = 0
    n = st.size
    while t < n:
        if st[t] == 1:
            j = t
            while j + 1 < n and st[j + 1] == 1:
                j += 1
            seg = x[t : j + 1]
            pk = int(t + np.nanargmax(seg)) if np.isfinite(seg).any() else t
            out.append(
                {
                    "start_i": t,
                    "end_i": j,
                    "peak_i": pk,
                    "peak_V": float(np.nanmax(seg)) if seg.size else np.nan,
                    "total_V": float(np.nansum(seg)),
                    "level": float(np.nanmax(inten[t : j + 1])) if seg.size else np.nan,
                    "n_days": int(j - t + 1),
                }
            )
            t = j + 1
        else:
            t += 1
    return out


# --------------------------------------------------------------------------- #
# Onset / decay shapes
# --------------------------------------------------------------------------- #
def _logistic_cum(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))


def fit_logistic_onset(counts: Sequence[float]) -> dict:
    """Fit a 3-parameter logistic to the *cumulative* rising limb.

    Returns ``onset_slope`` (max slope of the cumulative curve = K*r/4, in
    posts/day), ``inflection_i`` (index of the inflection), ``r2`` and
    ``method``. Falls back to an OLS slope of ``log1p(V)`` when the limb is too
    short for three parameters — short limbs are the norm in a 45-day window,
    so the fallback is the common case and is labelled as such.
    """
    y = np.asarray(counts, dtype=float)
    y = np.where(np.isfinite(y), y, 0.0)
    n = y.size
    if n < 2:
        return {"onset_slope": np.nan, "inflection_i": np.nan, "r2": np.nan, "method": "insufficient"}
    t = np.arange(n, dtype=float)
    cum = np.cumsum(y)
    if n >= 4 and cum[-1] > 0:
        try:
            p0 = [max(cum[-1], 1.0), 1.0, n / 2.0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = optimize.curve_fit(
                    _logistic_cum,
                    t,
                    cum,
                    p0=p0,
                    bounds=([max(cum[-1], 1.0) * 0.5, 1e-3, -n], [max(cum[-1], 1.0) * 20, 20.0, 2 * n]),
                    maxfev=20000,
                )
            K, r, t0 = (float(v) for v in popt)
            pred = _logistic_cum(t, K, r, t0)
            ss_res = float(np.sum((cum - pred) ** 2))
            ss_tot = float(np.sum((cum - cum.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            if np.isfinite(r2) and r2 >= 0.5:
                return {
                    "onset_slope": float(K * r / 4.0),
                    "inflection_i": float(t0),
                    "r2": float(r2),
                    "method": "logistic_cumulative",
                }
        except Exception:
            pass
    # fallback: log-linear slope of daily volume across the rising limb
    ly = np.log1p(y)
    try:
        coef = np.polyfit(t, ly, 1)
        pred = np.polyval(coef, t)
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "onset_slope": float(coef[0]),
            "inflection_i": np.nan,
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "method": "loglinear_slope",
        }
    except Exception:
        return {"onset_slope": np.nan, "inflection_i": np.nan, "r2": np.nan, "method": "failed"}


def fit_exp_decay(counts: Sequence[float], *, min_points: int = 4) -> dict:
    """Post-peak exponential decay: ln V = a - lambda t. Scoped to an episode."""
    y = np.asarray(counts, dtype=float)
    y = np.where(np.isfinite(y), y, 0.0)
    t = np.arange(y.size, dtype=float)
    m = y > 0
    if m.sum() < min_points:
        return {"decay_lambda": np.nan, "half_life_days": np.nan, "r2": np.nan, "ok": False}
    try:
        coef = np.polyfit(t[m], np.log(y[m]), 1)
    except Exception:
        return {"decay_lambda": np.nan, "half_life_days": np.nan, "r2": np.nan, "ok": False}
    lam = -float(coef[0])
    pred = np.polyval(coef, t[m])
    ly = np.log(y[m])
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if lam <= 0:
        return {"decay_lambda": lam, "half_life_days": np.nan, "r2": r2, "ok": False}
    hl = math.log(2.0) / lam
    return {
        "decay_lambda": lam,
        "half_life_days": float(hl),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "ok": bool(np.isfinite(r2) and r2 >= 0.3 and hl < 365),
    }


def rank_pct(x: Sequence[float]) -> np.ndarray:
    """Percentile rank in [0,1] over finite values; NaN preserved."""
    a = np.asarray(x, dtype=float)
    out = np.full(a.size, np.nan)
    ok = np.isfinite(a)
    n = int(ok.sum())
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.5
        return out
    r = stats.rankdata(a[ok], method="average")
    out[ok] = (r - 0.5) / n
    return out


# --------------------------------------------------------------------------- #
# Self-checks
# --------------------------------------------------------------------------- #
def run_self_checks(*, verbose: bool = False) -> dict:
    """Unit-test-style assertions. Returns {check_name: bool}; all must be True.

    The look-ahead checks are the important ones: they mutate the present and
    the future and assert the trailing baseline for the present is unchanged.
    """
    res: dict[str, bool] = {}
    rng = np.random.default_rng(RNG_SEED)

    # --- 1. trailing baseline excludes the CURRENT day ---------------------
    base = np.array([2, 3, 2, 4, 3, 2, 5, 3, 2, 4, 3, 3, 2, 4, 3, 2, 3, 4, 2, 3], float)
    spiked = base.copy()
    t_star = 15
    spiked[t_star] = 500.0
    f_a = trailing_baseline_fits(base, window=10, min_days=5, min_mean=0.5)
    f_b = trailing_baseline_fits(spiked, window=10, min_days=5, min_mean=0.5)
    res["baseline_excludes_current_day"] = bool(
        f_a[t_star] is not None
        and f_b[t_star] is not None
        and abs(f_a[t_star].mu - f_b[t_star].mu) < 1e-12
        and f_a[t_star].n == f_b[t_star].n
    )
    # the day AFTER must change (proof the window really moves)
    res["baseline_uses_past_day"] = bool(
        f_a[t_star + 1] is not None
        and abs(f_a[t_star + 1].mu - f_b[t_star + 1].mu) > 1.0
    )

    # --- 2. no look-ahead: future mutation cannot change today ------------
    fut = base.copy()
    fut[t_star + 1 :] = 999.0
    f_c = trailing_baseline_fits(fut, window=10, min_days=5, min_mean=0.5)
    res["baseline_no_future_leakage"] = bool(
        abs(f_a[t_star].mu - f_c[t_star].mu) < 1e-12
    )

    # --- 3. causal savgol has no look-ahead -------------------------------
    y = np.sin(np.linspace(0, 6, 40)) * 3 + 5
    d_a = causal_savgol_deriv(y, window=5, order=2, deriv=2)
    y2 = y.copy()
    y2[25:] += 100.0
    d_b = causal_savgol_deriv(y2, window=5, order=2, deriv=2)
    res["causal_savgol_no_future_leakage"] = bool(
        np.allclose(d_a[:25], d_b[:25], equal_nan=True)
    )
    # exactness: 2nd derivative of a pure quadratic is recovered
    q = 3.0 * np.arange(12.0) ** 2 + 5.0 * np.arange(12.0) + 1.0
    dq = causal_savgol_deriv(q, window=5, order=2, deriv=2)
    res["causal_savgol_exact_on_quadratic"] = bool(
        np.allclose(dq[4:], 6.0, atol=1e-6)
    )

    # --- 4. NB fit sanity --------------------------------------------------
    draw = stats.nbinom.rvs(
        5, 5 / (5 + 20), size=4000, random_state=np.random.RandomState(RNG_SEED)
    ).astype(float)
    nb = fit_negbinom(draw)
    res["nb_recovers_mu"] = bool(abs(nb.mu - 20.0) < 1.5)
    res["nb_recovers_theta"] = bool(nb.dist == "negbinom" and 3.0 < nb.theta < 8.0)
    pois_draw = stats.poisson.rvs(
        4.0, size=3000, random_state=np.random.RandomState(RNG_SEED)
    ).astype(float)
    res["nb_falls_back_to_poisson"] = bool(fit_negbinom(pois_draw).dist == "poisson")

    # --- 4b. Drift adjustment: encodes the MEASURED calibration finding so it
    # cannot silently regress. A flat series must be left alone; a rising one must
    # be corrected upward; and on a drifting null the rejection rate must come back
    # toward nominal instead of inflating.
    flat = np.full(28, 4.0)
    res["drift_flat_series_untouched"] = bool(abs(drift_factor(flat) - 1.0) < 1e-9)
    rising = 4.0 * np.exp(0.02 * np.arange(28))
    res["drift_detects_upward_trend"] = bool(drift_factor(rising) > 1.15)
    res["drift_projection_is_clipped"] = bool(
        drift_factor(np.exp(np.arange(28))) <= math.exp(1.5) + 1e-9
    )
    _r = np.random.default_rng(4242)
    _pi, _pa = [], []
    for _ in range(1500):
        lam = 4.0 * np.exp(0.02 * np.arange(29))
        s = _r.poisson(lam)
        b = s[:28].astype(float)
        f0 = fit_negbinom(b)
        _pi.append(exceedance_p(float(s[28]), f0))
        k = drift_factor(b)
        th = f0.theta * k if np.isfinite(f0.theta) else f0.theta
        _pa.append(exceedance_p(float(s[28]), NBFit(f0.mu * k, th, f0.dist, f0.method, f0.n, f0.var)))
    _fi = float(np.mean(np.asarray(_pi) < 0.05))
    _fa = float(np.mean(np.asarray(_pa) < 0.05))
    # Unadjusted must visibly over-reject on a drifting null; adjusted must not.
    res["drift_null_overrejects_unadjusted"] = bool(_fi > 0.08)
    res["drift_null_calibrated_after_adjust"] = bool(_fa < 0.075)

    # No fit may ship a five-figure theta labelled a real MLE: at that scale NB
    # is numerically Poisson (Var = mu + mu^2/theta) but the audit trail lies.
    nearly_pois = rng.poisson(4.0, size=400).astype(float)
    f_np = fit_negbinom(nearly_pois)
    res["no_unidentifiable_theta_labelled_mle"] = bool(
        f_np.dist == "poisson" or (np.isfinite(f_np.theta)
                                   and f_np.theta < _THETA_IDENTIFIABLE_MAX)
    )

    # --- 5. D1 regression test: a single post on a tiny baseline is NOT extreme
    tiny_base = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], float)
    fit = fit_negbinom(tiny_base)
    s1 = surprise_from_p(exceedance_p(1.0, fit))
    s2 = surprise_from_p(exceedance_p(2.0, fit))
    res["d1_single_post_not_extreme"] = bool(s1 < 1.0)  # p > 0.1
    res["d1_two_posts_not_extreme"] = bool(s2 < 2.0)  # p > 0.01
    # and a genuinely large exceedance IS flagged
    res["d1_real_spike_is_extreme"] = bool(
        surprise_from_p(exceedance_p(40.0, fit_negbinom(np.full(20, 2.0) + rng.integers(0, 2, 20)))) > 3.0
    )

    # --- 6. exceedance p is monotone & bounded ----------------------------
    f = fit_negbinom(np.array([3, 5, 2, 8, 4, 6, 3, 5, 7, 4, 5, 3, 6, 4, 5], float))
    ps = [exceedance_p(k, f) for k in range(0, 30)]
    res["exceedance_monotone"] = bool(all(ps[i] >= ps[i + 1] - 1e-15 for i in range(len(ps) - 1)))
    res["exceedance_p0_is_one"] = bool(abs(ps[0] - 1.0) < 1e-9)

    # --- 7. BH q-values ---------------------------------------------------
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = bh_qvalues(p)
    res["bh_monotone"] = bool(np.all(np.diff(q) >= -1e-12))
    res["bh_ge_p"] = bool(np.all(q >= p - 1e-12))
    res["bh_max_le_1"] = bool(np.nanmax(q) <= 1.0)
    res["by_more_conservative"] = bool(np.all(bh_qvalues(p, method="by") >= q - 1e-12))
    res["bh_nan_safe"] = bool(np.isnan(bh_qvalues([0.01, np.nan, 0.5])[1]))
    # uniform p-values under the null should give ~no discoveries
    unif = rng.uniform(size=2000)
    res["bh_null_calibrated"] = bool((bh_qvalues(unif) < 0.05).sum() <= 5)

    # --- 9. AR(1) ---------------------------------------------------------
    phi_true = 0.7
    e = rng.normal(size=600)
    xs = np.zeros(600)
    for i in range(1, 600):
        xs[i] = phi_true * xs[i - 1] + e[i]
    fitted = fit_ar1(xs)
    res["ar1_recovers_phi"] = bool(abs(fitted.phi - phi_true) < 0.08)
    res["ar1_half_life_matches"] = bool(
        abs(fitted.half_life_days - math.log(0.5) / math.log(phi_true)) < 0.6
    )
    res["ar1_ljungbox_detects_ac"] = bool(fitted.ljung_box_p < 0.01)
    res["ar1_ljungbox_clears_white_noise"] = bool(fit_ar1(rng.normal(size=600)).ljung_box_p > 0.01)
    res["ar1_short_series_gated"] = bool(not fit_ar1(np.arange(5.0)).ok)

    # --- 10. burst detection ---------------------------------------------
    quiet = np.full(30, 2.0)
    bursty = np.concatenate([np.full(20, 2.0), np.full(6, 40.0), np.full(8, 3.0)])
    kb = kleinberg_2state(bursty)
    res["kleinberg_finds_burst"] = bool(kb.state[20:26].sum() >= 5 and kb.state[:18].sum() == 0)
    res["kleinberg_quiet_no_burst"] = bool(kleinberg_2state(quiet).state.sum() == 0)
    cus = poisson_cusum(bursty, baseline_mu=np.full(bursty.size, 2.0))
    res["cusum_finds_burst"] = bool(cus.state[20:28].sum() >= 3)
    res["cusum_quiet_no_burst"] = bool(
        poisson_cusum(quiet, baseline_mu=np.full(30, 2.0)).state.sum() == 0
    )
    res["detect_bursts_routes_short_to_cusum"] = bool(
        detect_bursts(np.array([0, 1, 0, 2, 0, 1, 0.0])).method == "poisson_cusum"
    )
    res["detect_bursts_routes_long_to_kleinberg"] = bool(
        detect_bursts(bursty).method == "kleinberg"
    )
    eps = extract_episodes(kb.state, bursty, kb.intensity)
    res["episodes_extracted"] = bool(len(eps) == 1 and eps[0]["peak_V"] == 40.0)

    # --- 11. rank_pct -----------------------------------------------------
    rp = rank_pct([1.0, 2.0, 3.0, np.nan])
    res["rank_pct_ordered"] = bool(rp[0] < rp[1] < rp[2] and np.isnan(rp[3]))

    if verbose:
        for k, v in res.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return res


if __name__ == "__main__":
    checks = run_self_checks(verbose=True)
    n_fail = sum(1 for v in checks.values() if not v)
    print(f"\n{len(checks) - n_fail}/{len(checks)} self-checks passed")
    raise SystemExit(1 if n_fail else 0)
