"""Data download and estimation for the FIRE planner.

Monthly returns of one liquid ETF per asset class are downloaded fresh from
Yahoo Finance; expected returns and the covariance matrix are annualised.
Everything is in one currency (FX ignored); conversion to real terms uses
the exact Fisher relation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

UNIVERSE: Dict[str, str] = {
    "SPY": "US equities",
    "EWL": "Swiss equities",
    "EEM": "Emerging markets equities",
    "IEF": "Sovereign bonds (US Treasuries 7-10y)",
    "LQD": "Corporate bonds (investment grade)",
    "GLD": "Gold",
}
GOLD_TICKER = "GLD"
START_DATE = "2000-01-01"
MIN_MONTHS = 60
MONTHS_PER_YEAR = 12


def download_prices(tickers: List[str], start: str = START_DATE) -> pd.DataFrame:
    """Download daily dividend-adjusted close prices from Yahoo Finance up to today."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("No price data returned by Yahoo Finance.")
    return close_prices(raw, tickers)


def close_prices(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Close prices with columns in the order of `tickers` (yfinance sorts them alphabetically).

    Tickers without data become all-NaN columns so that validation can report them.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    elif "Close" in raw.columns and len(tickers) == 1:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        prices = raw
    prices = prices.reindex(columns=tickers)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


def _month_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Last available price of each calendar month."""
    try:
        return prices.resample("ME").last()
    except ValueError:  # pandas < 2.2 does not know the "ME" alias
        return prices.resample("M").last()


def monthly_returns(prices: pd.DataFrame, today: pd.Timestamp = None) -> pd.DataFrame:
    """Simple monthly returns on the common sample, without the current (incomplete) month."""
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    common = prices.dropna(how="any")
    monthly = _month_end_prices(common)
    monthly = monthly[monthly.index.to_period("M") < today.to_period("M")]
    return monthly.pct_change(fill_method=None).dropna(how="any")


def validate_returns(rets: pd.DataFrame, tickers: List[str],
                     min_months: int = MIN_MONTHS) -> List[str]:
    """Return a list of human-readable problems with the return data (empty if fine)."""
    problems: List[str] = []
    missing = [t for t in tickers if t not in rets.columns or rets[t].isna().all()]
    if missing:
        problems.append(f"No data for: {', '.join(missing)}.")
    if GOLD_TICKER not in tickers or GOLD_TICKER in missing:
        problems.append(f"Gold ({GOLD_TICKER}) must be in the universe for the 10% gold floor.")
    if len(rets) < min_months:
        problems.append(f"Only {len(rets)} months of common history (need at least {min_months}).")
    return problems


def estimate_moments(rets: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, int]:
    """Annualised expected returns (12 x monthly mean) and covariance (12 x monthly cov)."""
    mu = rets.mean() * MONTHS_PER_YEAR
    sigma = rets.cov() * MONTHS_PER_YEAR
    return mu, sigma, len(rets)


def to_real(nominal, inflation: float):
    """Exact Fisher conversion of a nominal rate (scalar or array) into a real rate."""
    return (1.0 + nominal) / (1.0 + inflation) - 1.0


def vol_to_real(vol, inflation: float):
    """Volatility of the real return (1+R)/(1+pi) - 1 for a deterministic inflation pi."""
    return vol / (1.0 + inflation)


# --------------------------------------------------------------------------------------
# Ledoit & Wolf (2003), "Honey, I Shrunk the Sample Covariance Matrix",
# constant-correlation target (Appendix A, eq. 3) and shrinkage intensity (Appendix B, eq. 5).
# Used only as a comparison in the technical details of the app.
# --------------------------------------------------------------------------------------

def constant_correlation_target(S: np.ndarray) -> Tuple[np.ndarray, float]:
    """Target F: f_ii = s_ii, f_ij = rbar * sqrt(s_ii s_jj) (eq. 3). Returns (F, rbar)."""
    n = S.shape[0]
    sd = np.sqrt(np.diag(S))
    r = S / np.outer(sd, sd)
    rbar = (r.sum() - n) / (n * (n - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))
    return F, rbar


def ledoit_wolf_delta(monthly: pd.DataFrame) -> float:
    """Estimated optimal shrinkage intensity delta* = max(0, min(kappa/T, 1)) (eq. 5)."""
    Y = np.asarray(monthly, dtype=float)
    T, n = Y.shape
    Xc = Y - Y.mean(axis=0)
    S = Xc.T @ Xc / T                       # sample covariance with 1/T, as in the paper
    F, rbar = constant_correlation_target(S)

    # pi_hat = sum_ij (1/T) sum_t {(y_it - ybar_i)(y_jt - ybar_j) - s_ij}^2
    pi_mat = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            pi_mat[i, j] = np.mean((Xc[:, i] * Xc[:, j] - S[i, j]) ** 2)
    pi_hat = pi_mat.sum()

    # theta[i, j] = (1/T) sum_t {(y_it - ybar_i)^2 - s_ii}{(y_it - ybar_i)(y_jt - ybar_j) - s_ij}
    theta = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            theta[i, j] = np.mean((Xc[:, i] ** 2 - S[i, i]) * (Xc[:, i] * Xc[:, j] - S[i, j]))

    # rho_hat = sum_i pi_ii + sum_{i != j} rbar/2 (sqrt(s_jj/s_ii) theta_ii,ij + sqrt(s_ii/s_jj) theta_jj,ij)
    rho_hat = np.trace(pi_mat)
    for i in range(n):
        for j in range(n):
            if i != j:
                rho_hat += rbar / 2.0 * (np.sqrt(S[j, j] / S[i, i]) * theta[i, j]
                                         + np.sqrt(S[i, i] / S[j, j]) * theta[j, i])

    gamma_hat = np.sum((F - S) ** 2)
    kappa_hat = (pi_hat - rho_hat) / gamma_hat
    return float(max(0.0, min(kappa_hat / T, 1.0)))


def shrunk_covariance(monthly: pd.DataFrame, sigma_annual: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Shrinkage estimator delta*F + (1-delta)*S (eq. 2) applied to the annualised sample covariance.

    delta* is estimated on monthly data exactly as in the paper; it does not depend on the
    scale of returns, so it is applied to the annualised covariance used everywhere else.
    """
    delta = ledoit_wolf_delta(monthly)
    S = np.asarray(sigma_annual, dtype=float)
    F, _ = constant_correlation_target(S)
    shrunk = delta * F + (1.0 - delta) * S
    return pd.DataFrame(shrunk, index=sigma_annual.index, columns=sigma_annual.columns), delta
