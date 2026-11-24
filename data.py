"""Data download and estimation for the FIRE planner.

Monthly total returns of one ETF per asset class (Yahoo Finance) are converted to CHF (hedged or
unhedged) and deflated with the Swiss CPI. Expected returns and the covariance matrix are
estimated on these REAL CHF returns and annualised.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# backfill: (proxy ticker, first day of the ETF's own data). Yahoo has no dividends before 2013
# for ETFs listed in Zurich, so before that date SMICHA.SW would be a price return: the SMI
# total-return index (SMIC.SW, dividends included) is used instead.
ASSETS: Dict[str, Dict] = {
    "SPY": dict(name="US equities", asset_class="Equity", ccy="USD"),
    "SMICHA.SW": dict(name="Swiss equities", asset_class="Equity", ccy="CHF",
                      backfill=("SMIC.SW", "2013-01-01")),
    "EEM": dict(name="Emerging markets equities", asset_class="Equity", ccy="USD"),
    "VNQ": dict(name="US real estate (REITs)", asset_class="Real estate", ccy="USD"),
    "IEF": dict(name="Sovereign bonds (US Treasuries 7-10y)", asset_class="Bonds", ccy="USD"),
    "LQD": dict(name="Corporate bonds (investment grade)", asset_class="Bonds", ccy="USD"),
    "TIP": dict(name="Inflation-linked bonds (US TIPS)", asset_class="Bonds", ccy="USD"),
    "GLD": dict(name="Gold", asset_class="Gold", ccy="USD"),
}
UNIVERSE: Dict[str, str] = {t: a["name"] for t, a in ASSETS.items()}
GOLD_TICKER = "GLD"
START_DATE = "2000-01-01"
MIN_MONTHS = 60
MONTHS_PER_YEAR = 12


def download_prices(tickers: List[str], start: str = START_DATE) -> pd.DataFrame:
    """Daily dividend-adjusted close prices from Yahoo Finance up to today, with backfills."""
    import yfinance as yf

    proxies = [ASSETS[t]["backfill"][0] for t in tickers if "backfill" in ASSETS.get(t, {})]
    raw = yf.download(tickers + proxies, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("No price data returned by Yahoo Finance.")
    return apply_backfills(close_prices(raw, tickers + proxies), tickers)


def apply_backfills(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Before its backfill date an asset follows its proxy, rescaled to join at the first common day."""
    out = prices[tickers].copy()
    for t in tickers:
        if "backfill" not in ASSETS.get(t, {}):
            continue
        proxy, date = ASSETS[t]["backfill"]
        both = prices.loc[date:, [t, proxy]].dropna()
        if both.empty:
            continue
        join = both.index[0]
        scale = both[t].iloc[0] / both[proxy].iloc[0]
        out[t] = pd.concat([prices.loc[prices.index < join, proxy] * scale,
                            prices.loc[prices.index >= join, t]])
    return out


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


def month_end_prices(prices: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Prices on the last common trading day of each month (index = that day), common sample,
    without the current (incomplete) month."""
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    common = prices.dropna(how="any")
    last = common.groupby(common.index.to_period("M")).tail(1)
    return last[last.index.to_period("M") < today.to_period("M")]


def _at_dates(daily: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Last known value of a daily series on each date."""
    return daily.reindex(daily.index.union(dates)).ffill().reindex(dates)


def hedge_carry(dates: pd.DatetimeIndex, usd_1y: pd.Series, chf_1y: pd.Series) -> pd.Series:
    """(1 + r_CHF) / (1 + r_USD) on each date: yearly factor of a 1-year forward (covered parity)."""
    months = dates.to_period("M")
    rates = chf_1y.reindex(chf_1y.index.union(months)).ffill().reindex(months)
    return (1.0 + pd.Series(rates.values, index=dates)) / (1.0 + _at_dates(usd_1y, dates))


def currency_facts(month_end: pd.DataFrame, usdchf: pd.Series, usd_1y: pd.Series,
                   chf_1y: pd.Series) -> Dict[str, float]:
    """Sample average and latest yearly effect of the hedge, and yearly drift of USD/CHF."""
    carry = hedge_carry(month_end.index, usd_1y, chf_1y) - 1.0
    S = _at_dates(usdchf, month_end.index)
    drift = (S.iloc[-1] / S.iloc[0]) ** (MONTHS_PER_YEAR / (len(S) - 1)) - 1.0
    return {"carry_mean": float(carry.iloc[:-1].mean()), "carry_last": float(carry.iloc[-1]),
            "fx_drift": float(drift)}


def real_chf_returns(month_end: pd.DataFrame, usdchf: pd.Series, usd_1y: pd.Series, chf_1y: pd.Series,
                     cpi: pd.Series, hedged: bool, assets: Dict[str, Dict] = ASSETS) -> pd.DataFrame:
    """Monthly real returns in CHF from month-end local prices.

    Unhedged (USD assets):  (1 + R) S_{t+1} / S_t - 1,  S = CHF per USD on the return dates.
    Hedged (USD assets):    (1 + R) ((1 + r_CHF) / (1 + r_USD))^dt - 1, dt = 1/12, 1-year rates known
                            at the start of the period (covered interest parity, forward rolled monthly).
    CHF assets are never hedged. Then divided by the growth of the Swiss CPI. Months whose CPI is
    not published yet are dropped. Monthly inputs (chf_1y, cpi) have a period index; a missing
    recent value of chf_1y is replaced by the last known one.
    """
    nominal = month_end.pct_change(fill_method=None)
    months = month_end.index.to_period("M")

    if hedged:
        carry = hedge_carry(month_end.index, usd_1y, chf_1y)
        factor = (carry ** (1.0 / MONTHS_PER_YEAR)).shift(1)
    else:
        S = _at_dates(usdchf, month_end.index)
        factor = S / S.shift(1)
    usd = [t for t in month_end.columns if assets[t]["ccy"] == "USD"]
    nominal[usd] = (1.0 + nominal[usd]).mul(factor, axis=0) - 1.0

    index = pd.Series(cpi.reindex(months).values, index=month_end.index)
    real = (1.0 + nominal).div(index / index.shift(1), axis=0) - 1.0
    return real.dropna(how="any")


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
