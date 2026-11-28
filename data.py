"""Data download and estimation for the FIRE planner.

Monthly total returns of one ETF per asset class (Yahoo Finance) are converted to CHF (hedged or
unhedged) and deflated with the Swiss CPI. Expected returns and the covariance matrix are
estimated on these REAL CHF returns and annualised.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    if len(rets) < min_months:
        problems.append(f"Only {len(rets)} months of common history (need at least {min_months}).")
    return problems


def estimate_moments(rets: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, int]:
    """Annualised expected returns (12 x monthly mean) and covariance (12 x monthly cov)."""
    mu = rets.mean() * MONTHS_PER_YEAR
    sigma = rets.cov() * MONTHS_PER_YEAR
    return mu, sigma, len(rets)

