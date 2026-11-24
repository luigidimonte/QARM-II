"""Market inputs besides prices: USD/CHF, US 1-year rate, Swiss Confederation curve, Swiss CPI.

Every series is downloaded (FRED CSV endpoint, SNB data portal). If a download fails, the local
copy in data/ is used and a warning with its last date is returned, to be shown in the UI.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SNB_URL = "https://data.snb.ch/api/cube/{}/data/csv/en"
# `rendoblim` stopped in July 2025. `rendeiduebm` is its continuation: identical values for the
# CHF Confederation bonds on every common month and maturity (it has no 15-year node).
SNB_CURVE_CUBE = "rendeiduebm"
SNB_CPI_CUBE, SNB_CPI_SERIES = "plkopr", "LD2010100"
TIMEOUT = 30

# (maturity in years, yearly spot rate as a decimal), by increasing maturity
Yields = Sequence[Tuple[float, float]]


# ----------------------------------------------------------------------------- parsing
def parse_fred(text: str, series: str) -> pd.Series:
    """Daily FRED series; holidays (empty values) are forward-filled."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    out = pd.Series(pd.to_numeric(df[series]).values, index=pd.to_datetime(df.iloc[:, 0]), name=series)
    return out.ffill().dropna()


def _read_snb(text: str) -> pd.DataFrame:
    """SNB format: two header lines, an empty line, then a ';'-separated table with a header row."""
    start = text.index('"Date"')
    return pd.read_csv(io.StringIO(text[start:]), sep=";")


def parse_snb_curve(text: str) -> pd.DataFrame:
    """Confederation spot rates (decimals): one row per month (period index), one column per
    maturity in years. Reads both cubes: rendeiduebm (D0 = issuer, D1 = maturity) and rendoblim
    (D0 = maturity)."""
    df = _read_snb(text)
    if "D1" in df.columns:
        df = df[df["D0"] == "CHF"].rename(columns={"D1": "maturity"})
    else:
        df = df.rename(columns={"D0": "maturity"})
    df = df[df["maturity"].str.fullmatch(r"\d+J")]
    table = df.assign(T=df["maturity"].str[:-1].astype(int)).pivot(index="Date", columns="T", values="Value")
    table.index = pd.PeriodIndex(table.index, freq="M")
    return table.sort_index(axis=1) / 100.0


def parse_snb_cpi(text: str) -> pd.Series:
    """Swiss CPI, monthly index (period index)."""
    df = _read_snb(text)
    df = df[df["D0"] == SNB_CPI_SERIES].dropna(subset=["Value"])
    return pd.Series(df["Value"].astype(float).values, index=pd.PeriodIndex(df["Date"], freq="M"), name="CPI")


# ----------------------------------------------------------------------------- download with fallback
def _download(url: str) -> str:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content.decode("utf-8-sig")


def _load(label: str, url: str, cache_file: str, parse: Callable):
    """(parsed data, warning or None): download, else the local copy."""
    try:
        data = parse(_download(url))
        if len(data) == 0:
            raise ValueError("empty series")
        return data, None
    except Exception as e:
        with open(os.path.join(CACHE_DIR, cache_file), encoding="utf-8-sig") as f:
            data = parse(f.read())
        last = data.index[-1]
        when = f"{last:%d %b %Y}" if isinstance(last, pd.Timestamp) else last.strftime("%b %Y")
        return data, f"{label}: download failed ({type(e).__name__}), using the local copy up to {when}."


@dataclass(frozen=True)
class Market:
    usdchf: pd.Series          # CHF per 1 USD, daily
    usd_1y: pd.Series          # US 1-year Treasury yield (decimal), daily
    chf_curve: pd.DataFrame    # Confederation spot rates (decimal), months x maturities
    cpi: pd.Series             # Swiss CPI index, monthly
    warnings: Tuple[str, ...]

    @property
    def curve_date(self) -> pd.Timestamp:
        return self.chf_curve.index[-1].to_timestamp()

    @property
    def yields(self) -> Yields:
        """Latest Confederation curve as (maturity, spot rate) nodes."""
        row = self.chf_curve.iloc[-1].dropna()
        return tuple((float(T), float(y)) for T, y in row.items())


def load_market() -> Market:
    """Download everything (local copies if a source is unreachable)."""
    fx, w1 = _load("USD/CHF (FRED DEXSZUS)", FRED_URL.format("DEXSZUS"), "DEXSZUS.csv",
                   lambda t: parse_fred(t, "DEXSZUS"))
    usd, w2 = _load("US 1-year rate (FRED DGS1)", FRED_URL.format("DGS1"), "DGS1.csv",
                    lambda t: parse_fred(t, "DGS1") / 100.0)
    curve, w3 = _load("Swiss Confederation curve (SNB)", SNB_URL.format(SNB_CURVE_CUBE), "snb_rendoblim.csv",
                      parse_snb_curve)
    cpi, w4 = _load("Swiss CPI (SNB plkopr)", SNB_URL.format(SNB_CPI_CUBE), "snb_plkopr.csv", parse_snb_cpi)
    return Market(fx, usd, curve, cpi, tuple(w for w in (w1, w2, w3, w4) if w))


# ----------------------------------------------------------------------------- risk-free rate
def zero_coupon(h: float, yields: Yields) -> float:
    """Z(h) by the interpolation of the Fixed Income and Credit Risk course.

    Z_i = (1 + y_i)^(-T_i) (spot rates). With T_{i-1} <= h <= T_i, Delta = T_i - T_{i-1} and
    tau = T_i - h:  Z(h) = Z_{i-1} tau / Delta + Z_i (Delta - tau) / Delta.
    h is clamped to the range of the nodes.
    """
    T, y = np.asarray(yields, dtype=float).T
    Z = (1.0 + y) ** (-T)
    h = float(np.clip(h, T[0], T[-1]))
    i = int(np.searchsorted(T, h))
    if T[i] == h:
        return float(Z[i])
    delta, tau = T[i] - T[i - 1], T[i] - h
    return float(Z[i - 1] * tau / delta + Z[i] * (delta - tau) / delta)


def nominal_rate(h: float, yields: Yields) -> float:
    """r(h) = Z(h)^(-1/h) - 1 (discrete compounding); the node's own yield if h is a node."""
    T, y = np.asarray(yields, dtype=float).T
    h = float(np.clip(h, T[0], T[-1]))
    i = int(np.searchsorted(T, h))
    if T[i] == h:
        return float(y[i])
    return float(zero_coupon(h, yields) ** (-1.0 / h) - 1.0)


@dataclass(frozen=True)
class RiskFree:
    """Real risk-free rate by horizon: Confederation curve deflated with expected inflation,
    or a flat real rate if `override` is given."""
    yields: Yields
    inflation: float
    override: Optional[float] = None

    def real(self, h: float) -> float:
        """r_f(h) = (1 + r(h)) / (1 + pi) - 1."""
        if self.override is not None:
            return float(self.override)
        return (1.0 + nominal_rate(h, self.yields)) / (1.0 + self.inflation) - 1.0
