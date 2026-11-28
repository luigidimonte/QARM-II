"""Sanity tests for data.py (synthetic data only, no network)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data  # noqa: E402

TICKERS = list(data.UNIVERSE)


@pytest.fixture
def monthly() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(TICKERS)
    A = rng.normal(size=(n, n)) * 0.02
    cov = A @ A.T + np.eye(n) * 1e-4
    rets = rng.multivariate_normal(np.full(n, 0.005), cov, size=240)
    idx = pd.date_range("2005-01-31", periods=240, freq="30D")
    return pd.DataFrame(rets, index=idx, columns=TICKERS)


def test_estimate_moments_annualised(monthly):
    mu, sigma, T = data.estimate_moments(monthly)
    assert T == len(monthly)
    np.testing.assert_allclose(mu.values, monthly.mean().values * 12)
    np.testing.assert_allclose(sigma.values, monthly.cov().values * 12)


def test_month_end_prices_drop_current_month_and_common_sample():
    days = pd.bdate_range("2020-01-01", "2020-06-15")
    prices = pd.DataFrame({"A": np.linspace(100, 120, len(days)),
                           "B": np.linspace(50, 55, len(days))}, index=days)
    prices.loc[:"2020-02-10", "B"] = np.nan          # B starts later
    prices.loc["2020-05-29", "A"] = np.nan           # holiday of one exchange at month-end
    me = data.month_end_prices(prices, today=pd.Timestamp("2020-06-15"))
    assert [str(d.date()) for d in me.index] == ["2020-02-28", "2020-03-31", "2020-04-30", "2020-05-28"]
    assert not me.isna().any().any()


# --------------------------------------------------------------- CHF conversion, hedging, deflation
ASSETS = {"U": dict(ccy="USD"), "C": dict(ccy="CHF")}
DATES = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
ME = pd.DataFrame({"U": [100.0, 110.0, 99.0], "C": [50.0, 51.0, 51.0]}, index=DATES)
FX = pd.Series([0.95, 0.90, 0.99], index=pd.to_datetime(["2020-01-30", "2020-02-28", "2020-03-31"]))
USD1Y = pd.Series([0.05, 0.04, 0.03], index=DATES)
CHF1Y = pd.Series([0.01, 0.00, 0.02], index=pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M"))
FLAT_CPI = pd.Series(100.0, index=CHF1Y.index)


def chf(hedged, cpi=FLAT_CPI, chf_1y=CHF1Y):
    return data.real_chf_returns(ME, FX, USD1Y, chf_1y, cpi, hedged, ASSETS)


def test_unhedged_fx_conversion_by_hand():
    r = chf(False)                                   # the 31 Jan rate is the one of 30 Jan (filled)
    np.testing.assert_allclose(r["U"], [1.10 * 0.90 / 0.95 - 1, 0.90 * 0.99 / 0.90 - 1])
    np.testing.assert_allclose(r["C"], [0.02, 0.0])  # CHF asset untouched


def test_hedged_conversion_by_hand_uses_rates_at_the_start_of_the_period():
    r = chf(True)
    np.testing.assert_allclose(r["U"], [1.10 * (1.01 / 1.05) ** (1 / 12) - 1, 0.90 * (1.00 / 1.04) ** (1 / 12) - 1])
    np.testing.assert_allclose(r["C"], [0.02, 0.0])  # CHF assets are never hedged


def test_stale_chf_rate_is_carried_forward():
    r = chf(True, chf_1y=CHF1Y.iloc[:1])
    assert r["U"].iloc[1] == pytest.approx(0.90 * (1.01 / 1.04) ** (1 / 12) - 1)


def test_deflation_and_unpublished_cpi():
    cpi = pd.Series([100.0, 102.0], index=CHF1Y.index[:2])            # March not published yet
    r = chf(False, cpi=cpi)
    assert len(r) == 1
    assert r["C"].iloc[0] == pytest.approx(1.02 / 1.02 - 1, abs=1e-15)
    assert r["U"].iloc[0] == pytest.approx(1.10 * 0.90 / 0.95 / 1.02 - 1)


def test_backfill_joins_without_a_jump(monkeypatch):
    monkeypatch.setattr(data, "ASSETS", {"E": dict(ccy="CHF", backfill=("IDX", "2020-01-03"))})
    days = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"E": [1.0, 1.0, 10.0, 11.0], "IDX": [400.0, 440.0, 500.0, 900.0]}, index=days)
    out = data.apply_backfills(prices, ["E"])
    assert list(out.columns) == ["E"]
    np.testing.assert_allclose(out["E"].pct_change().dropna(), [0.10, 500 / 440 - 1, 0.10])


def test_universe_configuration():
    for t in ("TIP", "GLD", "VNQ"):
        assert t in data.ASSETS
    assert "EWL" not in data.ASSETS and "^SSMI" not in data.ASSETS
    assert {a["ccy"] for a in data.ASSETS.values()} <= {"USD", "CHF"}
    assert [t for t, a in data.ASSETS.items() if a["ccy"] == "CHF"] == ["SMICHA.SW"]
    for a in data.ASSETS.values():
        assert {"name", "asset_class", "ccy"} <= set(a)


def test_close_prices_follow_requested_ticker_order():
    days = pd.bdate_range("2020-01-01", periods=5)
    cols = pd.MultiIndex.from_product([["Close", "Open"], sorted(TICKERS)])  # yfinance: alphabetical
    raw = pd.DataFrame(np.arange(5 * len(cols), dtype=float).reshape(5, -1), index=days, columns=cols)
    prices = data.close_prices(raw, TICKERS)
    assert list(prices.columns) == TICKERS
    for t in TICKERS:
        pd.testing.assert_series_equal(prices[t], raw[("Close", t)], check_names=False)
    with_missing = data.close_prices(raw, TICKERS + ["XYZ"])
    assert with_missing["XYZ"].isna().all()


def test_validate_returns_flags_problems(monthly):
    assert data.validate_returns(monthly, TICKERS) == []
    no_gold = [t for t in TICKERS if t != "GLD"]
    assert data.validate_returns(monthly[no_gold], no_gold) == []          # gold is optional
    assert any("No data for: XYZ" in p for p in data.validate_returns(monthly, TICKERS + ["XYZ"]))
    assert any("months" in p for p in data.validate_returns(monthly.iloc[:30], TICKERS))

