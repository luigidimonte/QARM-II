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


def test_monthly_returns_drop_current_month_and_common_sample():
    days = pd.bdate_range("2020-01-01", "2020-06-15")
    prices = pd.DataFrame({"A": np.linspace(100, 120, len(days)),
                           "B": np.linspace(50, 55, len(days))}, index=days)
    prices.loc[:"2020-01-20", "B"] = np.nan          # B starts later
    rets = data.monthly_returns(prices, today=pd.Timestamp("2020-06-15"))
    assert rets.index[-1].month == 5                 # June (in progress) dropped
    assert not rets.isna().any().any()
    assert len(rets) == 4                            # Feb..May (Jan is the base month)


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
    assert any("Gold" in p for p in data.validate_returns(monthly[no_gold], no_gold))
    assert any("months" in p for p in data.validate_returns(monthly.iloc[:30], TICKERS))


def test_fisher_conversion():
    assert data.to_real(0.03, 0.0) == pytest.approx(0.03)
    assert data.to_real(0.03, 0.02) == pytest.approx(1.03 / 1.02 - 1)
    assert data.vol_to_real(0.15, 0.02) == pytest.approx(0.15 / 1.02)


def test_constant_correlation_target_diag(monthly):
    S = monthly.cov().values
    F, rbar = data.constant_correlation_target(S)
    np.testing.assert_allclose(np.diag(F), np.diag(S))
    np.testing.assert_allclose(F, F.T)
    assert -1.0 <= rbar <= 1.0


def _covcor_reference(x: np.ndarray) -> float:
    """Independent vectorised port of Ledoit & Wolf's published covCor algorithm."""
    t, n = x.shape
    y = x - x.mean(axis=0)
    sample = y.T @ y / t
    var = np.diag(sample)
    sqrtvar = np.sqrt(var)
    rbar = (np.sum(sample / np.outer(sqrtvar, sqrtvar)) - n) / (n * (n - 1))
    prior = rbar * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(prior, var)
    y2 = y ** 2
    phi_mat = y2.T @ y2 / t - 2 * (y.T @ y) * sample / t + sample ** 2
    phi = phi_mat.sum()
    term1 = (y ** 3).T @ y / t
    help_ = y.T @ y / t
    term2 = np.diag(help_)[:, None] * sample
    term3 = help_ * var[:, None]
    term4 = var[:, None] * sample
    theta = term1 - term2 - term3 + term4
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(phi_mat) + rbar * np.sum(np.outer(1 / sqrtvar, sqrtvar) * theta)
    gamma = np.sum((sample - prior) ** 2)
    return float(max(0.0, min(1.0, (phi - rho) / gamma / t)))


def test_ledoit_wolf_delta_matches_reference_and_bounds(monthly):
    delta = data.ledoit_wolf_delta(monthly)
    assert 0.0 <= delta <= 1.0
    assert delta == pytest.approx(_covcor_reference(monthly.values), rel=1e-10, abs=1e-12)


def test_shrunk_covariance_is_convex_combination(monthly):
    _, sigma, _ = data.estimate_moments(monthly)
    shrunk, delta = data.shrunk_covariance(monthly, sigma)
    F, _ = data.constant_correlation_target(sigma.values)
    np.testing.assert_allclose(shrunk.values, delta * F + (1 - delta) * sigma.values)
    np.testing.assert_allclose(np.diag(shrunk.values), np.diag(sigma.values))
