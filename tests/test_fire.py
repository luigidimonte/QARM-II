"""Sanity tests for fire.py (synthetic moments, no network)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fire  # noqa: E402
import optimizer as opt  # noqa: E402
from tests.test_optimizer import TICKERS, assert_feasible  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    mu = np.array([0.10, 0.08, 0.09, 0.03, 0.04, 0.07])
    vol = np.array([0.15, 0.14, 0.22, 0.07, 0.08, 0.17])
    sigma = np.diag(vol ** 2) + 0.3 * np.outer(vol, vol) * (1 - np.eye(6))
    bounds = opt.frontier_bounds(mu, sigma, TICKERS)
    gammas = opt.gamma_grid(mu, sigma, TICKERS, n_grid=10, bounds=bounds)
    W = opt.resample_weights(mu / 12, sigma / 12, 120, gammas, TICKERS, n_resamples=10, seed=3)
    rs_bounds = opt.resampled_frontier_bounds(gammas, W, sigma)
    gb = fire.gamma_bounds_from_sigma(sigma, gammas, W, rs_bounds)
    return mu, sigma, rs_bounds, gammas, W, gb


PROFILE = fire.ClientProfile(age=30, income=120_000, expenses=60_000, wealth=100_000,
                             target_age=50, swr=0.04, rf_nominal=0.01, inflation=0.015)


def test_fire_target_and_savings():
    assert fire.fire_target(60_000, 0.04) == pytest.approx(1_500_000)
    assert PROFILE.savings == 60_000


def test_pv_savings_limits():
    assert fire.pv_savings(10.0, 5, 0.0) == pytest.approx(50.0)
    assert fire.pv_savings(10.0, 5, 1e-12) == pytest.approx(50.0)
    assert fire.pv_savings(10.0, 5, 1e-6) == pytest.approx(50.0, rel=1e-5)
    assert fire.pv_savings(10.0, 5, 0.03) == pytest.approx(sum(10 / 1.03 ** k for k in range(1, 6)))
    assert fire.pv_savings(10.0, -3, 0.03) == 0.0


def test_dynamic_gamma_bounded_and_min_when_fire_reached():
    rng = np.random.default_rng(0)
    g0, gmin, K = 2.0, 0.3, 1_000_000
    for _ in range(2000):
        W, H = rng.uniform(0, 2e6), rng.uniform(0, 2e6)
        g = fire.dynamic_gamma(W, H, K, g0, gmin)
        assert gmin - 1e-12 <= g <= g0 + 1e-12
        if W >= K:
            assert g == gmin
    assert fire.dynamic_gamma(0.0, 1e5, K, g0, gmin) == pytest.approx(g0)   # all human capital
    assert fire.dynamic_gamma(5e5, 0.0, K, g0, gmin) == pytest.approx(gmin)  # no savings left


def test_gamma_bounds_from_sigma(setup):
    mu, sigma, bounds, gammas, W, gb = setup
    sig_minus, sig_plus, _ = bounds
    assert sig_minus <= gb["sigma_min"] <= gb["sigma_max"] <= sig_plus
    assert gb["gamma_min"] <= gb["gamma_0"]
    clipped = fire.gamma_bounds_from_sigma(sigma, gammas, W, bounds, sigma_min=0.0, sigma_max=9.9)
    assert clipped["sigma_min"] == sig_minus and clipped["sigma_max"] == sig_plus
    assert clipped["notes"]


def test_moment_recursion_deterministic_when_no_risk():
    m, s = np.full(10, 0.05), np.zeros(10)
    res = fire.fire_probability_lognormal(1000.0, 100.0, m, s, K=2000.0)
    W = [1000.0]
    for _ in range(9):
        W.append(W[-1] * 1.05 + 100.0)
    np.testing.assert_allclose(res["M1"], W)
    np.testing.assert_allclose(res["M2"], np.array(W) ** 2)
    np.testing.assert_array_equal(res["P"], (np.array(W) >= 2000.0).astype(float))


def test_probability_increases_with_savings():
    m, s = np.full(20, 0.04), np.full(20, 0.10)
    low = fire.fire_probability_lognormal(1e5, 2e4, m, s, 1e6)["P"]
    high = fire.fire_probability_lognormal(1e5, 4e4, m, s, 1e6)["P"]
    assert np.all((low >= 0) & (low <= 1))
    assert np.all(high >= low - 1e-12)


def test_glide_path_feasible_and_consistent(setup):
    mu, sigma, _, gammas, W, gb = setup
    path = fire.simulate_glide_path(PROFILE, mu, sigma, TICKERS, gammas, W, gb["gamma_0"], gb["gamma_min"])
    assert path["age"].iloc[0] == PROFILE.age and path["age"].iloc[-1] == fire.horizon_end_age(PROFILE)
    assert_feasible(fire.weights_table(path, TICKERS).values)            # every portfolio on the path
    assert ((path["gamma"] >= gb["gamma_min"] - 1e-12) & (path["gamma"] <= gb["gamma_0"] + 1e-12)).all()
    assert (path.loc[path["W"] >= path["K"], "gamma"] == gb["gamma_min"]).all()
    assert (path.loc[path["age"] >= PROFILE.target_age, "gamma"] == gb["gamma_min"]).all()
    np.testing.assert_allclose(path["M1"], path["W"], rtol=1e-10)          # expected path = M1
    trades = fire.weights_table(path, TICKERS, "trade_amt_").iloc[1:]
    np.testing.assert_allclose(trades.sum(axis=1), PROFILE.savings, rtol=1e-9)  # trades invest c
    summary = fire.fire_summary(path, PROFILE.target_age)
    assert 0.0 <= summary["p_target"] <= 1.0


def test_default_risk_range_gives_a_moving_glide_path(setup):
    """Regression: with the default sigma range the weights must change between today and FIRE."""
    mu, sigma, _, gammas, W, gb = setup
    assert not fire.is_narrow_risk_range(gb["sigma_min"], gb["sigma_max"])
    assert gb["gamma_0"] > gb["gamma_min"] and not gb["notes"]
    path = fire.simulate_glide_path(PROFILE, mu, sigma, TICKERS, gammas, W, gb["gamma_0"], gb["gamma_min"])
    w = fire.weights_table(path, TICKERS)
    assert path["gamma"].iloc[0] > path["gamma"].iloc[-1]
    assert path["s"].iloc[0] > path["s"].iloc[-1] + 0.005               # risk falls by > 0.5pp
    assert np.abs(w.iloc[0] - w.iloc[-1]).max() > 0.05                  # some weight moves by > 5pp


def test_narrow_risk_range_flags_flat_glide_path(setup):
    mu, sigma, bounds, gammas, W, _ = setup
    sig_minus = bounds[0]
    gb = fire.gamma_bounds_from_sigma(sigma, gammas, W, bounds, sig_minus, sig_minus)
    assert fire.NARROW_RANGE_NOTE in gb["notes"]
    path = fire.simulate_glide_path(PROFILE, mu, sigma, TICKERS, gammas, W, gb["gamma_0"], gb["gamma_min"])
    w = fire.weights_table(path, TICKERS)
    assert np.abs(w.iloc[0] - w.iloc[-1]).max() < 1e-9                  # flat: the reported symptom
    assert fire.is_narrow_risk_range(0.07, 0.074) and not fire.is_narrow_risk_range(0.07, 0.075)


def test_held_portfolio_volatility_equals_risk_targets(setup):
    """Fix 4: the resampled weights actually held at gamma_0 / gamma_min have exactly the target
    volatilities, and a client with no wealth yet (H/(W+H) = 1, so gamma_t = gamma_0) starts with
    real volatility sigma_max / (1 + inflation)."""
    mu, sigma, _, gammas, W, gb = setup
    for g, target in ((gb["gamma_0"], gb["sigma_max"]), (gb["gamma_min"], gb["sigma_min"])):
        assert opt.resampled_vol(g, gammas, W, sigma) == pytest.approx(target, abs=1e-8)
    start = PROFILE.with_changes(wealth=0.0)
    path = fire.simulate_glide_path(start, mu, sigma, TICKERS, gammas, W, gb["gamma_0"], gb["gamma_min"])
    assert path["gamma"].iloc[0] == pytest.approx(gb["gamma_0"])
    assert path["s_nominal"].iloc[0] == pytest.approx(gb["sigma_max"], abs=1e-8)
    assert path["s"].iloc[0] == pytest.approx(gb["sigma_max"] / (1 + start.inflation), abs=1e-8)
    assert path["s_nominal"].iloc[-1] == pytest.approx(gb["sigma_min"], abs=1e-8)
    assert_feasible(fire.weights_table(path, TICKERS).values)            # 40% cap, 10% gold kept


def test_validate_profile():
    errors, _ = fire.validate_profile(PROFILE.with_changes(target_age=30))
    assert errors
    _, warnings = fire.validate_profile(PROFILE.with_changes(income=50_000))
    assert warnings
    assert fire.validate_profile(PROFILE) == ([], [])
