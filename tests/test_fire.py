"""Sanity tests for fire.py (synthetic moments, no network)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fire  # noqa: E402
import optimizer as opt  # noqa: E402
from market import RiskFree, nominal_rate  # noqa: E402
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
                             target_age=50, swr=0.04)
RF = RiskFree(((1.0, 0.01),), 0.015)                  # flat curve: real r_f = 1.01 / 1.015 - 1
N = 4000


def make_grid(setup) -> fire.ResampledGrid:
    mu, sigma, _, gammas, W, gb = setup
    return fire.ResampledGrid(tuple(TICKERS), gammas, W, mu, sigma, gb["gamma_0"], gb["gamma_min"])


@pytest.fixture(scope="module")
def grid(setup):
    return make_grid(setup)


def constant_grid(mu, sigma) -> fire.ResampledGrid:
    """Same weights at every gamma."""
    x = np.array([0.3, 0.1, 0.1, 0.2, 0.1, 0.2])
    return fire.ResampledGrid(tuple(TICKERS), np.array([0.0, 1.0]), np.array([x, x]), mu, sigma, 0.8, 0.2)


def test_fire_target_and_savings():
    assert fire.fire_target(60_000, 0.04) == pytest.approx(1_500_000)
    assert PROFILE.savings == 60_000


def test_human_capital_limits():
    assert fire.human_capital(10.0, 5, 0.0) == pytest.approx(50.0)
    assert fire.human_capital(10.0, 5, 1e-12) == pytest.approx(50.0)
    assert fire.human_capital(10.0, 5, 1e-6) == pytest.approx(50.0, rel=1e-5)
    assert fire.human_capital(10.0, 5, 0.03) == pytest.approx(sum(10 / 1.03 ** k for k in range(1, 6)))
    assert fire.human_capital(10.0, -3, 0.03) == 0.0


def test_gamma_t_bounded_and_min_when_fire_reached():
    rng = np.random.default_rng(0)
    g0, gmin, K = 2.0, 0.3, 1_000_000
    for _ in range(2000):
        W, c, h = rng.uniform(0, 2e6), rng.uniform(0, 1e5), rng.integers(0, 30)
        g = fire.gamma_t(W, c, h, 0.01, K, g0, gmin)
        assert gmin - 1e-12 <= g <= g0 + 1e-12
        if W >= K:
            assert g == gmin
    assert fire.gamma_t(0.0, 1e5, 10, 0.01, K, g0, gmin) == pytest.approx(g0)   # all human capital
    assert fire.gamma_t(5e5, 0.0, 10, 0.01, K, g0, gmin) == pytest.approx(gmin)  # no savings left


@pytest.mark.parametrize("h", [0, -1, -15])
def test_gamma_t_at_and_after_fire_age_is_gamma_min_without_rf(h):
    assert fire.gamma_t(1e5, 5e4, h, float("nan"), 1e6, 2.0, 0.3) == 0.3
    np.testing.assert_array_equal(fire.gamma_t(np.array([0.0, 1e5, 2e6]), 5e4, h, None, 1e6, 2.0, 0.3), 0.3)


def test_gamma_bounds_from_sigma(setup):
    mu, sigma, bounds, gammas, W, gb = setup
    sig_minus, sig_plus, _ = bounds
    assert sig_minus <= gb["sigma_min"] <= gb["sigma_max"] <= sig_plus
    assert gb["gamma_min"] <= gb["gamma_0"]
    clipped = fire.gamma_bounds_from_sigma(sigma, gammas, W, bounds, sigma_min=0.0, sigma_max=9.9)
    assert clipped["sigma_min"] == sig_minus and clipped["sigma_max"] == sig_plus
    assert clipped["notes"]


def scalar_pv(c, h, r):
    h = max(h, 0.0)
    return c * h if abs(r) < 1e-9 else c * (1.0 - (1.0 + r) ** (-h)) / r


def scalar_gamma(W, H, K, g0, gmin):
    """Reference written with plain ifs, independent of the vectorised code."""
    if W >= K or W + H <= 0:
        return gmin
    return gmin + (g0 - gmin) * min(max(H / (W + H), 0.0), 1.0)


def test_vectorised_gamma_and_human_capital_equal_scalar_versions():
    rng = np.random.default_rng(1)
    W = np.concatenate([rng.uniform(-5e5, 2e6, 500), [0.0, 1e6, 1e6 - 1e-6]])
    for c in (0.0, 4e4, -4e4):
        for r in (-0.0075, 0.0, 0.02):
            vec, H = fire.gamma_t(W, c, 12, r, 1e6, 2.0, 0.3), scalar_pv(c, 12, r)
            np.testing.assert_allclose(vec, [scalar_gamma(w, H, 1e6, 2.0, 0.3) for w in W], rtol=1e-14)
            np.testing.assert_array_equal(vec, [fire.gamma_t(w, c, 12, r, 1e6, 2.0, 0.3) for w in W])
    h = np.array([0, 1, 5, 20, -2])
    for r in (0.0, 1e-12, 0.02, -0.01):
        np.testing.assert_allclose(fire.human_capital(100.0, h, r), [scalar_pv(100.0, k, r) for k in h])


def test_interp_weights_vectorised(grid):
    g = np.linspace(-0.1, grid.gammas[-1] * 1.2, 101)
    vec = opt.interp_weights(g, grid.gammas, grid.weights)
    ref = np.column_stack([np.interp(g, grid.gammas, grid.weights[:, i]) for i in range(6)])
    np.testing.assert_allclose(vec, ref, atol=1e-13)
    np.testing.assert_allclose(opt.interp_weights(g[50], grid.gammas, grid.weights), ref[50], atol=1e-13)


def test_same_seed_identical_different_seed_within_mc_error(grid):
    a, b = (fire.simulate_wealth(PROFILE, grid, RF, N, 7) for _ in range(2))
    np.testing.assert_array_equal(a.wealth, b.wealth)
    assert a.path.equals(b.path)
    other = fire.simulate_wealth(PROFILE, grid, RF, N, 8)
    assert not np.array_equal(a.wealth, other.wealth)
    sa, so = (fire.fire_summary(x.path, PROFILE.target_age) for x in (a, other))
    assert abs(sa["p_target"] - so["p_target"]) < 4 * np.hypot(sa["se_target"], so["se_target"])


def test_probability_and_standard_error_are_path_fractions(grid):
    sim = fire.simulate_wealth(PROFILE, grid, RF, N, 7)
    s = fire.fire_summary(sim.path, PROFILE.target_age)
    w = sim.wealth_at(PROFILE.target_age)
    p = np.mean(w >= 1_500_000)
    assert 0 < p < 1 and s["p_target"] == p
    assert s["se_target"] == pytest.approx(np.sqrt(p * (1 - p) / N))
    row = sim.path.set_index("age").loc[PROFILE.target_age]
    np.testing.assert_allclose(row[["q5", "q25", "q50", "q75", "q95"]].astype(float),
                               np.percentile(w, [5, 25, 50, 75, 95]))
    assert row["W"] == pytest.approx(w.mean())
    assert ((sim.path["P"] >= 0) & (sim.path["P"] <= 1)).all()


def test_zero_volatility_is_the_hand_computed_recursion(setup):
    mu, _, _, gammas, W, gb = setup
    grid0 = fire.ResampledGrid(tuple(TICKERS), gammas, W, mu, np.zeros((6, 6)), gb["gamma_0"], gb["gamma_min"])
    curve = ((1.0, 0.005), (5.0, 0.02), (20.0, 0.03))          # r_f depends on the remaining horizon
    sim = fire.simulate_wealth(PROFILE, grid0, RiskFree(curve, 0.01), 3, 0)
    w, expected, c, K = 100_000.0, [], 60_000.0, 1_500_000.0
    for age in range(30, fire.horizon_end_age(PROFILE) + 1):
        expected.append(w)
        h = max(50 - age, 0)
        r = (1 + nominal_rate(h, curve)) / 1.01 - 1 if h > 0 else 0.0
        g = scalar_gamma(w, scalar_pv(c, h, r), K, gb["gamma_0"], gb["gamma_min"])
        x = np.array([np.interp(g, gammas, W[:, i]) for i in range(6)])
        w = w * (1 + x @ mu) + c                       # returns first, then savings
    np.testing.assert_allclose(sim.wealth, np.tile(expected, (3, 1)), rtol=1e-10)
    assert set(sim.path["P"]) <= {0.0, 1.0} and (sim.path["SE"] == 0).all()


def test_constant_weights_match_closed_form_moments(setup):
    mu, sigma = setup[0], setup[1]
    g, n = constant_grid(mu, sigma), 200_000
    p = PROFILE.with_changes(age=45)
    wT = fire.simulate_wealth(p, g, RF, n, 3).wealth_at(50)
    x, T, c = g.weights[0], 5, p.savings
    m = x @ mu
    s2 = x @ sigma @ x
    mean = p.wealth * (1 + m) ** T + c * ((1 + m) ** T - 1) / m
    var, M = 0.0, p.wealth                                     # Var[W'] = Var (1+m)^2 + s^2 E[W^2]
    for _ in range(T):
        var, M = var * (1 + m) ** 2 + s2 * (var + M ** 2), M * (1 + m) + c
    assert M == pytest.approx(mean)
    assert abs(wT.mean() - mean) < 4 * np.sqrt(var / n)
    assert wT.var() == pytest.approx(var, rel=0.02)


def test_lognormal_returns_match_mean_and_covariance(setup):
    mu, sigma = setup[0], setup[1]
    m, A = fire.lognormal_parameters(mu, sigma)
    V = A @ A.T
    gross = np.exp(m + np.diag(V) / 2)
    np.testing.assert_allclose(gross, 1 + mu)
    np.testing.assert_allclose(np.outer(gross, gross) * np.expm1(V), sigma, atol=1e-12)


def p_fire(profile, grid):
    return fire.fire_summary(fire.simulate_wealth(profile, grid, RF, N, 7).path, profile.target_age)["p_target"]


def test_probability_non_decreasing_in_savings_and_horizon(grid):
    by_c = [p_fire(PROFILE.with_changes(income=60_000 + c), grid) for c in range(20_000, 100_001, 10_000)]
    by_h = [p_fire(PROFILE.with_changes(target_age=a), grid) for a in range(40, 61, 2)]
    for probs in (by_c, by_h):
        assert all(0.0 <= q <= 1.0 for q in probs)
        assert np.all(np.diff(probs) >= 0) and probs[-1] > probs[0] + 0.5


def test_common_random_numbers_across_horizons(setup):
    g = constant_grid(setup[0], setup[1])
    a = fire.simulate_wealth(PROFILE, g, RF, 50, 7).wealth
    b = fire.simulate_wealth(PROFILE.with_changes(target_age=55), g, RF, 50, 7).wealth
    np.testing.assert_allclose(a, b[:, :a.shape[1]], rtol=1e-12)


def test_simulated_paths_rebuilt_by_hand_path_by_path(grid):
    """Scalar loop over a few paths: gamma_t uses that path's own W_t, returns first, then savings."""
    n, seed, c, K = 5, 21, 60_000.0, 1_500_000.0
    sim = fire.simulate_wealth(PROFILE, grid, RF, n, seed)
    m, A = fire.lognormal_parameters(grid.mu, grid.sigma)
    rng, w = np.random.default_rng(seed), np.full(n, 100_000.0)
    for t in range(sim.wealth.shape[1] - 1):
        np.testing.assert_allclose(sim.wealth[:, t], w, rtol=1e-9)
        z, h = rng.standard_normal((n, 6)), max(20 - t, 0)
        H = scalar_pv(c, h, 1.01 / 1.015 - 1)
        new = []
        for k in range(n):
            g = scalar_gamma(w[k], H, K, grid.gamma_0, grid.gamma_min)
            x = np.array([np.interp(g, grid.gammas, grid.weights[:, i]) for i in range(6)])
            new.append(w[k] * (x @ np.exp(m + A @ z[k])) + c)
        w = np.array(new)


def test_gamma_follows_each_path(grid):
    """Different wealth on the same date -> different portfolio: richer paths hold less risk."""
    W = np.array([1e5, 8e5, 1.6e6])
    _, H, g, x = fire.target_portfolio(W, 10, PROFILE, grid, RF)
    assert g[0] > g[1] > g[2] == grid.gamma_min
    np.testing.assert_allclose(g, [scalar_gamma(w, H, 1.5e6, grid.gamma_0, grid.gamma_min) for w in W])
    assert_feasible(x)


def test_glide_path_feasible_and_consistent(setup, grid):
    gb = setup[5]
    path = fire.simulate_wealth(PROFILE, grid, RF, N, 7).path
    assert path["age"].iloc[0] == PROFILE.age and path["age"].iloc[-1] == fire.horizon_end_age(PROFILE)
    assert_feasible(fire.weights_table(path, TICKERS).values)            # average portfolios
    assert ((path["gamma"] >= gb["gamma_min"] - 1e-12) & (path["gamma"] <= gb["gamma_0"] + 1e-12)).all()
    np.testing.assert_allclose(path.loc[path["age"] >= PROFILE.target_age, "gamma"], gb["gamma_min"])
    trades = fire.weights_table(path, TICKERS, "trade_amt_").iloc[1:]
    np.testing.assert_allclose(trades.sum(axis=1), PROFILE.savings, rtol=1e-9)  # trades invest c


def test_default_risk_range_gives_a_moving_glide_path(setup):
    """Regression: with the default sigma range the weights must change between today and FIRE."""
    gb = setup[5]
    assert not fire.is_narrow_risk_range(gb["sigma_min"], gb["sigma_max"])
    assert gb["gamma_0"] > gb["gamma_min"] and not gb["notes"]
    path = fire.simulate_wealth(PROFILE, make_grid(setup), RF, N, 7).path
    w = fire.weights_table(path, TICKERS)
    assert path["gamma"].iloc[0] > path["gamma"].iloc[-1]
    assert path["s"].iloc[0] > path["s"].iloc[-1] + 0.005               # risk falls by > 0.5pp
    assert np.abs(w.iloc[0] - w.iloc[-1]).max() > 0.05                  # some weight moves by > 5pp


def test_narrow_risk_range_flags_flat_glide_path(setup):
    mu, sigma, bounds, gammas, W, _ = setup
    sig_minus = bounds[0]
    gb = fire.gamma_bounds_from_sigma(sigma, gammas, W, bounds, sig_minus, sig_minus)
    assert fire.NARROW_RANGE_NOTE in gb["notes"]
    path = fire.simulate_wealth(PROFILE, make_grid((*setup[:5], gb)), RF, 200, 7).path
    w = fire.weights_table(path, TICKERS)
    assert np.abs(w.iloc[0] - w.iloc[-1]).max() < 1e-9                  # flat: the reported symptom
    assert fire.is_narrow_risk_range(0.07, 0.074) and not fire.is_narrow_risk_range(0.07, 0.075)


def test_held_portfolio_volatility_equals_risk_targets(setup):
    """Fix 4: the resampled weights actually held at gamma_0 / gamma_min have exactly the target
    volatilities, and a client with no wealth yet (H/(W+H) = 1, so gamma_t = gamma_0) starts with
    volatility sigma_max."""
    mu, sigma, _, gammas, W, gb = setup
    for g, target in ((gb["gamma_0"], gb["sigma_max"]), (gb["gamma_min"], gb["sigma_min"])):
        assert opt.resampled_vol(g, gammas, W, sigma) == pytest.approx(target, abs=1e-8)
    start = PROFILE.with_changes(wealth=0.0)
    path = fire.simulate_wealth(start, make_grid(setup), RF, N, 7).path
    assert fire.start_volatility(start, make_grid(setup), RF) == pytest.approx(path["s"].iloc[0])
    assert path["gamma"].iloc[0] == pytest.approx(gb["gamma_0"])
    assert path["s"].iloc[0] == pytest.approx(gb["sigma_max"], abs=1e-8)
    assert path["s"].iloc[-1] == pytest.approx(gb["sigma_min"], abs=1e-8)
    assert_feasible(fire.weights_table(path, TICKERS).values)            # 40% cap, 10% gold kept


def test_validate_profile():
    errors, _ = fire.validate_profile(PROFILE.with_changes(target_age=30))
    assert errors
    _, warnings = fire.validate_profile(PROFILE.with_changes(income=50_000))
    assert warnings
    assert fire.validate_profile(PROFILE) == ([], [])
