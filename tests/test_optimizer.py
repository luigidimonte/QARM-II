"""Sanity tests for optimizer.py (synthetic moments, no network)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optimizer as opt  # noqa: E402

TICKERS = ["SPY", "EWL", "EEM", "IEF", "LQD", "GLD"]
GOLD = TICKERS.index("GLD")
TOL = 1e-6


@pytest.fixture(scope="module")
def moments():
    mu = np.array([0.10, 0.08, 0.09, 0.03, 0.04, 0.07])
    vol = np.array([0.15, 0.14, 0.22, 0.07, 0.08, 0.17])
    corr = np.array([
        [1.00, 0.80, 0.75, -0.20, 0.30, 0.05],
        [0.80, 1.00, 0.70, -0.10, 0.30, 0.10],
        [0.75, 0.70, 1.00, -0.10, 0.35, 0.20],
        [-0.20, -0.10, -0.10, 1.00, 0.70, 0.30],
        [0.30, 0.30, 0.35, 0.70, 1.00, 0.25],
        [0.05, 0.10, 0.20, 0.30, 0.25, 1.00],
    ])
    sigma = corr * np.outer(vol, vol)
    return mu, sigma


@pytest.fixture(scope="module")
def bounds(moments):
    mu, sigma = moments
    return opt.frontier_bounds(mu, sigma, TICKERS)


def assert_feasible(x: np.ndarray) -> None:
    x = np.atleast_2d(x)
    np.testing.assert_allclose(x.sum(axis=-1), 1.0, atol=TOL)
    assert (x >= -TOL).all()
    assert (x <= opt.CONFIG["max_weight"] + TOL).all()
    assert (x[..., GOLD] >= opt.CONFIG["min_gold"] - TOL).all()


def test_gamma_zero_is_constrained_min_variance(moments):
    mu, sigma = moments
    x0 = opt.solve_gamma(mu, sigma, 0.0, TICKERS)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
            {"type": "ineq", "fun": lambda x: x[GOLD] - 0.10}]
    ref = minimize(lambda x: 0.5 * x @ sigma @ x, np.full(6, 1 / 6), method="SLSQP",
                   bounds=[(0, 0.40)] * 6, constraints=cons, options={"ftol": 1e-14, "maxiter": 500})
    assert ref.success
    np.testing.assert_allclose(x0, ref.x, atol=1e-4)
    assert x0 @ sigma @ x0 <= ref.x @ sigma @ ref.x + 1e-8
    assert_feasible(x0)


def test_negative_gamma_rejected(moments):
    mu, sigma = moments
    with pytest.raises(ValueError):
        opt.solve_gamma(mu, sigma, -0.1, TICKERS)


def test_frontier_volatility_increases_with_gamma(moments, bounds):
    mu, sigma = moments
    sig_minus, sig_plus, gamma_hi = bounds
    assert sig_minus < sig_plus
    gammas = np.linspace(0, gamma_hi, 15)
    vols = [opt.portfolio_stats(x, mu, sigma)[1] for x in opt.frontier_weights(mu, sigma, gammas, TICKERS)]
    assert np.all(np.diff(vols) >= -1e-7)


@pytest.mark.parametrize("frac", [0.1, 0.5, 0.9])
def test_sigma_problem_hits_target(moments, bounds, frac):
    mu, sigma = moments
    sig_minus, sig_plus, _ = bounds
    target = sig_minus + frac * (sig_plus - sig_minus)
    gamma, status = opt.solve_sigma(mu, sigma, target, TICKERS, bounds)
    assert status == "ok"
    x = opt.solve_gamma(mu, sigma, gamma, TICKERS)
    assert opt.portfolio_stats(x, mu, sigma)[1] == pytest.approx(target, abs=1e-4)


def test_sigma_problem_out_of_range(moments, bounds):
    mu, sigma = moments
    sig_minus, sig_plus, gamma_hi = bounds
    assert opt.solve_sigma(mu, sigma, sig_minus / 2, TICKERS, bounds) == (0.0, "below_min")
    assert opt.solve_sigma(mu, sigma, sig_plus * 2, TICKERS, bounds) == (gamma_hi, "capped")


def test_all_portfolios_feasible(moments, bounds):
    mu, sigma = moments
    gammas = opt.gamma_grid(mu, sigma, TICKERS, n_grid=8, bounds=bounds)
    assert gammas[0] == 0.0 and np.all(np.diff(gammas) > 0)
    assert_feasible(opt.frontier_weights(mu, sigma, gammas, TICKERS))

    draws = opt.resample_draws(mu / 12, sigma / 12, 120, gammas, TICKERS, n_resamples=15, seed=1)
    assert_feasible(draws.reshape(-1, len(TICKERS)))          # every single resampled portfolio
    avg = draws.mean(axis=0)
    assert_feasible(avg)                                       # averaged (resampled) portfolios
    np.testing.assert_allclose(avg, opt.resample_weights(mu / 12, sigma / 12, 120, gammas, TICKERS,
                                                         n_resamples=15, seed=1))
    for g in np.linspace(-1.0, gammas[-1] * 1.5, 25):          # interpolated (incl. clipping)
        assert_feasible(opt.interp_weights(g, gammas, avg))


def test_resampling_follows_the_four_steps(moments, bounds):
    """LN 3 slide 12 re-done by hand, draw by draw, must give the same weights."""
    mu, sigma = moments
    gammas, T, K = opt.gamma_grid(mu, sigma, TICKERS, n_grid=5, bounds=bounds), 120, 6
    rng, total = np.random.default_rng(11), 0.0
    for _ in range(K):                                                        # step 3
        sample = rng.multivariate_normal(mu / 12, sigma / 12, size=T)         # step 1
        mu_k, sigma_k = sample.mean(axis=0) * 12, np.cov(sample, rowvar=False) * 12
        total = total + opt.frontier_weights(mu_k, sigma_k, gammas, TICKERS)  # step 2, same grid
    W = opt.resample_weights(mu / 12, sigma / 12, T, gammas, TICKERS, n_resamples=K, seed=11)
    np.testing.assert_allclose(W, total / K, atol=1e-6)                       # step 4
    assert opt.CONFIG["n_resamples"] == 1000


def test_resampling_seed(moments, bounds):
    mu, sigma = moments
    gammas = opt.gamma_grid(mu, sigma, TICKERS, n_grid=5, bounds=bounds)
    run = lambda seed: opt.resample_weights(mu / 12, sigma / 12, 120, gammas, TICKERS, n_resamples=5, seed=seed)
    a = run(1)
    np.testing.assert_array_equal(a, run(1))
    assert not np.allclose(a, run(2))
    assert_feasible(a)


@pytest.mark.parametrize("tickers, msg", [
    (["SPY", "GLD"], "assets"),
    (["SPY", "EWL", "EEM", "IEF"], "gold"),
])
def test_feasibility_check(tickers, msg):
    with pytest.raises(ValueError, match=msg):
        opt.check_feasibility(tickers)


@pytest.fixture(scope="module")
def resampled(moments, bounds):
    mu, sigma = moments
    gammas = opt.gamma_grid(mu, sigma, TICKERS, n_grid=10, bounds=bounds)
    W = opt.resample_weights(mu / 12, sigma / 12, 120, gammas, TICKERS, n_resamples=10, seed=3)
    return gammas, W, opt.resampled_frontier_bounds(gammas, W, sigma)


@pytest.mark.parametrize("frac", [0.05, 0.3, 0.5, 0.8, 0.97])
def test_resampled_sigma_problem_hits_target(moments, resampled, frac):
    """Realised volatility of the averaged weights (original Sigma) equals the target."""
    _, sigma = moments
    gammas, W, (lo, hi, _) = resampled
    target = lo + frac * (hi - lo)
    gamma, status = opt.solve_sigma_resampled(target, gammas, W, sigma)
    assert status == "ok" and gammas[0] <= gamma <= gammas[-1]
    assert opt.resampled_vol(gamma, gammas, W, sigma) == pytest.approx(target, abs=1e-8)
    assert_feasible(opt.interp_weights(gamma, gammas, W)[None, :])


def test_resampled_sigma_problem_out_of_range(moments, resampled):
    _, sigma = moments
    gammas, W, (lo, hi, g_hi) = resampled
    assert opt.solve_sigma_resampled(lo / 2, gammas, W, sigma) == (gammas[0], "below_min")
    assert opt.solve_sigma_resampled(hi * 2, gammas, W, sigma) == (g_hi, "capped")
