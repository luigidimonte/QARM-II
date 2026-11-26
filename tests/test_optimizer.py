"""Sanity tests for optimizer.py (synthetic moments, no network)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize

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


def x_star(mu, sigma, gamma, tickers, config=None):
    """Constrained gamma-problem solved once: the frontier at a single gamma."""
    return opt.frontier_weights(mu, sigma, [gamma], tickers, config or opt.CONFIG)[0]


def assert_feasible(x: np.ndarray, config: dict = None, tickers=TICKERS) -> None:
    """Fully invested, long-only, and inside the per-asset bounds of config (default rules)."""
    lower, upper = opt.weight_bounds(list(tickers), config or opt.CONFIG)
    x = np.atleast_2d(x)
    np.testing.assert_allclose(x.sum(axis=-1), 1.0, atol=TOL)
    assert (x >= -TOL).all()
    assert (x >= lower - TOL).all() and (x <= upper + TOL).all()


def test_gamma_zero_is_constrained_min_variance(moments):
    mu, sigma = moments
    x0 = x_star(mu, sigma, 0.0, TICKERS)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    lower, upper = opt.weight_bounds(TICKERS)
    ref = minimize(lambda x: 0.5 * x @ sigma @ x, np.full(6, 1 / 6), method="SLSQP",
                   bounds=list(zip(lower, upper)), constraints=cons, options={"ftol": 1e-14, "maxiter": 500})
    assert ref.success
    np.testing.assert_allclose(x0, ref.x, atol=1e-4)
    assert x0 @ sigma @ x0 <= ref.x @ sigma @ ref.x + 1e-8
    assert_feasible(x0)


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
    x = x_star(mu, sigma, gamma, TICKERS)
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




@pytest.fixture(scope="module")
def resampled(moments, bounds):
    mu, sigma = moments
    gammas = opt.gamma_grid(mu, sigma, TICKERS, n_grid=10, bounds=bounds)
    W = opt.resample_weights(mu / 12, sigma / 12, 120, gammas, TICKERS, n_resamples=10, seed=3)
    return gammas, W, opt.resampled_frontier_bounds(gammas, W, sigma)


@pytest.mark.parametrize("frac", [0.05, 0.3, 0.5, 0.8, 0.97])
def test_resampled_sigma_problem_hits_target(moments, resampled, frac):
    """Realised volatility of the averaged weights (original Sigma) equals the target."""
    mu, sigma = moments
    gammas, W, (lo, hi, _) = resampled
    target = lo + frac * (hi - lo)
    gamma, status = opt.solve_sigma_resampled(target, gammas, W, sigma)
    assert status == "ok" and gammas[0] <= gamma <= gammas[-1]
    x = opt.interp_weights(gamma, gammas, W)
    assert opt.portfolio_stats(x, mu, sigma)[1] == pytest.approx(target, abs=1e-8)
    assert_feasible(opt.interp_weights(gamma, gammas, W)[None, :])


def test_resampled_sigma_problem_out_of_range(moments, resampled):
    _, sigma = moments
    gammas, W, (lo, hi, g_hi) = resampled
    assert opt.solve_sigma_resampled(lo / 2, gammas, W, sigma) == (gammas[0], "below_min")
    assert opt.solve_sigma_resampled(hi * 2, gammas, W, sigma) == (g_hi, "capped")


# ----------------------------------------------------------------------------- solver failures
def broken_minimize(monkeypatch, n_bad: int, status: int = 8):
    """minimize that returns an infeasible point (status `status`) on its first n_bad calls."""
    real, calls = opt.minimize, []

    def fake(*args, **kwargs):
        calls.append(args[1])                                  # starting point
        if len(calls) <= n_bad:
            return OptimizeResult(x=np.full(len(args[1]), 0.5), status=status, success=False,
                                  message="Positive directional derivative for linesearch")
        return real(*args, **kwargs)
    monkeypatch.setattr(opt, "minimize", fake)
    return calls


@pytest.mark.parametrize("warm", [False, True])
def test_infeasible_solution_is_retried_from_another_start(moments, monkeypatch, warm):
    mu, sigma = moments
    cons = opt.portfolio_constraints(TICKERS)
    equal = np.ones(6) / 6
    x0 = x_star(mu, sigma, 0.1, TICKERS) if warm else equal
    ref = opt.efficient_frontier(x0, mu, sigma, 0.2, cons)[2]
    calls = broken_minimize(monkeypatch, n_bad=1)
    x = opt.efficient_frontier(x0, mu, sigma, 0.2, cons)[2]
    assert len(calls) == 2 and not np.allclose(calls[0], calls[1])          # a different start
    if warm:
        np.testing.assert_allclose(calls[1], equal)                          # equal weights
    else:
        assert opt.constraints_hold(calls[1], cons)                          # inside the bounds
    assert_feasible(x)
    np.testing.assert_allclose(x, ref, atol=1e-6)


def test_persistent_failure_raises_a_clear_error(moments, monkeypatch):
    mu, sigma = moments
    broken_minimize(monkeypatch, n_bad=2)
    with pytest.raises(RuntimeError, match="respects the investment rules"):
        x_star(mu, sigma, 0.2, TICKERS)


def test_non_success_status_with_feasible_solution_is_accepted(moments, monkeypatch):
    """Status 8 alone is not a failure: the constraints decide."""
    mu, sigma = moments
    ref = x_star(mu, sigma, 0.2, TICKERS)
    real = opt.minimize

    def status8(*args, **kwargs):
        res = real(*args, **kwargs)
        res.status, res.success = 8, False
        return res
    monkeypatch.setattr(opt, "minimize", status8)
    np.testing.assert_allclose(x_star(mu, sigma, 0.2, TICKERS), ref)


def test_constraints_hold_tolerance():
    cons = opt.portfolio_constraints(TICKERS)
    x = np.array([0.3, 0.2, 0.1, 0.1, 0.2, 0.1])
    assert opt.constraints_hold(x, cons)
    assert opt.constraints_hold(x + np.array([5e-9, 0, 0, 0, 0, 0]), cons)          # within 1e-8
    assert not opt.constraints_hold(x + np.array([1e-6, 0, 0, 0, 0, 0]), cons)      # sum != 1
    assert not opt.constraints_hold(np.array([0.45, 0.15, 0.1, 0.1, 0.1, 0.1]), cons)  # cap


# ----------------------------------------------------------------------------- client constraints
def test_default_bounds_equal_explicit_bounds(moments, bounds):
    """1d: the default rules give exactly the results of explicit bounds 0 <= x_i <= 0.30,
    x_gold <= 0.15 and no gold minimum (frontier bounds, gamma grid, frontier and resampled
    weights, same seed)."""
    mu, sigma = moments
    explicit = {**opt.CONFIG, "lower": {t: 0.0 for t in TICKERS},
                "upper": {t: (0.15 if t == "GLD" else 0.30) for t in TICKERS}}
    lower, upper = opt.weight_bounds(TICKERS)
    np.testing.assert_array_equal(lower, opt.weight_bounds(TICKERS, explicit)[0])
    np.testing.assert_array_equal(upper, opt.weight_bounds(TICKERS, explicit)[1])
    b_exp = opt.frontier_bounds(mu, sigma, TICKERS, config=explicit)
    np.testing.assert_array_equal(bounds, b_exp)
    g_def = opt.gamma_grid(mu, sigma, TICKERS, n_grid=8, bounds=bounds)
    g_exp = opt.gamma_grid(mu, sigma, TICKERS, n_grid=8, bounds=b_exp, config=explicit)
    np.testing.assert_array_equal(g_def, g_exp)
    np.testing.assert_array_equal(opt.frontier_weights(mu, sigma, g_def, TICKERS),
                                  opt.frontier_weights(mu, sigma, g_exp, TICKERS, explicit))
    W_def = opt.resample_weights(mu / 12, sigma / 12, 120, g_def, TICKERS, 8, 5)
    W_exp = opt.resample_weights(mu / 12, sigma / 12, 120, g_exp, TICKERS, 8, 5, explicit)
    np.testing.assert_array_equal(W_def, W_exp)
    assert "min_gold" not in opt.CONFIG and lower[GOLD] == 0.0                # no gold minimum


def test_no_function_reads_module_bounds(moments, monkeypatch):
    """The bounds come only from the config argument."""
    mu, sigma = moments
    cfg = {**opt.CONFIG, "upper": {"SPY": 0.05}}
    x = x_star(mu, sigma, 1.0, TICKERS, cfg)
    for key in ("max_weight", "max_gold"):
        monkeypatch.setitem(opt.CONFIG, key, 0.9)
    np.testing.assert_allclose(x_star(mu, sigma, 1.0, TICKERS, cfg), x)


def test_default_rules():
    lower, upper = opt.weight_bounds(TICKERS)
    np.testing.assert_array_equal(lower, 0.0)
    np.testing.assert_allclose(upper, [0.30, 0.30, 0.30, 0.30, 0.30, 0.15])
    assert opt.feasibility_problems(TICKERS) == []
    import data
    universe = list(data.UNIVERSE)
    assert opt.weight_bounds(universe)[1].sum() >= 1.0 and opt.feasibility_problems(universe) == []


def test_default_rules_respected(moments, bounds):
    mu, sigma = moments
    gammas = opt.gamma_grid(mu, sigma, TICKERS, n_grid=8, bounds=bounds)
    W = opt.resample_draws(mu / 12, sigma / 12, 120, gammas, TICKERS, 6, 2).reshape(-1, 6)
    assert (W <= 0.30 + TOL).all() and (W[:, GOLD] <= 0.15 + TOL).all()
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=TOL)
    assert W[:, GOLD].max() > 0.15 - 1e-4                      # the gold cap binds somewhere


def test_gold_cap_only_if_gold_is_in_the_universe(moments):
    mu, sigma = moments
    no_gold = [t for t in TICKERS if t != "GLD"]
    keep = [TICKERS.index(t) for t in no_gold]
    assert opt.feasibility_problems(no_gold) == []
    np.testing.assert_allclose(opt.weight_bounds(no_gold)[1], 0.30)
    x = x_star(mu[keep], sigma[np.ix_(keep, keep)], 0.5, no_gold)
    assert_feasible(x, tickers=no_gold)


CUSTOM = {**opt.CONFIG, "lower": {"IEF": 0.20, "GLD": 0.05}, "upper": {"EEM": 0.0, "SPY": 0.50, "GLD": 0.25}}


def test_custom_bounds_respected(moments):
    mu, sigma = moments
    b = opt.frontier_bounds(mu, sigma, TICKERS, config=CUSTOM)
    gammas = opt.gamma_grid(mu, sigma, TICKERS, n_grid=8, bounds=b, config=CUSTOM)
    draws = opt.resample_draws(mu / 12, sigma / 12, 120, gammas, TICKERS, 6, 2, CUSTOM)
    for W in (opt.frontier_weights(mu, sigma, gammas, TICKERS, CUSTOM), draws.reshape(-1, 6), draws.mean(axis=0)):
        assert_feasible(W, CUSTOM)
        assert np.all(W[:, TICKERS.index("EEM")] == 0.0)                     # excluded asset
        assert (W[:, TICKERS.index("IEF")] >= 0.20 - TOL).all()              # forced minimum
        assert (W[:, GOLD] >= 0.05 - TOL).all() and (W[:, GOLD] <= 0.25 + TOL).all()
    assert (draws[..., TICKERS.index("SPY")] > 0.30 + 1e-3).any()            # a loosened cap is used
    rs = opt.resampled_frontier_bounds(gammas, draws.mean(axis=0), sigma)
    for g in np.linspace(0, gammas[-1], 7):
        assert_feasible(opt.interp_weights(g, gammas, draws.mean(axis=0)), CUSTOM)
    assert rs[0] < rs[1]


def test_bounds_change_the_frontier(moments):
    mu, sigma = moments
    assert opt.frontier_bounds(mu, sigma, TICKERS) != opt.frontier_bounds(mu, sigma, TICKERS, config=CUSTOM)


NAMES = {t: f"Asset {t}" for t in TICKERS}


@pytest.mark.parametrize("lower, upper, msg", [
    ({"SPY": 0.4}, {"SPY": 0.2}, "Asset SPY: the minimum weight (40%) is higher than the maximum (20%)"),
    ({"SPY": 0.6, "IEF": 0.5}, {"SPY": 0.6, "IEF": 0.5}, "minimum weights add up to 110%"),
    ({}, {t: 0.1 for t in TICKERS}, "maximum weights add up to only 60%"),
    ({"LQD": -0.1}, {}, "Asset LQD: the minimum and maximum weight must be between 0% and 100%"),
    ({}, {"GLD": 1.2}, "Asset GLD: the minimum and maximum weight must be between 0% and 100%"),
])
def test_infeasible_rules_are_named(lower, upper, msg):
    cfg = {**opt.CONFIG, "lower": lower, "upper": upper}
    problems = opt.feasibility_problems(TICKERS, cfg, NAMES)
    assert any(msg in p for p in problems), problems
    with pytest.raises(ValueError, match=msg.split(":")[0]):
        opt.check_feasibility(TICKERS, cfg, NAMES)
    with pytest.raises(ValueError):
        opt.portfolio_constraints(TICKERS, cfg)                 # never reaches minimize


def test_too_few_assets_for_the_default_cap():
    assert any("add up to only 60%" in p for p in opt.feasibility_problems(["SPY", "EEM"]))


# ----------------------------------------------------------------------------- flat stretches of sigma(gamma)
# Real CHF moments (rounded) of the 8-asset universe, Dec 2004 - Oct 2026: with these custom bounds
# the optimum stays at a corner of the constraints for a range of gamma, so sigma(x*(gamma)) is
# flat there and fsolve stalls (regression: the app failed with "No portfolio with volatility").
U8 = ["SPY", "SMICHA.SW", "EEM", "VNQ", "IEF", "LQD", "TIP", "GLD"]
MU8 = np.array([0.09261, 0.07783, 0.06742, 0.0696, 0.01134, 0.01918, 0.0123, 0.08754])
SIG8 = np.array([
    [0.022127, 0.013357, 0.022814, 0.023523, -0.000399, 0.005367, 0.002801, 0.002387],
    [0.013357, 0.016133, 0.012744, 0.015964, -0.00018, 0.00327, 0.00127, -0.001515],
    [0.022814, 0.012744, 0.043505, 0.026779, -0.000207, 0.007244, 0.004188, 0.012066],
    [0.023523, 0.015964, 0.026779, 0.046297, 0.002162, 0.009107, 0.005273, 0.004947],
    [-0.000399, -0.00018, -0.000207, 0.002162, 0.004642, 0.003787, 0.003003, 0.003781],
    [0.005367, 0.00327, 0.007244, 0.009107, 0.003787, 0.006689, 0.003602, 0.004058],
    [0.002801, 0.00127, 0.004188, 0.005273, 0.003003, 0.003602, 0.003541, 0.004452],
    [0.002387, -0.001515, 0.012066, 0.004947, 0.003781, 0.004058, 0.004452, 0.030165]])
CORNER = {**opt.CONFIG, "lower": {"IEF": 0.20}, "upper": {"EEM": 0.0, "GLD": 0.0, "SPY": 0.50}}


def test_sigma_problem_on_a_flat_stretch(monkeypatch):
    b = opt.frontier_bounds(MU8, SIG8, U8, config=CORNER)
    gammas = opt.gamma_grid(MU8, SIG8, U8, bounds=b, config=CORNER)
    W = opt.frontier_weights(MU8, SIG8, gammas, U8, CORNER)
    vols = [opt.portfolio_stats(x, MU8, SIG8)[1] for x in W]
    np.testing.assert_allclose(vols, np.linspace(b[0], b[1], len(gammas)), atol=1e-7)
    assert_feasible(W, CORNER, U8)
    monkeypatch.setattr(opt, "bisect_root", lambda *a, **k: None)          # fsolve alone fails here
    with pytest.raises(RuntimeError, match="No portfolio with volatility"):
        opt.gamma_grid(MU8, SIG8, U8, bounds=b, config=CORNER)


def test_resampled_sigma_problem_falls_back_to_bisection(moments, resampled, monkeypatch):
    mu, sigma = moments
    gammas, W, (lo, hi, _) = resampled
    target = lo + 0.4 * (hi - lo)
    monkeypatch.setattr(opt, "fsolve", lambda f, x0, args, full_output: (np.array([1e9]),))
    gamma, status = opt.solve_sigma_resampled(target, gammas, W, sigma)
    assert status == "ok"
    assert opt.portfolio_stats(opt.interp_weights(gamma, gammas, W), mu, sigma)[1] == pytest.approx(target, abs=1e-8)


def test_frontier_end_with_low_return_assets():
    """Regression: with bonds only sigma hardly moves at small gamma; the frontier must still run
    to the maximum-return portfolio (100% LQD here), not stop at the minimum-variance one."""
    cfg = {**opt.CONFIG, "upper": dict(zip(U8, (0, 0, 0, 0, 1.0, 1.0, 1.0, 0)))}
    lower, upper = opt.weight_bounds(U8, cfg)
    x_end = opt.max_return_portfolio(MU8, lower, upper)
    np.testing.assert_allclose(x_end, np.eye(8)[U8.index("LQD")])
    sig_minus, sig_plus, _ = opt.frontier_bounds(MU8, SIG8, U8, config=cfg)
    assert sig_plus == pytest.approx(np.sqrt(SIG8[5, 5]), abs=1e-6) and sig_plus - sig_minus > 0.02


def test_max_return_portfolio_respects_bounds():
    lower = np.array([0.1, 0.0, 0.2, 0.0, 0.0, 0.0])
    upper = np.array([0.3, 0.3, 0.3, 0.0, 0.5, 0.15])
    x = opt.max_return_portfolio(np.array([0.10, 0.08, 0.09, 0.30, 0.04, 0.07]), lower, upper)
    np.testing.assert_allclose(x, [0.3, 0.3, 0.3, 0.0, 0.0, 0.1])          # excluded asset stays at 0
