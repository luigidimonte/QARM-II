"""Constrained mean-variance optimisation, written as in the course labs (numpy and scipy only).

Lab Markowitz LN1:  QP(x, sigma, mu, gamma) minimised with scipy.optimize.minimize under
                    LinearConstraints; volatility target solved with fsolve (sigma-problem).
Lab Stability LN3:  no short-selling constraint, tight tolerance (tol, maxiter) and resampling of
                    the efficient frontier (draw, re-estimate, loop over gamma, average weights).

gamma-problem (LN 1):  x*(gamma) = argmin 1/2 x'Sigma x - gamma x'mu
                       u.c. 1'x = 1, lower <= x <= upper (per-asset vectors, lower >= 0)
Default rules: 0 <= x_i <= 0.30, x_gold <= 0.15; a client may set its own lower and upper.
gamma is the risk TOLERANCE (gamma >= 0); gamma = 0 gives the constrained minimum-variance
portfolio.
sigma-problem (LN 1): find gamma such that sigma(x*(gamma)) = sigma*, by root finding.
Resampling (LN 3 slides 11-12, Michaud 2007), steps 1-4: resample_draws and resample_weights.
The risk targets of the plan are hit on the RESAMPLED frontier (the weights actually held):
solve_sigma_resampled finds gamma such that the averaged weights, evaluated with the original
mu and Sigma, have volatility sigma*. solve_sigma (point frontier) only builds the gamma grid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, fsolve, minimize

CONFIG: Dict = {
    # DEFAULT rules only; a client's own rules are passed as config["lower"] / config["upper"]
    # ({ticker: weight}), see weight_bounds. Long-only and full investment are always enforced.
    # Project assumption (not from the course): 30% cap per asset for diversification; gold
    # capped at 15% because its historical return in the sample window (2004-2026) is
    # exceptionally high for an asset without cash flows.
    # No minimum weight by default (x_i >= 0, long-only), no gold minimum.
    "max_weight": 0.30,      # x_i <= 30% for every asset
    "gold_ticker": "GLD",
    "max_gold": 0.15,        # x_gold <= 15% (applies only if gold is in the universe)
    "n_resamples": 1000,     # K draws of the resampling
    "n_grid": 40,
    "seed": 42,
}
_SIGMA_TOL = 1e-7            # volatility targets this close to the frontier ends are the ends
_FSOLVE_TOL = 1e-8           # largest accepted |volatility - target| of an fsolve solution
_FEAS_TOL = 1e-8             # largest accepted violation of a constraint by a minimize solution


def weight_bounds(tickers: List[str], config: Dict = CONFIG) -> Tuple[np.ndarray, np.ndarray]:
    """Per-asset bounds (lower, upper) with lower <= x <= upper.

    Default rules from config: 0 <= x_i <= max_weight for every asset, x_gold <= max_gold if the
    gold ticker is in the universe. config["lower"] and config["upper"] ({ticker: weight})
    replace the default of the tickers they contain (client preferences).
    """
    gold = config["gold_ticker"]
    lower = np.zeros(len(tickers))
    upper = np.array([config["max_gold"] if t == gold else config["max_weight"] for t in tickers], float)
    for i, t in enumerate(tickers):
        lower[i] = config.get("lower", {}).get(t, lower[i])
        upper[i] = config.get("upper", {}).get(t, upper[i])
    return lower, upper


def feasibility_problems(tickers: List[str], config: Dict = CONFIG,
                         names: Optional[Dict[str, str]] = None) -> List[str]:
    """Plain-language reasons why no fully invested portfolio satisfies the bounds (empty if fine).
    names maps a ticker to the asset class shown in the messages."""
    names = names or {}
    lower, upper = weight_bounds(tickers, config)
    problems: List[str] = []
    for t, lo, hi in zip(tickers, lower, upper):
        name = names.get(t, t)
        if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
            problems.append(f"{name}: the minimum and maximum weight must be between 0% and 100%.")
        elif lo > hi + 1e-12:
            problems.append(f"{name}: the minimum weight ({lo:.0%}) is higher than the maximum ({hi:.0%}).")
    if lower.sum() > 1.0 + 1e-12:
        problems.append(f"The minimum weights add up to {lower.sum():.0%}: together they must be at "
                        f"most 100%.")
    if upper.sum() < 1.0 - 1e-12:
        problems.append(f"The maximum weights add up to only {upper.sum():.0%}: together they must be "
                        f"at least 100%, otherwise the money cannot be fully invested.")
    return problems


def check_feasibility(tickers: List[str], config: Dict = CONFIG,
                      names: Optional[Dict[str, str]] = None) -> None:
    """Raise ValueError (plain-language message) if the constraints cannot be met."""
    problems = feasibility_problems(tickers, config, names)
    if problems:
        raise ValueError(" ".join(problems))


def QP(x, sigma, mu, gamma):
    """Quadratic programming function (Lab Markowitz LN1)."""
    v = 0.5 * x.T @ sigma @ x - gamma * x.T @ mu
    return v


def portfolio_constraints(tickers: List[str], config: Dict = CONFIG) -> List[LinearConstraint]:
    """Fully invested and lower <= x <= upper (LinearConstraint with vector bounds, as in the labs)."""
    check_feasibility(tickers, config)
    n = len(tickers)
    lower, upper = weight_bounds(tickers, config)
    return [LinearConstraint(np.ones(n), 1, 1),
            LinearConstraint(np.eye(n), lower, upper)]


def portfolio_stats(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Tuple[float, float]:
    """Expected return mu(x) = x'mu and volatility sigma(x) = sqrt(x'Sigma x)."""
    x, mu, sigma = np.asarray(x), np.asarray(mu), np.asarray(sigma)
    return float(x @ mu), float(np.sqrt(max(x @ sigma @ x, 0.0)))


def constraints_hold(x: np.ndarray, constraints: List[LinearConstraint], tol: float = _FEAS_TOL) -> bool:
    """True if lb - tol <= A x <= ub + tol for every LinearConstraint."""
    for c in constraints:
        Ax = np.atleast_2d(c.A) @ x
        if np.any(Ax < c.lb - tol) or np.any(Ax > c.ub + tol):
            return False
    return True


def box_bounds(constraints: List[LinearConstraint], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-asset bounds (lower, upper) contained in the constraints: rows with a single 1."""
    lower, upper = np.zeros(n), np.ones(n)
    for c in constraints:
        A = np.atleast_2d(c.A)
        for row, lb, ub in zip(A, np.broadcast_to(c.lb, len(A)), np.broadcast_to(c.ub, len(A))):
            if np.count_nonzero(row) == 1:
                i = np.flatnonzero(row)[0]
                lower[i], upper[i] = max(lower[i], lb), min(upper[i], ub)
    return lower, upper


def retry_start(x0: np.ndarray, constraints: List[LinearConstraint]) -> np.ndarray:
    """Second starting point for minimize: equal weights, or, if x0 already is equal weights,
    equal weights moved inside the per-asset bounds: lower + t (upper - lower) with sum 1."""
    n = len(x0)
    equal = np.ones(n) / n
    if not np.allclose(x0, equal):
        return equal
    lower, upper = box_bounds(constraints, n)
    t = (1.0 - lower.sum()) / max((upper - lower).sum(), 1e-12)
    return lower + np.clip(t, 0.0, 1.0) * (upper - lower)


def efficient_frontier(x0: np.ndarray, mu: np.ndarray, sigma: np.ndarray, gam: float,
                       constraints: List[LinearConstraint]) -> Tuple[float, float, np.ndarray]:
    """Return, volatility and weights of the efficient portfolio at gam, starting from x0
    (efficient_frontier2 of Lab Stability LN3, with a tighter tolerance).

    With tol close to machine precision minimize may stop with a non-success status (e.g. 8,
    "positive directional derivative in linesearch") although it has reached the optimum, so the
    status is not the test: a solution is accepted only if all constraints hold within 1e-8.
    Otherwise minimize is run once more from a different start (retry_start); if that fails too,
    a plain-language error is raised.
    """
    lower, upper = box_bounds(constraints, len(x0))
    for start in (x0, retry_start(x0, constraints)):
        res = minimize(QP, start, args=(sigma, mu, gam), constraints=constraints,
                       tol=1e-15, options={"maxiter": 1000})
        # +-1e-17 outside a bound is rounding: an excluded asset gets exactly 0, never -0.0%
        optimized_weights = np.clip(res.x, lower, upper)
        if constraints_hold(optimized_weights, constraints):
            mu_optimized, vol_optimized = portfolio_stats(optimized_weights, mu, sigma)
            return mu_optimized, vol_optimized, optimized_weights
    raise RuntimeError("the optimiser could not find a portfolio that respects the investment rules "
                       "(fully invested, every asset class within its minimum and maximum weight), "
                       "even after a second attempt")


def frontier_weights(mu: np.ndarray, sigma: np.ndarray, gammas: np.ndarray,
                     tickers: List[str], config: Dict = CONFIG) -> np.ndarray:
    """Constrained frontier: x*(gamma) for each gamma (shape n_grid x n). Equally weighted start,
    then warm start from the solution of the previous gamma."""
    constraints = portfolio_constraints(tickers, config)
    x = np.ones(len(tickers)) / len(tickers)
    weights = []
    for gam in gammas:
        x = efficient_frontier(x, mu, sigma, gam, constraints)[2]
        weights.append(x)
    return np.array(weights)


def target_vol(x, target: float, mu: np.ndarray, sigma: np.ndarray,
               constraints: List[LinearConstraint]) -> float:
    """Volatility of the efficient portfolio at gamma = x minus the target (Lab Markowitz LN1)."""
    x0 = np.ones(len(mu)) / len(mu)
    return efficient_frontier(x0, mu, sigma, np.asarray(x).item(), constraints)[1] - target


def max_return_portfolio(mu: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """End of the constrained frontier (gamma -> infinity): the fully invested portfolio with the
    highest x'mu under lower <= x <= upper. Start from the minimum weights and give the rest of
    the budget to the assets with the highest mu, each up to its maximum."""
    x = np.array(lower, dtype=float)
    budget = 1.0 - x.sum()
    for i in np.argsort(-np.asarray(mu)):
        add = min(upper[i] - x[i], budget)
        x[i] += add
        budget -= add
    return x


def frontier_bounds(mu: np.ndarray, sigma: np.ndarray, tickers: List[str],
                    max_doublings: int = 60, config: Dict = CONFIG) -> Tuple[float, float, float]:
    """Volatility range of the constrained efficient frontier.

    Returns (sigma_minus, sigma_plus, gamma_hi): sigma_minus = sigma(x*(0)), the constrained
    minimum-variance portfolio; sigma_plus is the end of the frontier (gamma -> infinity),
    found by doubling gamma until sigma(x*(gamma)) stops increasing at the volatility of the
    maximum-return portfolio. (With low-return assets sigma hardly moves at small gamma: "stops
    increasing" alone would end the frontier too early.) gamma_hi, the end of the gamma grid, is
    one doubling further: a margin that keeps the grid of the app (and so its risk range)
    unchanged with respect to the previous solver.
    """
    gammas = np.append(0.0, 1e-3 * 2.0 ** np.arange(max_doublings + 1))
    constraints = portfolio_constraints(tickers, config)
    sig_end = portfolio_stats(max_return_portfolio(mu, *weight_bounds(tickers, config)), mu, sigma)[1]
    sig_minus = target_vol(0.0, 0.0, mu, sigma, constraints)
    sig_prev = target_vol(gammas[1], 0.0, mu, sigma, constraints)
    for gamma in gammas[2:]:
        sig_next = target_vol(gamma, 0.0, mu, sigma, constraints)
        if sig_next - sig_prev < _SIGMA_TOL and sig_next > sig_end - 10 * _SIGMA_TOL:
            break
        sig_prev = sig_next
    return sig_minus, max(sig_next, sig_minus), 2 * float(gamma)


def bisect_root(f, lo: float, hi: float, args: tuple, tol: float = _FSOLVE_TOL,
                max_iter: int = 200) -> Optional[float]:
    """Root of f(x, *args) between lo and hi with f(lo) < 0 <= f(hi), by bisection: fallback when
    fsolve stalls. Needed because with tight bounds sigma(x*(gamma)) can be flat over a range of
    gamma (the optimum stays at a corner of the constraints), where fsolve sees no slope.
    Returns None if |f| does not drop below tol."""
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gap = f(mid, *args)
        if abs(gap) < tol:
            return mid
        if gap < 0:
            lo = mid
        else:
            hi = mid
    return None


def solve_sigma(mu: np.ndarray, sigma: np.ndarray, sigma_target: float, tickers: List[str],
                bounds: Optional[Tuple[float, float, float]] = None,
                config: Dict = CONFIG) -> Tuple[float, str]:
    """sigma-problem: gamma such that sigma(x*(gamma)) = sigma_target, with fsolve (Lab Markowitz
    LN1 and Lab Stability LN3).

    Returns (gamma, status) with status "ok", "below_min" (target below the constrained
    minimum variance: gamma = 0) or "capped" (target above the frontier end: gamma = gamma_hi);
    outside the frontier fsolve would fail silently.

    As in the labs fsolve starts from a high gamma. Because of the bounds the volatility is flat
    at the end of the frontier, where fsolve cannot move: the start is halved until it is less
    than twice the solution, and again if fsolve does not hit the target. If no start works
    (flat stretches inside the frontier), the root is found by bisection on [0, gamma_hi]:
    sigma(x*(gamma)) is continuous and non-decreasing in gamma.
    """
    sig_minus, sig_plus, gamma_hi = bounds or frontier_bounds(mu, sigma, tickers, config=config)
    if sigma_target <= sig_minus + _SIGMA_TOL:
        return 0.0, "below_min"
    if sigma_target >= sig_plus - _SIGMA_TOL:
        return gamma_hi, "capped"
    args = (sigma_target, mu, sigma, portfolio_constraints(tickers, config))
    for start in gamma_hi / 2.0 ** np.arange(30):
        if target_vol(start / 2, *args) > 0:                    # start is still too high
            continue
        gamma = fsolve(target_vol, start, args=args, full_output=True)[0].item()
        if gamma >= 0 and abs(target_vol(gamma, *args)) < _FSOLVE_TOL:
            return gamma, "ok"
    gamma = bisect_root(target_vol, 0.0, gamma_hi, args)
    if gamma is not None:
        return gamma, "ok"
    raise RuntimeError(f"No portfolio with volatility {sigma_target:.4%} was found.")


def gamma_grid(mu: np.ndarray, sigma: np.ndarray, tickers: List[str],
               n_grid: int = CONFIG["n_grid"],
               bounds: Optional[Tuple[float, float, float]] = None,
               config: Dict = CONFIG) -> np.ndarray:
    """gamma values whose (point-estimate) portfolios are evenly spaced in volatility
    along the constrained frontier, from gamma = 0 to gamma_hi."""
    bounds = bounds or frontier_bounds(mu, sigma, tickers, config=config)
    sig_minus, sig_plus, gamma_hi = bounds
    targets = np.linspace(sig_minus, sig_plus, n_grid)[1:-1]
    inner = [solve_sigma(mu, sigma, s, tickers, bounds, config)[0] for s in targets]
    return np.unique(np.concatenate([[0.0], inner, [gamma_hi]]))


def resample_draws(mu_m: np.ndarray, sigma_m: np.ndarray, T: int, gammas: np.ndarray,
                   tickers: List[str], n_resamples: int = CONFIG["n_resamples"],
                   seed: int = CONFIG["seed"], config: Dict = CONFIG) -> np.ndarray:
    """Resampling, LN 3 slide 12, steps 1-3 (Michaud 2007), as in Lab Stability LN3.

    Step 1: draw T returns (T = number of historical observations) from a joint normal i.i.d.
    process with the empirical monthly means, variances and covariances; compute the sampled
    mean vector mu_k and covariance matrix Sigma_k (annualised like the point estimates).
    Step 2: constrained MV frontier of (mu_k, Sigma_k), on the same gamma grid for every draw.
    Step 3: repeat for n_resamples draws. Returns all weights (n_resamples x n_grid x n).
    """
    rng = np.random.default_rng(seed)
    all_weights = []
    for k in range(n_resamples):
        simulated_r = rng.multivariate_normal(mu_m, sigma_m, size=T)
        simulated_mu = simulated_r.mean(axis=0) * 12
        simulated_cov_mat = np.cov(simulated_r, rowvar=False) * 12
        all_weights.append(frontier_weights(simulated_mu, simulated_cov_mat, gammas, tickers, config))
    return np.array(all_weights)


def resample_weights(mu_m: np.ndarray, sigma_m: np.ndarray, T: int, gammas: np.ndarray,
                     tickers: List[str], n_resamples: int = CONFIG["n_resamples"],
                     seed: int = CONFIG["seed"], config: Dict = CONFIG) -> np.ndarray:
    """Resampling, LN 3 slide 12, step 4 (Lab Stability LN3): for each gamma, average the weights
    across draws (n_grid x n). The average of feasible portfolios is feasible because all
    constraints are linear. Risk and return of these portfolios use the original mu and Sigma."""
    return resample_draws(mu_m, sigma_m, T, gammas, tickers, n_resamples, seed, config).mean(axis=0)


def interp_weights(gamma, gammas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Resampled weights at gamma by linear interpolation between grid points (a convex
    combination of two feasible portfolios, hence feasible). gamma is clipped to the grid.
    Scalar gamma -> weights of shape (n,); gamma of shape (N,) -> weights of shape (N, n)."""
    return np.stack([np.interp(gamma, gammas, w) for w in weights.T], axis=-1)


def target_vol_resampled(x, target: float, gammas: np.ndarray, weights: np.ndarray,
                         sigma: np.ndarray) -> float:
    """Volatility, under the original Sigma, of the resampled weights at gamma = x minus the target."""
    w = interp_weights(np.asarray(x).item(), gammas, weights)
    return np.sqrt(w @ sigma @ w) - target


def resampled_frontier_bounds(gammas: np.ndarray, weights: np.ndarray,
                              sigma: np.ndarray) -> Tuple[float, float, float]:
    """Volatility range of the resampled frontier: (sigma_lo, sigma_hi, gamma_at_hi).

    sigma_lo is the volatility of the resampled portfolio at gamma = 0, sigma_hi the largest
    volatility on the grid (interpolated portfolios between grid points are convex combinations,
    so their volatility never exceeds that of the two end points).
    """
    vols = [target_vol_resampled(g, 0.0, gammas, weights, sigma) for g in gammas]
    j = np.argmax(vols)
    return float(vols[0]), float(vols[j]), float(gammas[j])


def solve_sigma_resampled(sigma_target: float, gammas: np.ndarray, weights: np.ndarray,
                          sigma: np.ndarray,
                          bounds: Optional[Tuple[float, float, float]] = None) -> Tuple[float, str]:
    """sigma-problem on the resampled frontier: gamma such that the volatility of the averaged
    (interpolated) weights, computed with the original Sigma, equals sigma_target.

    Takes the first grid segment whose end reaches the target. np.interp between its two ends is
    only approximate (volatility is not linear in gamma), so it is the starting point of fsolve.
    Statuses as in solve_sigma: "ok", "below_min" (gamma = gammas[0]), "capped" (gamma_at_hi).
    """
    lo, hi, gamma_hi = bounds or resampled_frontier_bounds(gammas, weights, sigma)
    if sigma_target <= lo + _SIGMA_TOL:
        return float(gammas[0]), "below_min"
    if sigma_target >= hi - _SIGMA_TOL:
        return gamma_hi, "capped"
    args = (sigma_target, gammas, weights, sigma)
    gaps = [target_vol_resampled(g, *args) for g in gammas]
    j = np.argmax(np.array(gaps) >= 0.0)           # first grid point at or above the target (j >= 1)
    start = np.interp(0.0, gaps[j - 1:j + 1], gammas[j - 1:j + 1])
    gamma = fsolve(target_vol_resampled, start, args=args, full_output=True)[0].item()
    gamma = float(np.clip(gamma, gammas[j - 1], gammas[j]))
    if abs(target_vol_resampled(gamma, *args)) > _FSOLVE_TOL:            # fallback: bisection
        gamma = bisect_root(target_vol_resampled, float(gammas[j - 1]), float(gammas[j]), args)
    if gamma is None:
        raise RuntimeError(f"No resampled portfolio with volatility {sigma_target:.4%} was found.")
    return gamma, "ok"
