"""Constrained mean-variance optimisation (QARM II, LN 1 and LN 3).

gamma-problem (LN 1):  x*(gamma) = argmin 1/2 x'Sigma x - gamma x'mu
                       u.c. 1'x = 1, 0 <= x_i <= 0.40, x_gold >= 0.10
gamma is the risk TOLERANCE (gamma >= 0); gamma = 0 gives the constrained minimum-variance
portfolio. Because of the box constraints there is no closed form: the QP is solved
numerically with cvxpy.

sigma-problem (LN 1): find gamma such that sigma(x*(gamma)) = sigma*, by root finding.
Resampling (LN 3, Michaud): sample T returns from N(mu_hat, Sigma_hat), re-estimate,
solve the same constrained problem, average the weights.
The risk targets of the plan are hit on the RESAMPLED frontier (the weights actually held):
solve_sigma_resampled finds gamma such that the averaged weights, evaluated with the original
mu and Sigma, have volatility sigma*. solve_sigma (point frontier) only builds the gamma grid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import brentq

CONFIG: Dict = {
    "max_weight": 0.40,      # x_i <= 40% for every asset
    "min_gold": 0.10,        # x_gold >= 10%
    "gold_ticker": "GLD",
    "n_resamples": 200,
    "n_grid": 40,
    "seed": 42,
}
MONTHS_PER_YEAR = 12
_SIGMA_TOL = 1e-7


def check_feasibility(tickers: List[str], config: Dict = CONFIG) -> None:
    """Raise ValueError (plain-language message) if the constraints cannot be met."""
    n = len(tickers)
    cap, gold_min = config["max_weight"], config["min_gold"]
    if n * cap < 1.0 - 1e-12:
        raise ValueError(f"With a {cap:.0%} cap per asset at least {int(np.ceil(1 / cap))} assets "
                         f"are needed to be fully invested; the universe has {n}.")
    if config["gold_ticker"] not in tickers:
        raise ValueError(f"The universe must contain gold ({config['gold_ticker']}) "
                         f"to satisfy the {gold_min:.0%} gold floor.")
    if gold_min > cap:
        raise ValueError("The gold floor is larger than the per-asset cap.")


class GammaProblem:
    """Parameterised constrained gamma-problem, compiled once and re-solved for new inputs.

    The quadratic term is written as 1/2 ||L'x||^2 with Sigma = L L' and the linear term as
    (gamma mu)'x, so that cvxpy can reuse the compiled problem (DPP) when mu, Sigma or gamma change.
    """

    def __init__(self, tickers: List[str], config: Dict = CONFIG):
        check_feasibility(tickers, config)
        n = len(tickers)
        self.n = n
        self.x = cp.Variable(n)
        self.L = cp.Parameter((n, n))
        self.gamma_mu = cp.Parameter(n)
        gold = tickers.index(config["gold_ticker"])
        constraints = [cp.sum(self.x) == 1,
                       self.x >= 0,
                       self.x <= config["max_weight"],
                       self.x[gold] >= config["min_gold"]]
        objective = 0.5 * cp.sum_squares(self.L.T @ self.x) - self.gamma_mu @ self.x
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        self.mu: Optional[np.ndarray] = None

    def set_moments(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Set expected returns and covariance matrix (annualised)."""
        sigma = np.asarray(sigma, dtype=float)
        sigma = 0.5 * (sigma + sigma.T)
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(sigma + 1e-12 * np.eye(self.n))
        self.L.value = L
        self.mu = np.asarray(mu, dtype=float)

    def solve(self, gamma: float) -> np.ndarray:
        """Return x*(gamma) for the current moments."""
        if gamma < 0:
            raise ValueError("gamma (risk tolerance) must be >= 0.")
        self.gamma_mu.value = gamma * self.mu
        for solver in ("CLARABEL", "OSQP"):
            try:
                self.problem.solve(solver=solver)
            except cp.error.SolverError:
                continue
            if self.problem.status == cp.OPTIMAL:
                return np.asarray(self.x.value, dtype=float)
        raise RuntimeError(f"Optimiser failed (status: {self.problem.status}).")


def _problem(mu: np.ndarray, sigma: np.ndarray, tickers: List[str]) -> GammaProblem:
    prob = GammaProblem(tickers)
    prob.set_moments(mu, sigma)
    return prob


def portfolio_stats(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Tuple[float, float]:
    """Expected return mu(x) = x'mu and volatility sigma(x) = sqrt(x'Sigma x)."""
    x, mu, sigma = np.asarray(x), np.asarray(mu), np.asarray(sigma)
    return float(x @ mu), float(np.sqrt(max(x @ sigma @ x, 0.0)))


def solve_gamma(mu: np.ndarray, sigma: np.ndarray, gamma: float, tickers: List[str]) -> np.ndarray:
    """Solve the constrained gamma-problem once."""
    return _problem(mu, sigma, tickers).solve(gamma)


def _vol(prob: GammaProblem, gamma: float, sigma: np.ndarray) -> float:
    x = prob.solve(gamma)
    return float(np.sqrt(max(x @ sigma @ x, 0.0)))


def frontier_bounds(mu: np.ndarray, sigma: np.ndarray, tickers: List[str],
                    max_doublings: int = 60) -> Tuple[float, float, float]:
    """Volatility range of the constrained efficient frontier.

    Returns (sigma_minus, sigma_plus, gamma_hi): sigma_minus = sigma(x*(0)), the constrained
    minimum-variance portfolio; sigma_plus is the end of the frontier (gamma -> infinity),
    reached at gamma_hi, found by doubling gamma until sigma(x*(gamma)) stops increasing.
    """
    sigma = np.asarray(sigma, dtype=float)
    prob = _problem(mu, sigma, tickers)
    sig_minus = _vol(prob, 0.0, sigma)
    gamma, sig_prev = 1e-3, _vol(prob, 1e-3, sigma)
    for _ in range(max_doublings):
        sig_next = _vol(prob, 2 * gamma, sigma)
        gamma *= 2
        if sig_next - sig_prev < _SIGMA_TOL:
            break
        sig_prev = sig_next
    return sig_minus, max(sig_next, sig_minus), gamma


def solve_sigma(mu: np.ndarray, sigma: np.ndarray, sigma_target: float, tickers: List[str],
                bounds: Optional[Tuple[float, float, float]] = None) -> Tuple[float, str]:
    """sigma-problem: gamma such that sigma(x*(gamma)) = sigma_target (brentq).

    Returns (gamma, status) with status "ok", "below_min" (target below the constrained
    minimum variance: gamma = 0) or "capped" (target above the frontier end: gamma = gamma_hi).
    """
    sigma = np.asarray(sigma, dtype=float)
    sig_minus, sig_plus, gamma_hi = bounds or frontier_bounds(mu, sigma, tickers)
    if sigma_target <= sig_minus + _SIGMA_TOL:
        return 0.0, "below_min"
    if sigma_target >= sig_plus - _SIGMA_TOL:
        return gamma_hi, "capped"
    prob = _problem(mu, sigma, tickers)
    gamma = brentq(lambda g: _vol(prob, g, sigma) - sigma_target, 0.0, gamma_hi,
                   xtol=1e-12, rtol=1e-10, maxiter=200)
    return float(gamma), "ok"


def gamma_grid(mu: np.ndarray, sigma: np.ndarray, tickers: List[str],
               n_grid: int = CONFIG["n_grid"],
               bounds: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """gamma values whose (point-estimate) portfolios are evenly spaced in volatility
    along the constrained frontier, from gamma = 0 to gamma_hi."""
    bounds = bounds or frontier_bounds(mu, sigma, tickers)
    sig_minus, sig_plus, gamma_hi = bounds
    targets = np.linspace(sig_minus, sig_plus, n_grid)[1:-1]
    inner = [solve_sigma(mu, sigma, s, tickers, bounds)[0] for s in targets]
    grid = np.unique(np.concatenate([[0.0], inner, [gamma_hi]]))
    return grid


def frontier_weights(mu: np.ndarray, sigma: np.ndarray, gammas: np.ndarray,
                     tickers: List[str]) -> np.ndarray:
    """Point-estimate constrained frontier: x*(gamma) for each gamma (shape n_grid x n)."""
    prob = _problem(mu, sigma, tickers)
    return np.array([prob.solve(g) for g in gammas])


def resample_draws(mu_m: np.ndarray, sigma_m: np.ndarray, T: int, gammas: np.ndarray,
                   tickers: List[str], n_resamples: int = CONFIG["n_resamples"],
                   seed: int = CONFIG["seed"]) -> np.ndarray:
    """LN 3 resampling, steps 1-3. For each draw: sample T monthly returns from
    N(mu_hat, Sigma_hat), re-estimate and annualise mu and Sigma, solve the same constrained
    gamma-problem on the gamma grid. Returns all weights (n_resamples x n_grid x n)."""
    rng = np.random.default_rng(seed)
    prob = GammaProblem(tickers)
    out = np.empty((n_resamples, len(gammas), len(tickers)))
    for k in range(n_resamples):
        sample = rng.multivariate_normal(np.asarray(mu_m), np.asarray(sigma_m), size=T)
        prob.set_moments(sample.mean(axis=0) * MONTHS_PER_YEAR,
                         np.cov(sample, rowvar=False) * MONTHS_PER_YEAR)
        for j, g in enumerate(gammas):
            out[k, j] = prob.solve(g)
    return out


def resample_weights(mu_m: np.ndarray, sigma_m: np.ndarray, T: int, gammas: np.ndarray,
                     tickers: List[str], n_resamples: int = CONFIG["n_resamples"],
                     seed: int = CONFIG["seed"]) -> np.ndarray:
    """LN 3 step 4: average the resampled weights (n_grid x n). The average of feasible
    portfolios is feasible because all constraints are linear."""
    return resample_draws(mu_m, sigma_m, T, gammas, tickers, n_resamples, seed).mean(axis=0)


def interp_weights(gamma: float, gammas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Resampled weights at gamma by linear interpolation between grid points (a convex
    combination of two feasible portfolios, hence feasible). gamma is clipped to the grid."""
    g = float(np.clip(gamma, gammas[0], gammas[-1]))
    return np.array([np.interp(g, gammas, weights[:, i]) for i in range(weights.shape[1])])


def resampled_vol(gamma: float, gammas: np.ndarray, weights: np.ndarray, sigma: np.ndarray) -> float:
    """Volatility, under the original Sigma, of the resampled weights interpolated at gamma."""
    x = interp_weights(gamma, gammas, weights)
    return float(np.sqrt(max(x @ np.asarray(sigma) @ x, 0.0)))


def resampled_frontier_bounds(gammas: np.ndarray, weights: np.ndarray,
                              sigma: np.ndarray) -> Tuple[float, float, float]:
    """Volatility range of the resampled frontier: (sigma_lo, sigma_hi, gamma_at_hi).

    sigma_lo is the volatility of the resampled portfolio at gamma = 0, sigma_hi the largest
    volatility on the grid (interpolated portfolios between grid points are convex combinations,
    so their volatility never exceeds that of the two end points).
    """
    vols = np.array([resampled_vol(g, gammas, weights, sigma) for g in gammas])
    j = int(np.argmax(vols))
    return float(vols[0]), float(vols[j]), float(gammas[j])


def solve_sigma_resampled(sigma_target: float, gammas: np.ndarray, weights: np.ndarray,
                          sigma: np.ndarray,
                          bounds: Optional[Tuple[float, float, float]] = None) -> Tuple[float, str]:
    """sigma-problem on the resampled frontier: gamma such that the volatility of the averaged
    (interpolated) weights, computed with the original Sigma, equals sigma_target.

    Takes the smallest such gamma: the first grid segment whose end reaches the target, then
    Brent's method inside it (volatility is continuous in gamma along the interpolation).
    Statuses as in solve_sigma: "ok", "below_min" (gamma = gammas[0]), "capped" (gamma_at_hi).
    """
    lo, hi, gamma_hi = bounds or resampled_frontier_bounds(gammas, weights, sigma)
    if sigma_target <= lo + _SIGMA_TOL:
        return float(gammas[0]), "below_min"
    if sigma_target >= hi - _SIGMA_TOL:
        return gamma_hi, "capped"
    f = lambda g: resampled_vol(g, gammas, weights, sigma) - sigma_target
    vals = np.array([f(g) for g in gammas])
    j = int(np.argmax(vals >= 0.0))                # first grid point at or above the target (j >= 1)
    if vals[j] == 0.0:
        return float(gammas[j]), "ok"
    gamma = brentq(f, gammas[j - 1], gammas[j], xtol=1e-14, rtol=1e-12, maxiter=200)
    return float(gamma), "ok"
