"""FIRE layer: target capital, dynamic risk tolerance, Monte Carlo simulation of wealth.

All amounts are in real terms (today's CHF). mu and Sigma are estimated on real CHF returns, so
nothing is deflated here; expected inflation only enters the real risk-free rate (market.RiskFree).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import optimizer as opt
from market import RiskFree

SIGMA_MIN_MARGIN = 0.005      # default sigma_min = constrained min-variance vol + 0.5pp
SIGMA_MAX_FRACTION = 0.90     # default sigma_max = 90% of the maximum feasible vol
PROB_LEVELS = (0.50, 0.75, 0.90)
PERCENTILES = (5, 25, 50, 75, 95)
N_PATHS = 10_000
SEED = 42
CLIP_TOL = 1e-4               # ignore clipping smaller than 0.01pp (rounding of UI inputs)
MIN_RISK_SPREAD = 0.005       # below 0.5pp between sigma_min and sigma_max the glide path is ~flat
NARROW_RANGE_NOTE = ("The risk range is (almost) zero wide, so the portfolio will hardly change over "
                     "time. Widen the gap between the lowest and highest risk to get a glide path.")


@dataclass(frozen=True)
class ClientProfile:
    """Client inputs. Money amounts are yearly and in today's CHF."""
    age: int
    income: float
    expenses: float
    wealth: float
    target_age: int
    swr: float = 0.04

    @property
    def savings(self) -> float:
        return yearly_savings(self.income, self.expenses)

    def with_changes(self, **kwargs) -> "ClientProfile":
        return replace(self, **kwargs)


def validate_profile(p: ClientProfile) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) in plain language."""
    errors: List[str] = []
    warnings: List[str] = []
    if p.target_age <= p.age:
        errors.append("The target FIRE age must be later than the current age.")
    if p.swr <= 0:
        errors.append("The safe withdrawal rate must be positive.")
    if p.expenses <= 0:
        errors.append("Yearly expenses must be positive.")
    if p.wealth < 0:
        errors.append("Current wealth cannot be negative.")
    if p.savings <= 0:
        warnings.append("Expenses are at least as high as income: there are no yearly savings, "
                        "so wealth can only grow through returns.")
    return errors, warnings


def fire_target(expenses: float, swr: float) -> float:
    """FIRE capital K = E / SWR."""
    return expenses / swr


def yearly_savings(income: float, expenses: float) -> float:
    """c = income - expenses."""
    return income - expenses


def human_capital(c: float, h, r):
    """H_t = c (1 - (1+r)^-h) / r : present value of the remaining savings (c*h if r ~ 0).
    Works on scalars and arrays."""
    h = np.maximum(h, 0.0)
    zero = np.abs(r) < 1e-9
    safe = np.where(zero, 1.0, r)
    return np.where(zero, c * h, c * (1.0 - (1.0 + safe) ** (-h)) / safe)[()]


def gamma_t(W, c: float, h_t: float, r_f, K: float, gamma_0: float, gamma_min: float):
    """gamma_t = gamma_min + (gamma_0 - gamma_min) * H_t / (W_t + H_t); gamma_min if W_t >= K.

    At and after the FIRE age (h_t <= 0) there are no savings left: gamma_min, r_f is not used.
    W may be a scalar or an array (one W_t per simulated path).
    """
    W = np.asarray(W, dtype=float)
    if h_t <= 0:
        return np.full(W.shape, gamma_min)[()]
    H = human_capital(c, h_t, r_f)
    total = W + H
    share = np.clip(H / np.where(total > 0, total, 1.0), 0.0, 1.0)
    return np.where((W >= K) | (total <= 0), gamma_min, gamma_min + (gamma_0 - gamma_min) * share)[()]


def default_sigma_range(sig_minus: float, sig_plus: float) -> Tuple[float, float]:
    """Defaults from the feasible volatility range (resampled frontier): (sigma_min, sigma_max)."""
    s_min = min(sig_minus + SIGMA_MIN_MARGIN, sig_plus)
    s_max = max(SIGMA_MAX_FRACTION * sig_plus, s_min)
    return s_min, s_max


def is_narrow_risk_range(sigma_min: float, sigma_max: float) -> bool:
    """True if sigma_max - sigma_min is too small for gamma_t (hence the weights) to move."""
    return sigma_max - sigma_min < MIN_RISK_SPREAD - 1e-12


def gamma_bounds_from_sigma(sigma: np.ndarray, gammas: np.ndarray, weights: np.ndarray,
                            bounds: Tuple[float, float, float],
                            sigma_min: Optional[float] = None,
                            sigma_max: Optional[float] = None) -> Dict:
    """gamma_min and gamma_0 from sigma_min and sigma_max through the sigma-problem on the
    resampled frontier, so that the portfolios actually held at gamma_min and gamma_0 have
    exactly these volatilities (real CHF returns, original Sigma).

    `bounds` is opt.resampled_frontier_bounds(gammas, weights, sigma). User values are clipped to
    that feasible range; sigma_min <= sigma_max is enforced. A note is added when the range is
    narrower than MIN_RISK_SPREAD (the glide path is then ~flat).
    """
    sig_minus, sig_plus, _ = bounds
    d_min, d_max = default_sigma_range(sig_minus, sig_plus)
    s_min = d_min if sigma_min is None else sigma_min
    s_max = d_max if sigma_max is None else sigma_max
    notes: List[str] = []
    clipped_min, clipped_max = float(np.clip(s_min, sig_minus, sig_plus)), float(np.clip(s_max, sig_minus, sig_plus))
    if max(abs(clipped_min - s_min), abs(clipped_max - s_max)) > CLIP_TOL:
        notes.append(f"Risk levels were limited to the feasible range "
                     f"{sig_minus:.1%} - {sig_plus:.1%} of the constrained portfolios.")
    if clipped_max < clipped_min:
        clipped_max = clipped_min
        notes.append("The maximum risk level was raised to the minimum risk level.")
    if is_narrow_risk_range(clipped_min, clipped_max):
        notes.append(NARROW_RANGE_NOTE)
    gamma_min, st_min = opt.solve_sigma_resampled(clipped_min, gammas, weights, sigma, bounds)
    gamma_0, st_max = opt.solve_sigma_resampled(clipped_max, gammas, weights, sigma, bounds)
    return {"sigma_min": clipped_min, "sigma_max": clipped_max, "gamma_min": gamma_min,
            "gamma_0": gamma_0, "status_min": st_min, "status_max": st_max, "notes": notes}


def horizon_end_age(p: ClientProfile) -> int:
    """Last age of the projection: max(target + 15, 65), capped at 100."""
    return int(min(max(p.target_age + 15, 65), 100))


@dataclass(frozen=True)
class ResampledGrid:
    """Resampled portfolios, computed once: weights (n_grid x n) on the gamma grid, the original
    annual moments of the real CHF returns, and the gamma range from the sigma-problem."""
    tickers: Tuple[str, ...]
    gammas: np.ndarray
    weights: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    gamma_0: float
    gamma_min: float


@dataclass(frozen=True)
class WealthSimulation:
    """Result of simulate_wealth. `path` has one row per age (averages over the simulated paths,
    percentiles of wealth, probability of FIRE); `wealth` is n_paths x number of ages."""
    path: pd.DataFrame
    wealth: np.ndarray
    n_paths: int
    seed: int

    def wealth_at(self, age: int) -> np.ndarray:
        """Simulated wealth of every path at the given age."""
        return self.wealth[:, int(age) - int(self.path["age"].iloc[0])]


def target_portfolio(W, h: float, p: ClientProfile, grid: ResampledGrid, rf: RiskFree):
    """(W_t, h_t) -> real r_f(h_t) -> H_t -> gamma_t -> resampled constrained weights x_t.

    W is one value or one value per path. Returns (r_f or NaN if h_t <= 0, H_t, gamma_t, x_t).
    """
    r = rf.real(h) if h > 0 else float("nan")
    H = float(human_capital(p.savings, h, r)) if h > 0 else 0.0
    g = gamma_t(W, p.savings, h, r, fire_target(p.expenses, p.swr), grid.gamma_0, grid.gamma_min)
    return r, H, g, opt.interp_weights(g, grid.gammas, grid.weights)


# --------------------------------------------------------------------------------------
# Project layer (NOT from the course slides): Monte Carlo simulation of wealth.
# --------------------------------------------------------------------------------------

def lognormal_parameters(mu: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(m, A) such that 1 + R = exp(m + A z), z ~ N(0, I), has mean 1 + mu and covariance sigma.

    Joint lognormal gross returns matched to the first two moments: V = ln(1 + sigma_ij /
    ((1+mu_i)(1+mu_j))) = A A', m = ln(1+mu) - diag(V)/2. An eigenvalue factor is used so that a
    singular sigma (e.g. zero volatility) is allowed.
    """
    gross = 1.0 + np.asarray(mu, dtype=float)
    V = np.log1p(np.asarray(sigma, dtype=float) / np.outer(gross, gross))
    values, vectors = np.linalg.eigh(V)
    if values.min() < -1e-10:
        raise ValueError("The covariance matrix cannot be matched by lognormal returns.")
    return np.log(gross) - np.diag(V) / 2.0, vectors * np.sqrt(np.maximum(values, 0.0))


def simulate_wealth(p: ClientProfile, grid: ResampledGrid, rf: RiskFree,
                    n_paths: int = N_PATHS, seed: int = SEED) -> WealthSimulation:
    """Simulate n_paths yearly wealth paths until horizon_end_age, all in real terms.

    Each year, on every path: W_t -> gamma_t -> x_t (target_portfolio), then real asset returns
    are drawn jointly (lognormal_parameters, independent across years), and
        W_{t+1} = W_t (1 + x_t'R_t) + c        (savings c enter at the end of the year).
    Normals are drawn year by year from default_rng(seed), so two runs with the same seed and
    n_paths share their random numbers even if the profiles or horizons differ.
    """
    K, c = fire_target(p.expenses, p.swr), p.savings
    ages = np.arange(p.age, horizon_end_age(p) + 1)
    n = len(grid.tickers)
    m_log, A = lognormal_parameters(grid.mu, grid.sigma)
    rng = np.random.default_rng(seed)
    wealth = np.empty((n_paths, len(ages)))
    wealth[:, 0] = p.wealth
    before = np.zeros((n_paths, n))                # money per ETF before rebalancing (today: cash)
    rows: List[Dict] = []
    for t, age in enumerate(ages):
        W = wealth[:, t]
        h = max(p.target_age - int(age), 0)
        r, H, g, x = target_portfolio(W, h, p, grid, rf)
        m = x @ grid.mu
        s = np.sqrt(np.maximum(np.einsum("ki,ij,kj->k", x, grid.sigma, x), 0.0))
        invested = before.sum(axis=1, keepdims=True)
        drift = np.where(invested > 0, before / np.where(invested > 0, invested, 1.0), 0.0)
        trade = x * W[:, None] - before
        P = float(np.mean(W >= K))
        row = {"age": int(age), "years": t, "W": W.mean(), "h": h, "rf": r, "H": H, "gamma": g.mean(),
               "m": m.mean(), "s": s.mean(),
               "K": K, "P": P, "SE": np.sqrt(P * (1.0 - P) / n_paths)}
        row.update({f"q{q}": v for q, v in zip(PERCENTILES, np.percentile(W, PERCENTILES))})
        for i, tick in enumerate(grid.tickers):
            row[f"w_{tick}"] = x[:, i].mean()
            row[f"drift_{tick}"] = drift[:, i].mean()
            row[f"trade_pct_{tick}"] = x[:, i].mean() - drift[:, i].mean()
            row[f"trade_amt_{tick}"] = trade[:, i].mean()
        rows.append(row)
        if t + 1 < len(ages):
            R = np.exp(m_log + rng.standard_normal((n_paths, n)) @ A.T) - 1.0
            before = W[:, None] * x * (1.0 + R)          # drift over the year
            wealth[:, t + 1] = before.sum(axis=1) + c
    return WealthSimulation(pd.DataFrame(rows), wealth, n_paths, seed)


def fire_summary(path: pd.DataFrame, target_age: int,
                 levels: Tuple[float, ...] = PROB_LEVELS) -> Dict:
    """P(FIRE) at the target age, its Monte Carlo standard error, and the first ages at which P
    reaches each level."""
    at_target = path.loc[path["age"] == target_age].iloc[0]
    ages: Dict[float, Optional[int]] = {}
    for lv in levels:
        hit = path.loc[path["P"] >= lv, "age"]
        ages[lv] = int(hit.iloc[0]) if len(hit) else None
    return {"p_target": float(at_target["P"]), "se_target": float(at_target["SE"]), "ages": ages}


def weights_table(path: pd.DataFrame, tickers: List[str], prefix: str = "w_") -> pd.DataFrame:
    """Columns of the path with the given prefix, indexed by age, named by ticker."""
    out = path.set_index("age")[[f"{prefix}{t}" for t in tickers]]
    out.columns = tickers
    return out
