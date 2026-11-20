"""FIRE layer: target capital, dynamic risk tolerance, glide path and probability of FIRE.

All amounts are in real terms (today's money). The optimiser works on nominal moments;
portfolio moments are converted to real terms here with the exact Fisher relation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

import optimizer as opt
from data import to_real, vol_to_real

SIGMA_MIN_MARGIN = 0.005      # default sigma_min = constrained min-variance vol + 0.5pp
SIGMA_MAX_FRACTION = 0.90     # default sigma_max = 90% of the maximum feasible vol
PROB_LEVELS = (0.50, 0.75, 0.90)


@dataclass(frozen=True)
class ClientProfile:
    """Client inputs. Money amounts are yearly and in today's money."""
    age: int
    income: float
    expenses: float
    wealth: float
    target_age: int
    swr: float = 0.04
    rf_nominal: float = 0.01
    inflation: float = 0.015

    @property
    def savings(self) -> float:
        return yearly_savings(self.income, self.expenses)

    @property
    def rf_real(self) -> float:
        return float(to_real(self.rf_nominal, self.inflation))

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


def pv_savings(c: float, h: float, r: float) -> float:
    """H_t = c (1 - (1+r)^-h) / r : present value of the remaining savings (c*h if r ~ 0)."""
    h = max(h, 0.0)
    if abs(r) < 1e-9:
        return c * h
    return c * (1.0 - (1.0 + r) ** (-h)) / r


def dynamic_gamma(W: float, H: float, K: float, gamma_0: float, gamma_min: float) -> float:
    """gamma_t = gamma_min + (gamma_0 - gamma_min) * H_t / (W_t + H_t); gamma_min if W_t >= K."""
    if W >= K or W + H <= 0:
        return gamma_min
    share = min(max(H / (W + H), 0.0), 1.0)
    return gamma_min + (gamma_0 - gamma_min) * share


def default_sigma_range(sig_minus: float, sig_plus: float) -> Tuple[float, float]:
    """Defaults from the constrained frontier: (sigma_min, sigma_max)."""
    s_min = min(sig_minus + SIGMA_MIN_MARGIN, sig_plus)
    s_max = max(SIGMA_MAX_FRACTION * sig_plus, s_min)
    return s_min, s_max


def gamma_bounds_from_sigma(mu: np.ndarray, sigma: np.ndarray, tickers: List[str],
                            bounds: Tuple[float, float, float],
                            sigma_min: Optional[float] = None,
                            sigma_max: Optional[float] = None) -> Dict:
    """gamma_min and gamma_0 from sigma_min and sigma_max through the sigma-problem.

    User values are clipped to the feasible range [sigma-, sigma+]; sigma_min <= sigma_max is enforced.
    """
    sig_minus, sig_plus, _ = bounds
    d_min, d_max = default_sigma_range(sig_minus, sig_plus)
    s_min = d_min if sigma_min is None else sigma_min
    s_max = d_max if sigma_max is None else sigma_max
    notes: List[str] = []
    clipped_min, clipped_max = float(np.clip(s_min, sig_minus, sig_plus)), float(np.clip(s_max, sig_minus, sig_plus))
    if clipped_min != s_min or clipped_max != s_max:
        notes.append(f"Risk levels were limited to the feasible range "
                     f"{sig_minus:.1%} - {sig_plus:.1%} of the constrained portfolios.")
    if clipped_max < clipped_min:
        clipped_max = clipped_min
        notes.append("The maximum risk level was raised to the minimum risk level.")
    gamma_min, st_min = opt.solve_sigma(mu, sigma, clipped_min, tickers, bounds)
    gamma_0, st_max = opt.solve_sigma(mu, sigma, clipped_max, tickers, bounds)
    return {"sigma_min": clipped_min, "sigma_max": clipped_max, "gamma_min": gamma_min,
            "gamma_0": gamma_0, "status_min": st_min, "status_max": st_max, "notes": notes}


def horizon_end_age(p: ClientProfile) -> int:
    """Last age of the projection: max(target + 15, 65), capped at 100."""
    return int(min(max(p.target_age + 15, 65), 100))


def simulate_glide_path(p: ClientProfile, mu: np.ndarray, sigma: np.ndarray, tickers: List[str],
                        gammas: np.ndarray, grid_weights: np.ndarray,
                        gamma_0: float, gamma_min: float) -> pd.DataFrame:
    """Yearly loop on the expected wealth path.

    At each age: (W_t, h_t) -> H_t -> gamma_t -> resampled constrained weights x_t. Over the
    year the holdings drift with the (real) expected asset returns and the savings c are added;
    next year's trades are the new target weights minus the drifted weights.
    """
    K = fire_target(p.expenses, p.swr)
    c, r = p.savings, p.rf_real
    mu = np.asarray(mu, dtype=float)
    mu_real_assets = to_real(mu, p.inflation)
    rows: List[Dict] = []
    W = float(p.wealth)
    holdings: Optional[np.ndarray] = None          # money per ETF before rebalancing
    for age in range(p.age, horizon_end_age(p) + 1):
        h = max(p.target_age - age, 0)
        H = pv_savings(c, h, r)
        g = dynamic_gamma(W, H, K, gamma_0, gamma_min)
        x = opt.interp_weights(g, gammas, grid_weights)
        m_nom, s_nom = opt.portfolio_stats(x, mu, sigma)
        if holdings is None:                       # today: invest current wealth from cash
            before_w = np.zeros(len(x))
            trade_amt = x * W
        else:
            before_w = holdings / holdings.sum() if holdings.sum() > 0 else np.zeros(len(x))
            trade_amt = x * W - holdings
        row = {"age": age, "years": age - p.age, "W": W, "h": h, "H": H, "gamma": g,
               "m": float(to_real(m_nom, p.inflation)), "s": float(vol_to_real(s_nom, p.inflation)),
               "m_nominal": m_nom, "s_nominal": s_nom}
        for i, t in enumerate(tickers):
            row[f"w_{t}"] = x[i]
            row[f"drift_{t}"] = before_w[i]
            row[f"trade_pct_{t}"] = x[i] - before_w[i]
            row[f"trade_amt_{t}"] = trade_amt[i]
        rows.append(row)
        holdings = W * x * (1.0 + mu_real_assets)  # drift over the year
        W = float(holdings.sum() + c)              # = W (1 + m_real) + c
    path = pd.DataFrame(rows)
    path["K"] = K
    probs = fire_probability_lognormal(p.wealth, c, path["m"].values, path["s"].values, K)
    for col in ("M1", "M2", "P"):
        path[col] = probs[col]
    return path


# --------------------------------------------------------------------------------------
# Project layer (NOT from the course slides): probability of FIRE without Monte Carlo.
# --------------------------------------------------------------------------------------

def fire_probability_lognormal(W0: float, c: float, m: np.ndarray, s: np.ndarray,
                               K: float) -> Dict[str, np.ndarray]:
    """P(W_t >= K) at each date from the first two moments of wealth.

    W_{t+1} = W_t (1 + R_t) + c with E[R_t] = m_t, Var[R_t] = s_t^2 (R_t independent of W_t):
        M1_{t+1} = M1_t (1 + m_t) + c
        M2_{t+1} = M2_t ((1 + m_t)^2 + s_t^2) + 2 c (1 + m_t) M1_t + c^2
    A lognormal is fitted by moment matching: s^2 = ln(M2 / M1^2), m = ln M1 - s^2 / 2,
    and P(W_t >= K) = 1 - Phi((ln K - m) / s). Element t uses the return of the year
    starting at t, so element 0 is today's (known) wealth.
    """
    n = len(m)
    M1, M2 = np.empty(n), np.empty(n)
    M1[0], M2[0] = W0, W0 ** 2
    for t in range(n - 1):
        g = 1.0 + m[t]
        M1[t + 1] = M1[t] * g + c
        M2[t + 1] = M2[t] * (g ** 2 + s[t] ** 2) + 2.0 * c * g * M1[t] + c ** 2
    P = np.empty(n)
    for t in range(n):
        if M1[t] <= 0:
            P[t] = 0.0
            continue
        ratio = M2[t] / M1[t] ** 2
        if ratio <= 1.0 + 1e-14:                   # no uncertainty: deterministic wealth
            P[t] = 1.0 if M1[t] >= K else 0.0
            continue
        s2 = np.log(ratio)
        mean_log = np.log(M1[t]) - s2 / 2.0
        P[t] = 1.0 - norm.cdf((np.log(K) - mean_log) / np.sqrt(s2))
    return {"M1": M1, "M2": M2, "P": P}


def fire_summary(path: pd.DataFrame, target_age: int,
                 levels: Tuple[float, ...] = PROB_LEVELS) -> Dict:
    """P(FIRE) at the target age and the first ages at which P reaches each level."""
    at_target = path.loc[path["age"] == target_age, "P"]
    ages: Dict[float, Optional[int]] = {}
    for lv in levels:
        hit = path.loc[path["P"] >= lv, "age"]
        ages[lv] = int(hit.iloc[0]) if len(hit) else None
    return {"p_target": float(at_target.iloc[0]) if len(at_target) else float("nan"), "ages": ages}


def weights_table(path: pd.DataFrame, tickers: List[str], prefix: str = "w_") -> pd.DataFrame:
    """Columns of the path with the given prefix, indexed by age, named by ticker."""
    out = path.set_index("age")[[f"{prefix}{t}" for t in tickers]]
    out.columns = tickers
    return out
