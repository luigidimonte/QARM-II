"""FIRE planner - Streamlit UI (QARM II project).

UI only: all computations live in data.py, optimizer.py and fire.py; the look lives in ui_style.py.
Run with:  streamlit run app_fire.py
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import fire
import market as mk
import optimizer as opt
import ui_style as ui

# numpy 2.0.x + macOS Accelerate emits spurious FP warnings in matmul (results are correct).
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
# An excluded asset (lower = upper = 0) makes one row of LinearConstraint(np.eye(n), lower, upper) an
# equality: scipy only notes that splitting the constraint would be faster (results are the same).
warnings.filterwarnings("ignore", message="Equality and inequality constraints are specified")
st.set_page_config(page_title="Switzerland FIRE Planner", layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()

TICKERS: List[str] = list(data.UNIVERSE)
BLUE, STEEL, NAVY, GREY = ui.BLUE, ui.STEEL, ui.NAVY, ui.GREY
COLORS: Dict[str, str] = ui.ASSET_COLORS
PRESETS: Dict[str, Dict] = {
    "Early career (28)": dict(age=28, income=85_000, expenses=50_000, wealth=30_000, target_age=50),
    "Mid career (40)": dict(age=40, income=150_000, expenses=85_000, wealth=400_000, target_age=55),
    "Late starter (50)": dict(age=50, income=130_000, expenses=90_000, wealth=600_000, target_age=60),
}
DEFAULTS: Dict = dict(PRESETS["Mid career (40)"], name="", swr=4.0, infl=1.0, seed=fire.SEED,
                      hedged=True, rf_manual=False, rf_flat=0.0)
NO_PREF, OWN_PREF = "No preference", "Yes"
PREF_QUESTION = "Do you have preferences about where you would like to be invested?"
DEFAULT_LOWER, DEFAULT_UPPER = (tuple(float(v) for v in b) for b in opt.weight_bounds(TICKERS))
DEFAULTS.update(prefs=NO_PREF, rules=None, mu_adj=0.0, risk_manual=False,
                **{f"lo_{t}": round(lo * 100, 6) for t, lo in zip(TICKERS, DEFAULT_LOWER)},
                **{f"hi_{t}": round(hi * 100, 6) for t, hi in zip(TICKERS, DEFAULT_UPPER)})
DEFAULT_RULES_TEXT = (f"At most {opt.CONFIG['max_weight']:.0%} in any asset class, at most "
                      f"{opt.CONFIG['max_gold']:.0%} in gold; fully invested, no short selling.")
HEDGE_TEXT = ("Hedging removes the dollar/franc risk, but costs about the interest-rate gap between "
              "USD and CHF every year.")


# ----------------------------------------------------------------------------- formatting
def chf(x: float) -> str:
    return f"CHF\u00a0{x:,.0f}".replace(",", "'")     # non-breaking space: "CHF" never wraps off the number


def num(n: int) -> str:
    return f"{n:,}".replace(",", "'")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def label(t: str) -> str:
    return f"{data.UNIVERSE[t]} ({t})"


# ----------------------------------------------------------------------------- cached model
@st.cache_data(ttl=86_400, show_spinner=False)
def load_prices(tickers: tuple) -> pd.DataFrame:
    """Fresh daily prices (cached for one day)."""
    return data.download_prices(list(tickers))


@st.cache_data(ttl=86_400, show_spinner=False)
def load_market() -> mk.Market:
    """USD/CHF, US 1-year rate, Confederation curve and Swiss CPI (cached for one day)."""
    return mk.load_market()


def rules_config(tickers: List[str], lower: tuple, upper: tuple) -> Dict:
    """optimizer config with the per-asset bounds lower <= x <= upper."""
    return {**opt.CONFIG, "lower": dict(zip(tickers, lower)), "upper": dict(zip(tickers, upper))}


@st.cache_data(show_spinner=False)
def build_model(rets: pd.DataFrame, hedged: bool, n_resamples: int, seed: int,
                lower: tuple, upper: tuple, mu_adj: float = 0.0) -> Dict:
    """Estimates on real CHF returns, constrained frontier, gamma grid and resampled weights:
    the heavy step, run once when the questionnaire is submitted. One cache entry per data sample
    (which includes the hedging choice), set of bounds lower <= x <= upper, expected return
    adjustment, K and seed; it does not depend on any other answer. mu_adj (yearly, e.g. 0.01 = 1pp) is subtracted from the historical mu of
    every asset before the optimisation and the resampling draws; Sigma is unchanged."""
    tickers = list(rets.columns)
    config = rules_config(tickers, lower, upper)
    mu, sigma, T = data.estimate_moments(rets)
    mu = data.adjust_expected_returns(mu, mu_adj)
    mu_v, sig_v = mu.values, sigma.values
    bounds = opt.frontier_bounds(mu_v, sig_v, tickers, config=config)
    gammas = opt.gamma_grid(mu_v, sig_v, tickers, bounds=bounds, config=config)
    resampled = opt.resample_weights((mu / 12).values, (sigma / 12).values, T, gammas, tickers,
                                     n_resamples, seed, config)
    rs_bounds = opt.resampled_frontier_bounds(gammas, resampled, sig_v)
    return {"tickers": tickers, "mu": mu_v, "sigma": sig_v, "rs_bounds": rs_bounds, "gammas": gammas,
            "resampled": resampled, "mu_adj": mu_adj, "start": rets.index[0], "end": rets.index[-1]}


@st.cache_data(show_spinner=False)
def point_risk_range(rets: pd.DataFrame, lower: tuple, upper: tuple, mu_adj: float) -> tuple:
    """Volatility range (sigma_minus, sigma_plus) of the point-estimate constrained frontier: cheap,
    used only for the manual risk sliders before the plan is computed."""
    mu, sigma, _ = data.estimate_moments(rets)
    mu = data.adjust_expected_returns(mu, mu_adj)
    tickers = list(rets.columns)
    return opt.frontier_bounds(mu.values, sigma.values, tickers, config=rules_config(tickers, lower, upper))[:2]


@st.cache_data(show_spinner=False)
def gamma_bounds(sigma: np.ndarray, gammas: np.ndarray, weights: np.ndarray, rs_bounds: tuple,
                 sigma_min: float, sigma_max: float) -> Dict:
    return fire.gamma_bounds_from_sigma(sigma, gammas, weights, rs_bounds, sigma_min, sigma_max)


@st.cache_data(show_spinner=False, max_entries=32)
def simulate(profile: fire.ClientProfile, grid: fire.ResampledGrid, rf: mk.RiskFree,
             n_paths: int, seed: int) -> fire.WealthSimulation:
    """Monte Carlo of wealth; the cache key is every input (profile, weights, moments, curve, n, seed)."""
    return fire.simulate_wealth(profile, grid, rf, n_paths, seed)


def risk_setting(key: str) -> Optional[float]:
    """The manual risk level in session state, or None for the default (set from the resampled portfolios)."""
    s = st.session_state
    return s[key] / 100 if s["risk_manual"] and key in s else None


def resampled_grid(model: Dict):
    """(gamma bounds, resampled portfolios) for the risk range in session state (cheap: fsolve on the
    stored resampled weights)."""
    gb = gamma_bounds(model["sigma"], model["gammas"], model["resampled"], model["rs_bounds"],
                      risk_setting("sig_min"), risk_setting("sig_max"))
    return gb, fire.ResampledGrid(tuple(model["tickers"]), model["gammas"], model["resampled"],
                                  model["mu"], model["sigma"], gb["gamma_0"], gb["gamma_min"])


def risk_free() -> mk.RiskFree:
    """Real risk-free rate by horizon for the answers in session state."""
    s = st.session_state
    return mk.RiskFree(load_market().yields, s["infl"] / 100, s["rf_flat"] / 100 if s["rf_manual"] else None)


def run_simulation(profile: fire.ClientProfile, grid: fire.ResampledGrid) -> fire.WealthSimulation:
    """Same seed for every plan (common random numbers), so plans are compared on the same markets."""
    return simulate(profile, grid, risk_free(), fire.N_PATHS, int(st.session_state["seed"]))


def plan(model: Dict, profile: fire.ClientProfile):
    """(gamma bounds, resampled grid, simulation) for the answers in session state."""
    gb, grid = resampled_grid(model)
    return gb, grid, run_simulation(profile, grid)


# ----------------------------------------------------------------------------- questionnaire
STEPS = ["Welcome", "Your age", "Income and expenses", "Your wealth", "Investment preferences",
         "Assumptions", "Summary"]
PREFS, ASSUMPTIONS_STEP = STEPS.index("Investment preferences"), STEPS.index("Assumptions")
SUMMARY, RESULTS = len(STEPS) - 1, len(STEPS)


def init_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("step", 0)


def go_to(step: int) -> None:
    st.session_state["step"] = step


def apply_preset(name: str) -> None:
    st.session_state.update(PRESETS[name])


def choose_preset(name: str) -> None:
    """Example client: fill all answers and jump to the summary."""
    apply_preset(name)
    for k in [k for k in st.session_state.keys() if str(k).startswith("_")]:
        del st.session_state[k]                   # widgets must show the new values
    go_to(SUMMARY)
    st.toast(f"Example client loaded: {name}")


def draft_bounds() -> tuple:
    """(lower, upper) from the answers of the preferences step (not yet applied)."""
    s = st.session_state
    if s["prefs"] == NO_PREF:
        return DEFAULT_LOWER, DEFAULT_UPPER
    return (tuple(s[f"lo_{t}"] / 100 for t in TICKERS), tuple(s[f"hi_{t}"] / 100 for t in TICKERS))


def current_bounds() -> tuple:
    """(lower, upper) actually used: the rules applied when the preferences step was left."""
    rules = st.session_state["rules"]
    return (DEFAULT_LOWER, DEFAULT_UPPER) if rules is None else rules


def is_default(lower, upper) -> bool:
    return np.allclose(lower, DEFAULT_LOWER) and np.allclose(upper, DEFAULT_UPPER)


def apply_rules() -> None:
    """Leaving the preferences step: use the new bounds (feasible, checked on screen). New bounds
    give new portfolios, so the risk range restarts at its defaults."""
    lower, upper = draft_bounds()
    new = None if is_default(lower, upper) else (lower, upper)
    if new != st.session_state["rules"]:
        st.session_state["rules"] = new
        for k in ("sig_min", "sig_max", "_sig_min", "_sig_max"):
            st.session_state.pop(k, None)


def next_step(step: int) -> None:
    if step == PREFS:
        apply_rules()
    go_to(step + 1)


def reset_rules() -> None:
    for t, lo, hi in zip(TICKERS, DEFAULT_LOWER, DEFAULT_UPPER):
        st.session_state[f"lo_{t}"], st.session_state[f"hi_{t}"] = lo * 100, hi * 100
        st.session_state.pop(f"_lo_{t}", None)
        st.session_state.pop(f"_hi_{t}", None)


def rules_text(lower, upper) -> str:
    """Plain-language description of the bounds in use."""
    if is_default(lower, upper):
        return "Default rules: " + DEFAULT_RULES_TEXT[0].lower() + DEFAULT_RULES_TEXT[1:]
    parts = []
    for t, lo, hi, dlo, dhi in zip(TICKERS, lower, upper, DEFAULT_LOWER, DEFAULT_UPPER):
        if np.isclose(lo, dlo) and np.isclose(hi, dhi):
            continue
        name = data.UNIVERSE[t]
        if hi <= 0:
            parts.append(f"{name} excluded")
        elif lo > 0:
            parts.append(f"{name} {lo:.0%} - {hi:.0%}")
        else:
            parts.append(f"{name} at most {hi:.0%}")
    return ("Your own rules: " + "; ".join(parts) + ". Other asset classes follow the default rules; "
            "fully invested, no short selling.")


def set_hedged() -> None:
    """Currency hedging toggled: a new set of portfolios, so the risk range restarts at its defaults."""
    st.session_state["hedged"] = st.session_state["_hedged"]
    for k in ("sig_min", "sig_max", "_sig_min", "_sig_max"):
        st.session_state.pop(k, None)


def set_mu_adj() -> None:
    """Expected return adjustment changed: new portfolios, so the risk range restarts at its defaults."""
    st.session_state["mu_adj"] = st.session_state["_mu_adj"]
    for k in ("sig_min", "sig_max", "_sig_min", "_sig_max"):
        st.session_state.pop(k, None)


def mu_label(mu_adj: float) -> str:
    """How the expected returns shown were obtained."""
    return "historical estimate" + (f" minus {mu_adj * 100:.1f} pp" if mu_adj > 0 else "")


def hedge_toggle() -> None:
    field(st.toggle, "Currency hedging", "hedged", on_change=set_hedged)
    st.caption(HEDGE_TEXT)


def field(widget, label_text: str, key: str, *args, **kwargs):
    """Widget whose answer is kept while its step is not on screen
    (Streamlit forgets the state of widgets that are not displayed)."""
    wkey = "_" + key
    if wkey not in st.session_state:
        st.session_state[wkey] = st.session_state[key]
    value = widget(label_text, *args, key=wkey, **kwargs)
    st.session_state[key] = value
    return value


def manual_risk_range(rets: pd.DataFrame):
    """Range of the manual risk sliders (point-estimate frontier, cheap) and their starting values;
    keeps stored values inside the range."""
    lower, upper = current_bounds()
    sig_minus, sig_plus = point_risk_range(rets, lower, upper, st.session_state["mu_adj"] / 100)
    d_min, d_max = fire.default_sigma_range(sig_minus, sig_plus)
    lo, hi = round(sig_minus * 100, 2), round(sig_plus * 100, 2)
    st.session_state.setdefault("sig_min", round(d_min * 100, 2))
    st.session_state.setdefault("sig_max", round(d_max * 100, 2))
    for k in ("sig_min", "sig_max"):            # keep stored values inside today's feasible range
        st.session_state[k] = float(np.clip(st.session_state[k], lo, hi))
    return lo, hi


def profile_from_state() -> fire.ClientProfile:
    s = st.session_state
    return fire.ClientProfile(age=int(s["age"]), income=float(s["income"]), expenses=float(s["expenses"]),
                              wealth=float(s["wealth"]), target_age=int(s["target_age"]),
                              swr=s["swr"] / 100)


def profile_rows(s, gb: Optional[Dict] = None, start_vol: Optional[float] = None) -> List[tuple]:
    """(title, text, questionnaire step) for the answers given so far.

    All volatilities are those of real CHF returns, the same basis as the Portfolio tab. gb (gamma
    bounds of the computed plan) gives the risk levels actually used; start_vol is the real
    volatility of today's portfolio."""
    how = "set manually" if s["risk_manual"] else "default"
    if gb is not None:
        risk = (f"maximum risk setting {pct(gb['sigma_max'], 2)} (real) · "
                f"minimum risk setting {pct(gb['sigma_min'], 2)} (real) · {how}")
    elif s["risk_manual"] and "sig_min" in s:
        risk = (f"maximum risk setting {pct(s['sig_max'] / 100, 2)} (real) · "
                f"minimum risk setting {pct(s['sig_min'] / 100, 2)} (real) · set manually")
    else:
        risk = "default: set from the feasible portfolios when your plan is computed"
    if start_vol is not None:
        risk += f" · your starting volatility {pct(start_vol, 2)} (real)"
    return [("Age", f"{s['age']} years, goal at {s['target_age']}", 1),
            ("Income and expenses", f"{chf(s['income'])} income, {chf(s['expenses'])} expenses "
                                    f"({chf(s['income'] - s['expenses'])} saved per year)", 2),
            ("Invested wealth", chf(s["wealth"]), 3),
            ("Investment rules", rules_text(*current_bounds()), PREFS),
            ("Assumptions", f"withdrawal rate {s['swr']:.2f}%, inflation {s['infl']:.2f}%"
                            + (f", expected returns lowered by {s['mu_adj']:.1f} pp" if s["mu_adj"] > 0 else ""),
             ASSUMPTIONS_STEP),
            ("Currency hedging", "yes" if s["hedged"] else "no", ASSUMPTIONS_STEP),
            ("Risk range", risk, ASSUMPTIONS_STEP)]


def show_risk_free(h: int) -> None:
    """Read-only r_f at the initial horizon, with the date of the curve."""
    rf, m = risk_free(), load_market()
    if h <= 0:
        return
    if rf.override is not None:
        st.metric("Risk-free rate used (real)", pct(rf.override, 2))
        st.caption("Set manually, the same for every horizon ($r_f$).")
    else:
        st.metric(f"Risk-free rate for {h} years (real)", pct(rf.real(h), 2))
        st.caption(f"Swiss Confederation curve, {m.curve_date:%B %Y} (month-end): "
                   f"{pct(mk.nominal_rate(h, rf.yields), 2)} nominal, {pct(rf.inflation, 2)} inflation.")


def questionnaire(rets: pd.DataFrame) -> None:
    """One question per screen, then a summary; the results are shown after the last step."""
    s = st.session_state
    step = s["step"]
    st.progress(step / SUMMARY, text=f"Step {step + 1} of {len(STEPS)}  ·  {STEPS[step]}")
    ok = True
    with st.container(border=True):
        if step == 0:
            st.subheader("Welcome", anchor=False)
            st.write(f"Enter your name below to begin the {len(STEPS)}-step questionnaire, or load an "
                     f"example client.")
            field(st.text_input, "What is your name?", "name", placeholder="Optional")
            st.caption("Or start from an example client.")
            for col, name in zip(st.columns(len(PRESETS)), PRESETS):
                col.button(name, on_click=choose_preset, args=(name,), width="stretch")

        elif step == 1:
            st.subheader("Your age", anchor=False)
            age = field(st.slider, "How old are you?", "age", 18, 75)
            target = field(st.slider, "At what age would you like to be financially independent?",
                           "target_age", 19, 80)
            if target <= age:
                st.error("The target age must be later than your current age.")
                ok = False
            else:
                st.info(f"Your goal is **{target - age} years** from now.")

        elif step == 2:
            st.subheader("Income and expenses", anchor=False)
            income = field(st.number_input, "Net yearly income (CHF)", "income", 0, 500_000, step=1_000)
            expenses = field(st.number_input, "Yearly expenses (CHF)", "expenses", 5_000, 300_000, step=1_000,
                             help="Also the yearly spending you want to sustain once financially independent.")
            c1, c2 = st.columns(2)
            c1.metric("You save per year", chf(income - expenses))
            c2.metric("Savings rate", pct((income - expenses) / income, 0) if income > 0 else "-")
            if income <= expenses:
                st.warning("You are not saving anything: wealth can only grow through returns.")

        elif step == 3:
            st.subheader("Your wealth", anchor=False)
            wealth = field(st.number_input, "How much have you already invested (CHF)?", "wealth",
                           0, 5_000_000, step=10_000)
            if wealth <= 0 and s["income"] <= s["expenses"]:
                st.error("With no wealth and no savings there is nothing to invest. "
                         "Go back and check your income and expenses.")
                ok = False

        elif step == PREFS:
            st.subheader("Investment preferences", anchor=False)
            field(st.radio, PREF_QUESTION, "prefs", [NO_PREF, OWN_PREF], horizontal=True)
            if s["prefs"] == NO_PREF:
                st.info(f"**Default rules**  \n{DEFAULT_RULES_TEXT}")
            else:
                st.write("For each asset class, choose the minimum and maximum share of your portfolio. "
                         "Set the maximum to 0% to exclude an asset class.")
                widths = [4, 2, 2]
                head = st.columns(widths)
                for col, text in zip(head, ("Asset class", "Minimum (%)", "Maximum (%)")):
                    col.markdown(f"**{text}**")
                for t in TICKERS:
                    c1, c2, c3 = st.columns(widths, vertical_alignment="center")
                    c1.write(label(t))
                    with c2:
                        field(st.number_input, f"Minimum {t} (%)", f"lo_{t}", 0.0, 100.0, step=5.0,
                              format="%.0f", label_visibility="collapsed")
                    with c3:
                        field(st.number_input, f"Maximum {t} (%)", f"hi_{t}", 0.0, 100.0, step=5.0,
                              format="%.0f", label_visibility="collapsed")
                st.button("Reset to the default rules", on_click=reset_rules)
                st.caption("Fully invested and no short selling always apply.")
            lower, upper = draft_bounds()
            problems = opt.feasibility_problems(TICKERS, rules_config(TICKERS, lower, upper), data.UNIVERSE)
            for problem in problems:
                st.error(problem)
            ok = not problems
            if ok and (None if is_default(lower, upper) else (lower, upper)) != s["rules"]:
                st.caption("The portfolios are recomputed with these rules when you continue; this can take "
                           "a couple of minutes.")

        elif step == ASSUMPTIONS_STEP:
            st.subheader("Assumptions", anchor=False)
            st.write("These values are already set to sensible defaults; you can simply continue.")
            field(st.slider, "Safe withdrawal rate (%)", "swr", 2.5, 6.0, step=0.25,
                  help="Share of your wealth you plan to spend each year once independent.")
            hedge_toggle()
            st.info(f"**Investment rules**  \n{rules_text(*current_bounds())}")
            with st.expander("Advanced: risk range, expected returns, inflation and simulation"):
                field(st.checkbox, "Set the risk range manually", "risk_manual")
                if s["risk_manual"]:
                    lo, hi = manual_risk_range(rets)
                    st.caption(f"Feasible volatility with the investment rules: **{lo:.2f}% - {hi:.2f}%** "
                               f"(real, yearly, point estimates). The plan uses the resampled portfolios, whose "
                               f"range is slightly different: levels outside it are limited when the plan is "
                               f"computed.")
                    field(st.slider, "Lowest risk (volatility post-FIRE, %)", "sig_min", lo, hi, step=0.05)
                    field(st.slider, "Highest risk (volatility early on, %)", "sig_max", lo, hi, step=0.05)
                else:
                    st.caption("Default risk range, set when your plan is computed: lowest risk = volatility of "
                               "the minimum-risk portfolio + 0.5 pp, highest risk = 90% of the highest feasible "
                               "volatility.")
                field(st.slider, "Expected return adjustment (percentage points)", "mu_adj", 0.0, 4.0,
                      step=0.5, on_change=set_mu_adj,
                      help="Subtracted from the expected real return of every asset class. The expected "
                           "returns are historical averages (2004-2026) that include a long equity bull "
                           "market and the gold rally, so they may be optimistic. 0 = historical averages.")
                st.caption(f"By default (0) the expected returns are the historical means of "
                           f"{rets.index[0]:%b %Y} - {rets.index[-1]:%b %Y}, which can be optimistic: the period "
                           f"includes a long equity bull market and the gold rally. The same amount is subtracted "
                           f"from every asset class.")
                field(st.slider, "Expected inflation (%)", "infl", 0.0, 5.0, step=0.25,
                      help="Turns the nominal risk-free rate into a real one. Past returns are already "
                           "deflated with the Swiss consumer price index.")
                field(st.checkbox, "Set the risk-free rate manually", "rf_manual")
                if s["rf_manual"]:
                    field(st.number_input, "Flat real risk-free rate (%)", "rf_flat", -3.0, 5.0, step=0.1)
                show_risk_free(s["target_age"] - s["age"])
                field(st.number_input, "Random seed", "seed", 0, 2**31 - 1, step=1,
                      help="The same seed always gives the same simulated markets.")
            if s["risk_manual"] and s["sig_min"] > s["sig_max"]:
                st.error("The lowest risk cannot be higher than the highest risk.")
                ok = False
            elif s["risk_manual"] and fire.is_narrow_risk_range(s["sig_min"] / 100, s["sig_max"] / 100):
                st.warning(fire.NARROW_RANGE_NOTE)

        else:
            st.subheader(f"Summary{', ' + s['name'] if s['name'].strip() else ''}", anchor=False)
            errors, _ = fire.validate_profile(profile_from_state())
            for title, text, target_step in profile_rows(s):
                c1, c2, c3 = st.columns([2, 5, 1], vertical_alignment="center")
                c1.markdown(f"**{title}**")
                c2.write(text)
                c3.button("Edit", key=f"edit_{title}", on_click=go_to, args=(target_step,))
            st.caption("Showing your plan computes the portfolios once (about two minutes the first time). "
                       "Afterwards only a change of currency hedging, investment rules or expected return "
                       "adjustment computes them again.")
            for e in errors:
                st.error(e)
            ok = not errors

    back, _, nxt = st.columns([1, 2.5, 1])
    if step > 0:
        back.button("Back", on_click=go_to, args=(step - 1,), width="stretch")
    if step < SUMMARY:
        nxt.button("Next", type="primary", disabled=not ok, on_click=next_step, args=(step,),
                   width="stretch")
    else:
        nxt.button("Show my plan", type="primary", disabled=not ok, on_click=go_to, args=(RESULTS,),
                   width="stretch")


# ----------------------------------------------------------------------------- charts
HOVER_PCT = "Age %{x}<br>%{y:.1f}%<extra>%{fullData.name}</extra>"


def prob_chart(paths: Dict[str, pd.DataFrame], target_ages: Dict[str, int]) -> go.Figure:
    """Chance of FIRE by age; x axis until max(target + 15, 65) of the latest target (at most 100)."""
    fig = go.Figure()
    end = min(max(max(target_ages.values()) + 15, 65), 100)
    for i, (name, path) in enumerate(paths.items()):
        path = path.loc[path["age"] <= end]
        fig.add_trace(go.Scatter(x=path["age"], y=path["P"] * 100, name=name, mode="lines+markers",
                                 line=dict(width=2.5 if i == 0 else 2, dash=None if i == 0 else "dash",
                                           color=BLUE if i == 0 else STEEL),
                                 marker=dict(size=5), hovertemplate=HOVER_PCT))
    for lv in fire.PROB_LEVELS:
        fig.add_hline(y=lv * 100, line=dict(color=ui.LINE, width=1, dash="dot"),
                      annotation_text=f"{lv:.0%}", annotation_position="right",
                      annotation_font=dict(color=GREY, size=11))
    for i, (name, age) in enumerate(target_ages.items()):
        fig.add_vline(x=age, line=dict(color=NAVY if i == 0 else STEEL, width=1, dash="dash"),
                      annotation_text=f"target {age}" if i == 0 else f"alt. target {age}",
                      annotation_position="top left" if i == 0 else "bottom right",
                      annotation_font=dict(color=GREY, size=11))
    fig.update_layout(xaxis_title="Age", yaxis_title="Chance of being financially independent (%)",
                      yaxis_range=[0, 102], height=420, margin=dict(r=45))
    return fig


N_SAMPLE_PATHS = 50
POST_FIRE = "post-FIRE, minimum risk"


def fan_chart(sim: fire.WealthSimulation, K: float, target_age: int) -> go.Figure:
    """Percentiles of simulated wealth by age (10-90 and 25-75 bands, median) until fire.HORIZON_AGE
    and the first N_SAMPLE_PATHS simulated paths, coloured by the FIRE condition (W >= K at the
    target age). The y axis stops a little above the 90th percentile, so the bands stay readable."""
    path = sim.path
    fig = go.Figure()
    hover = "%{fullData.name}: CHF %{y:,.0f}<extra></extra>"
    sample = sim.wealth[:N_SAMPLE_PATHS]                # drawn first: the bands lie on top of them
    ok = sim.wealth_at(target_age)[:N_SAMPLE_PATHS] >= K
    for rank, (name, rows, color) in enumerate((("Sample paths that miss K", sample[~ok], ui.MISSED),
                                                ("Sample paths that reach K", sample[ok], ui.REACHED)), 2):
        xs = np.tile(np.append(path["age"].values.astype(float), np.nan), len(rows))   # NaN = pen up
        ys = np.column_stack([rows, np.full(len(rows), np.nan)]).ravel()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.7, color=color), opacity=0.25,
                                 connectgaps=False, showlegend=False, hoverinfo="skip", name=name))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=2, color=color),
                                 name=f"{name} ({len(rows)})", legendrank=6 - rank))  # legend key; Plotly lists fills last-rank first
    for rank, (lo, hi, fill, edge) in ((1, (10, 90, ui.FAN_OUTER, dict(width=1.2, color=STEEL))),
                                       (2, (25, 75, ui.FAN_INNER, dict(width=0.5, color=ui.FAN_INNER)))):
        fig.add_trace(go.Scatter(x=path["age"], y=path[f"q{hi}"], mode="lines", line=edge, showlegend=False,
                                 name=f"{hi}th percentile", hovertemplate=hover))
        fig.add_trace(go.Scatter(x=path["age"], y=path[f"q{lo}"], mode="lines", line=edge, showlegend=False,
                                 name=f"{lo}th percentile", hovertemplate=hover, fill="tonexty", fillcolor=fill))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{lo}th to {hi}th percentile",
                                 marker=dict(size=12, symbol="square", color=fill), legendrank=rank))
    fig.add_trace(go.Scatter(x=path["age"], y=path["q50"], name="Median", mode="lines",
                             line=dict(width=3, color=ui.SLATE_BLUE), hovertemplate=hover, legendrank=5))
    fig.add_hline(y=K, line=dict(color=NAVY, width=1.2, dash="dash"), annotation_text=f"FIRE capital {chf(K)}",
                  annotation_position="top left", annotation_font=dict(color=NAVY, size=12))
    fig.add_vline(x=target_age, line=dict(color=NAVY, width=1.2, dash="dash"),
                  annotation_text=f"target {target_age}", annotation_position="top left",
                  annotation_font=dict(color=NAVY, size=12))
    fig.update_layout(xaxis_title="Age", yaxis_title="Today's CHF", yaxis_tickformat=",.0f",
                      yaxis_range=[0, 1.1 * max(float(path["q90"].max()), K)], height=540,
                      hovermode="x unified")
    return fig


HIST_BINS, HIST_TOP = 50, 99.0                   # bins; percentile where the x axis stops


def wealth_histogram(wealth: np.ndarray, K: float, age: int) -> go.Figure:
    """Distribution of simulated wealth at the target age, split at the FIRE capital."""
    lo, hi = float(wealth.min()), max(float(np.percentile(wealth, HIST_TOP)), float(wealth.min()) + 1.0)
    width = (hi - lo) / HIST_BINS
    if lo < K < hi:                              # put a bin edge exactly on K, so colours are exact
        lo = K - np.ceil((K - lo) / width) * width
        hi = lo + np.ceil((hi - lo) / width) * width
    edges = np.linspace(lo, hi, int(round((hi - lo) / width)) + 1)
    counts, _ = np.histogram(wealth, bins=edges)
    left, right = edges[:-1], edges[1:]
    reached = (left + right) / 2 >= K
    fig = go.Figure()
    share = float(np.mean(wealth >= K))
    for name, mask, color in ((f"Below the FIRE capital ({pct(1 - share)})", ~reached, ui.MISSED),
                              (f"FIRE capital reached ({pct(share)})", reached, ui.REACHED)):
        fig.add_trace(go.Bar(x=((left + right) / 2)[mask], y=counts[mask] / len(wealth) * 100, name=name,
                             width=width, marker=dict(color=color, line=ui.OUTLINE),
                             customdata=np.column_stack([left[mask], right[mask]]),
                             hovertemplate="CHF %{customdata[0]:,.0f} - %{customdata[1]:,.0f}<br>"
                                           "%{y:.1f}% of paths<extra></extra>"))
    fig.add_vline(x=K, line=dict(color=NAVY, width=1, dash="dash"), annotation_text=f"FIRE capital {chf(K)}",
                  annotation_position="top right", annotation_font=dict(color=GREY, size=11))
    fig.update_layout(xaxis_title=f"Wealth at {age} (today's CHF)", yaxis_title="Share of paths (%)",
                      xaxis_tickformat=",.0f", height=380, barmode="overlay")
    return fig


def glide_chart(weights: pd.DataFrame) -> go.Figure:
    """Animated glide path: portfolio weights by ETF, one frame per year of age."""
    long = (weights.rename_axis("age").reset_index()
            .melt(id_vars="age", var_name="ETF", value_name="weight"))
    long["weight"] *= 100
    long["asset"] = long["ETF"].map(data.UNIVERSE)
    tickers = list(weights.columns)
    fig = px.bar(long, x="ETF", y="weight", color="ETF", animation_frame="age",
                 color_discrete_map=COLORS, category_orders={"ETF": tickers},
                 custom_data=["asset"], range_y=[0, max(45.0, long["weight"].max() * 1.1)],
                 labels={"weight": "Weight (%)", "ETF": "", "age": "Age"}, template="fire")
    hover = "%{customdata[0]} (%{x})<br>%{y:.1f}%<extra></extra>"
    fig.update_traces(hovertemplate=hover, marker_line=ui.OUTLINE, width=0.6)
    for fr in fig.frames:
        for d in fr.data:
            d.hovertemplate = hover
            d.marker.line = ui.OUTLINE
            d.width = 0.6
    menu = fig.layout.updatemenus[0]
    menu.buttons[0].args[1]["frame"]["duration"] = 450
    menu.buttons[0].args[1]["transition"]["duration"] = 300
    menu.update(x=0, y=-0.12, xanchor="left", yanchor="top", pad=dict(t=0, r=10),
                bgcolor="#FFFFFF", bordercolor=ui.LINE, font=dict(color=NAVY, size=12))
    menu.buttons[0].label, menu.buttons[1].label = "Play", "Pause"
    slider = fig.layout.sliders[0]
    slider.update(x=0.14, len=0.86, y=-0.08, pad=dict(t=0, b=0), bordercolor=ui.LINE, bgcolor=ui.MIST,
                  activebgcolor=BLUE, tickcolor=ui.LINE, font=dict(size=10, color=GREY),
                  currentvalue=dict(prefix="Age ", font=dict(size=13, color=NAVY), xanchor="left"),
                  transition=dict(duration=300))
    fig.update_layout(showlegend=False, height=440, margin=dict(t=30, b=90),
                      yaxis_title="Weight (%)", xaxis_title=None)
    return fig


def donut_chart(x: np.ndarray, tickers: List[str], wealth: Optional[float] = None) -> go.Figure:
    held = np.asarray(x) > 1e-9                       # excluded / unused assets have no slice
    x, tickers = np.asarray(x)[held], [t for t, h in zip(tickers, held) if h]
    fig = go.Figure(go.Pie(labels=tickers, values=x, hole=0.54, sort=False, direction="clockwise",
                           marker=dict(colors=[COLORS.get(t) for t in tickers],
                                       line=dict(color="#FFFFFF", width=1.5)),
                           textinfo="label", textposition="auto", textfont=dict(size=12),
                           insidetextorientation="horizontal",
                           customdata=[data.UNIVERSE[t] for t in tickers],
                           hovertemplate="%{customdata} (%{label})<br>%{value:.1%}<extra></extra>"))
    annotations = []
    if wealth is not None:
        label, value = ("Total invested", chf(wealth)) if wealth > 0 else ("Target mix", "100%")
        annotations.append(dict(
            text=f"<span style='font-size:11px;color:{GREY}'>{label}</span><br><b>{value}</b>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=NAVY),
        ))
    fig.update_layout(height=390, margin=dict(t=0, b=16, l=36, r=0), showlegend=False,
                      annotations=annotations)
    return fig
  


# ----------------------------------------------------------------------------- tables
PCT1 = "{:.1%}".format


def allocation_table(x: np.ndarray, wealth: float, tickers: List[str]) -> pd.DataFrame:
    out = pd.DataFrame({"ETF": tickers, "Asset class": [data.UNIVERSE[t] for t in tickers], "Weight": list(x)})
    if wealth > 0:
        out["Amount"] = [w * wealth for w in x]
    return out


def trade_table(row: pd.Series, tickers: List[str]) -> pd.DataFrame:
    out = []
    for t in tickers:
        amt = row[f"trade_amt_{t}"]
        out.append({"ETF": t, "Asset class": data.UNIVERSE[t],
                    "Weight after one year of market moves": row[f"drift_{t}"],
                    "New target weight": row[f"w_{t}"],
                    "Change (pp)": row[f"trade_pct_{t}"] * 100,
                    "Trade": "Buy" if amt >= 0 else "Sell",
                    "Amount": abs(amt)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- tabs
def mc_caption(summary: Dict, sim: fire.WealthSimulation) -> str:
    return (f"Monte Carlo standard error of the chance by the target age: "
            f"±{summary['se_target'] * 100:.2f} percentage points "
            f"({num(sim.n_paths)} simulated paths, seed {sim.seed}).")


def tab_profile(profile: fire.ClientProfile, sim: fire.WealthSimulation, gb: Dict) -> None:
    s, path = st.session_state, sim.path
    K = fire.fire_target(profile.expenses, profile.swr)
    at_target = path.loc[path["age"] == profile.target_age].iloc[0]

    st.subheader("Client profile", anchor=False)
    with st.container(border=True):
        for title, text, _ in profile_rows(s, gb, float(path["s"].iloc[0])):
            c1, c2 = st.columns([1.8, 5.2])
            c1.markdown(f'<div class="fp-eyebrow" style="margin-top:0.2rem">{title}</div>', unsafe_allow_html=True)
            c2.write(text)
    st.caption("Currency hedging, investment rules and the other answers are fixed for this plan. To change "
               "them, use *Change my answers* and show the plan again. Your starting volatility is below the "
               "maximum risk setting when part of your wealth is already invested: the maximum applies only "
               "while all of it is still future savings.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital needed for FIRE", chf(K), help="Yearly expenses / safe withdrawal rate")
    c2.metric("Yearly savings", chf(profile.savings))
    c3.metric("Savings rate", pct(profile.savings / profile.income, 0) if profile.income > 0 else "-")
    c4.metric(f"Median wealth at {profile.target_age}", chf(at_target["q50"]),
              delta=chf(at_target["q50"] - K) + " vs target",
              help="Half of the simulated paths end above this amount, half below.")
    st.divider()


    st.subheader("How your wealth could develop", anchor=False)
    ui.show(fan_chart(sim, K, profile.target_age))
    st.caption(f"Simulated wealth in today's CHF, with {N_SAMPLE_PATHS} of the {num(sim.n_paths)} simulated paths. "
               f"A path reaches the goal if its wealth satisfies $W_T \\ge K$ at age {profile.target_age}. "
               f"Average of the simulated paths at {profile.target_age}: {chf(at_target['W'])}. From "
               f"{profile.target_age} on you stop saving and spend {chf(profile.expenses)} a year from the "
               f"{POST_FIRE} portfolio.")

def tab_probability(profile: fire.ClientProfile, sim: fire.WealthSimulation, grid: fire.ResampledGrid,
                    summary: Dict) -> None:
    ages, path = summary["ages"], sim.path
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Chance of FIRE by {profile.target_age}", pct(summary["p_target"]))
    for col, lv in zip((c2, c3, c4), fire.PROB_LEVELS):
        col.metric(f"Age with {lv:.0%} chance", str(ages[lv]) if ages[lv] else
                   f"after {int(path['age'].iloc[-1])}")
    c5.metric(f"Chance your money lasts until {fire.HORIZON_AGE}", pct(summary["p_last"]))
    st.caption(mc_caption(summary, sim))
    st.divider()
    st.subheader(f"Possible wealth at {profile.target_age}", anchor=False)
    K = fire.fire_target(profile.expenses, profile.swr)
    ui.show(wealth_histogram(sim.wealth_at(profile.target_age), K, profile.target_age))
    st.caption(f"Distribution of the simulated paths; the top {100 - HIST_TOP:.0f}% of outcomes is not drawn.")
    st.divider()
    what_if(profile, sim, grid, summary)


def tab_portfolio(model: Dict, profile: fire.ClientProfile, path: pd.DataFrame) -> None:
    tickers = model["tickers"]
    W = fire.weights_table(path, tickers)
    x_today = W.iloc[0].values
    st.subheader("Your portfolio today", anchor=False)
    c1, c2 = st.columns([1.4, 1], gap="large", vertical_alignment="top")
    with c1:
        table = allocation_table(x_today, profile.wealth, tickers)
        ui.show_table(table, {k: f for k, f in {"Weight": PCT1, "Amount": chf}.items() if k in table})
        risk = (f"Expected return {pct(path['m'].iloc[0], 2)} (real, {mu_label(model['mu_adj'])}) and "
                f"volatility {pct(path['s'].iloc[0], 2)} (real) per year, i.e. after inflation.")
        if profile.wealth > 0:
            st.caption(f"{risk} The expected return is an average of past returns "
                       f"({model['start']:%b %Y} - {model['end']:%b %Y}), not a forecast.")
        else:
            st.caption(f"You start with CHF\u00a00. These weights apply to your first savings "
                       f"({chf(profile.savings)}). {risk}")
    with c2:
        ui.show(donut_chart(x_today, tickers, profile.wealth))
    st.divider()

    st.subheader("How the portfolio changes over time", anchor=False)
    st.caption("Press Play to follow the target weights year by year, or drag the slider to a given age.")
    ui.show(glide_chart(W))
    st.caption("While most of your future savings are still ahead, the portfolio takes more risk. "
               "As your wealth grows towards the FIRE capital it becomes more defensive. The actual portfolio "
               "adapts to your wealth each year; the chart shows the average over the simulated paths. "
               f"From age {profile.target_age} on you hold the {POST_FIRE} portfolio.")
    with st.expander("Model & methodology"):
        st.markdown(
            "For each risk-aversion level $\\gamma$ on the resampled frontier ($B = 1{,}000$ bootstrap draws on "
            "monthly real CHF returns), portfolio weights solve:"
        )
        st.latex(
            r"\max_{w} \; w' \hat{\mu}_{\text{adj}} - \frac{\gamma}{2}\, w' \hat{\Sigma} w "
            r"\quad \text{s.t.} \quad \mathbf{1}'w = 1, \; \ell \le w \le u"
        )
        st.markdown(
            "As wealth $W_t$ progresses toward the FIRE target $K = C_{\\text{annual}} / \\text{SWR}$, "
            "the glide path smoothly transitions risk exposure from $\\sigma_{\\max}$ toward $\\sigma_{\\min}$, "
            "holding the minimum-risk portfolio from the target age onward."
        )
    st.divider()

    if len(path) > 1:
        row = path.iloc[1]
        st.subheader(f"Rebalancing after the first year (age {int(row['age'])})", anchor=False)
        ui.show_table(trade_table(row, tickers),
                      {"Weight after one year of market moves": PCT1, "New target weight": PCT1,
                       "Change (pp)": "{:+.1f}".format, "Amount": chf})
        st.caption(f"Trades bring the portfolio back to its new target and invest this year's savings "
                   f"({chf(profile.savings)}); amounts in today's CHF, average of the simulated paths, no transaction costs.")


def reset_what_if(rate: int, age: int) -> None:
    """What if back to the answers of the questionnaire."""
    st.session_state["wi_rate"], st.session_state["wi_age"] = rate, age


def what_if(profile: fire.ClientProfile, sim: fire.WealthSimulation, grid: fire.ResampledGrid,
            summary: Dict) -> None:
    st.subheader("What if", anchor=False)
    st.caption("Change one or two levers and compare with your current plan.")
    base_rate = profile.savings / profile.income if profile.income > 0 else 0.0
    min_age = min(profile.age + 1, 80)
    initial_rate = int(round(min(max(base_rate, 0.0), 0.9) * 100))
    st.session_state.setdefault("wi_rate", initial_rate)
    st.session_state.setdefault("wi_age", profile.target_age)
    st.session_state["wi_age"] = int(np.clip(st.session_state["wi_age"], min_age, 80))
    st.button("Reset to initial answers", on_click=reset_what_if, args=(initial_rate, profile.target_age))
    c1, c2 = st.columns(2)
    alt_rate = c1.slider("Savings rate (%)", 0, 90, key="wi_rate",
                         help="Spending less also lowers the capital you need.") / 100
    alt_age = c2.slider("Target FIRE age", min_age, 80, key="wi_age") if min_age < 80 else 80
    # at the questionnaire's (rounded) savings rate the alternative keeps the exact expenses
    expenses = (profile.expenses if round(alt_rate * 100) == initial_rate
                else max(profile.income * (1 - alt_rate), 1.0))
    alt = profile.with_changes(expenses=expenses, target_age=alt_age)
    alt_path = run_simulation(alt, grid).path
    alt_summary = fire.fire_summary(alt_path, alt.target_age)
    ui.show(prob_chart({"Current plan": sim.path, "Alternative": alt_path},
                       {"Current plan": profile.target_age, "Alternative": alt.target_age}))

    def col(p: fire.ClientProfile, s: Dict) -> List[str]:
        a = s["ages"]
        return [pct(p.savings / p.income if p.income else 0, 0), chf(p.expenses),
                chf(fire.fire_target(p.expenses, p.swr)), str(p.target_age), pct(s["p_target"]),
                f"±{s['se_target'] * 100:.2f}\u00a0pp"] + \
               [str(a[lv]) if a[lv] else "-" for lv in fire.PROB_LEVELS] + [pct(s["p_last"])]

    idx = ["Savings rate", "Yearly expenses", "FIRE capital", "Target age", "Chance by target age",
           "Standard error"] + \
          [f"Age with {lv:.0%} chance" for lv in fire.PROB_LEVELS] + \
          [f"Chance your money lasts until {fire.HORIZON_AGE}"]
    ui.show_table(pd.DataFrame({"Current plan": col(profile, summary), "Alternative": col(alt, alt_summary)},
                               index=idx), hide_index=False, align="right")


# ----------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    ui.header(
        "FIRE Planner",
        "<strong>FIRE</strong> (<em>Financial Independence, Retire Early</em>) means building enough "
        f"invested wealth to live on your portfolio without depending on a salary. "
        "Use our tool to discover when you can achieve financial independence. "
        f"Our analysis is built on portfolio optimization across {len(TICKERS)} distinct securities, "
        "progressively lowering risk as you approach your goal. "
        "Enter your details below to uncover your ideal allocation and projected path to FIRE.",
    )

    lower, upper = current_bounds()
    try:                                              # before any computation
        opt.check_feasibility(TICKERS, rules_config(TICKERS, lower, upper), data.UNIVERSE)
    except ValueError as e:
        st.error(f"The investment rules cannot be met: {e}")
        st.stop()
    try:
        with st.spinner("Retrieving market data"):
            prices, market = load_prices(tuple(TICKERS)), load_market()
        month_end = data.month_end_prices(prices)
        rets = data.real_chf_returns(month_end, market.usdchf, market.usd_1y,
                                     market.chf_curve[1], market.cpi, bool(st.session_state["hedged"]))
    except Exception as e:  # network / provider problems
        st.error(f"Market data could not be downloaded right now ({type(e).__name__}). Please try again in a minute.")
        st.stop()
    for w in market.warnings:
        st.warning(w)
    problems = data.validate_returns(rets, TICKERS)
    if problems:
        st.error("The market data is not usable: " + " ".join(problems))
        st.stop()
    if st.session_state["step"] < RESULTS:
        with st.container(key="fp_quiz"):
            questionnaire(rets)
        return                    # not st.stop(): it would skip the clean-up that `field` relies on                   # not st.stop(): it would skip the clean-up that `field` relies on

    try:                          # the heavy step: once per submitted set of inputs (cached)
        with st.spinner("Computing your portfolios (1'000 resampled frontiers). This takes about two minutes "
                        "and runs again only if you change the hedging, the investment rules or the expected "
                        "return adjustment."):
            model = build_model(rets, bool(st.session_state["hedged"]), opt.CONFIG["n_resamples"],
                                opt.CONFIG["seed"], lower, upper, float(st.session_state["mu_adj"]) / 100)
    except Exception as e:
        st.error(f"The portfolio optimisation failed ({e}). Please reload the page.")
        st.stop()

    name = st.session_state["name"].strip()
    head, top1, top2 = st.columns([4, 1, 1], vertical_alignment="bottom")
    head.markdown('<div class="fp-eyebrow">Your plan</div>', unsafe_allow_html=True)
    if name:
        head.subheader(f"{name}, here is your plan", anchor=False)
    top1.button("Change my answers", on_click=go_to, args=(SUMMARY,), width="stretch", key="fp_change")
    top2.button("Start over", on_click=go_to, args=(0,), width="stretch", key="fp_restart")

    profile = profile_from_state()
    errors, warns = fire.validate_profile(profile)
    for e in errors:
        st.error(e)
    if errors:
        st.stop()
    for w in warns:
        st.warning(w)

    try:
        gb, grid, sim = plan(model, profile)
    except Exception as e:
        st.error(f"The plan could not be computed ({e}).")
        st.stop()
    for note in gb["notes"]:
        st.warning(note)
    summary = fire.fire_summary(sim.path, profile.target_age)

    t1, t2, t3 = st.tabs(["Client profile", "Portfolio", "Probability of FIRE"])
    with t1:
        tab_profile(profile, sim, gb)
    with t2:
        tab_portfolio(model, profile, sim.path)
    with t3:
        tab_probability(profile, sim, grid, summary)


main()