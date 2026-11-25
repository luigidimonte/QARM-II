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

st.set_page_config(page_title="FIRE Planner", layout="wide", initial_sidebar_state="collapsed")
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
HEDGE_TEXT = ("Hedging removes the dollar/franc risk, but costs about the interest-rate gap between "
              "USD and CHF every year.")


# ----------------------------------------------------------------------------- formatting
def chf(x: float) -> str:
    return f"CHF {x:,.0f}".replace(",", "'")


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


@st.cache_data(show_spinner=False)
def build_model(rets: pd.DataFrame, hedged: bool, n_resamples: int, seed: int) -> Dict:
    """Estimates on real CHF returns, constrained frontier, gamma grid and resampled weights
    (one cache entry per dataset and hedging choice)."""
    tickers = list(rets.columns)
    mu, sigma, T = data.estimate_moments(rets)
    mu_v, sig_v = mu.values, sigma.values
    bounds = opt.frontier_bounds(mu_v, sig_v, tickers)
    gammas = opt.gamma_grid(mu_v, sig_v, tickers, bounds=bounds)
    point = opt.frontier_weights(mu_v, sig_v, gammas, tickers)
    resampled = opt.resample_weights((mu / 12).values, (sigma / 12).values, T, gammas, tickers,
                                     n_resamples, seed)
    rs_bounds = opt.resampled_frontier_bounds(gammas, resampled, sig_v)
    shrunk, delta = data.shrunk_covariance(rets, sigma)
    return {"tickers": tickers, "mu": mu_v, "sigma": sig_v, "T": T, "bounds": bounds, "rs_bounds": rs_bounds,
            "gammas": gammas, "point": point, "resampled": resampled,
            "shrunk": shrunk.values, "delta": delta, "hedged": hedged, "n_resamples": n_resamples,
            "start": rets.index[0], "end": rets.index[-1]}


@st.cache_data(show_spinner=False)
def gamma_bounds(sigma: np.ndarray, gammas: np.ndarray, weights: np.ndarray, rs_bounds: tuple,
                 sigma_min: float, sigma_max: float) -> Dict:
    return fire.gamma_bounds_from_sigma(sigma, gammas, weights, rs_bounds, sigma_min, sigma_max)


@st.cache_data(show_spinner=False, max_entries=32)
def simulate(profile: fire.ClientProfile, grid: fire.ResampledGrid, rf: mk.RiskFree,
             n_paths: int, seed: int) -> fire.WealthSimulation:
    """Monte Carlo of wealth; the cache key is every input (profile, weights, moments, curve, n, seed)."""
    return fire.simulate_wealth(profile, grid, rf, n_paths, seed)


def resampled_grid(model: Dict):
    """(gamma bounds, resampled portfolios) for the current risk range in session state."""
    s = st.session_state
    gb = gamma_bounds(model["sigma"], model["gammas"], model["resampled"], model["rs_bounds"],
                      s["sig_min"] / 100, s["sig_max"] / 100)
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
STEPS = ["Welcome", "Your age", "Income and expenses", "Your wealth", "Assumptions", "Summary"]
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


def reset_risk(sig_min: float, sig_max: float) -> None:
    st.session_state["sig_min"], st.session_state["sig_max"] = sig_min * 100, sig_max * 100
    st.session_state.pop("_sig_min", None)
    st.session_state.pop("_sig_max", None)
    st.toast("Risk range reset to defaults")


def set_hedged() -> None:
    """Currency hedging toggled: a new set of portfolios, so the risk range restarts at its defaults."""
    st.session_state["hedged"] = st.session_state["_hedged"]
    for k in ("sig_min", "sig_max", "_sig_min", "_sig_max"):
        st.session_state.pop(k, None)


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


def risk_range(model: Dict):
    """Feasible volatility range (resampled portfolios) and defaults; keeps stored values inside the range."""
    sig_minus, sig_plus, _ = model["rs_bounds"]
    d_min, d_max = fire.default_sigma_range(sig_minus, sig_plus)
    lo, hi = round(sig_minus * 100, 2), round(sig_plus * 100, 2)
    st.session_state.setdefault("sig_min", round(d_min * 100, 2))
    st.session_state.setdefault("sig_max", round(d_max * 100, 2))
    for k in ("sig_min", "sig_max"):            # keep stored values inside today's feasible range
        st.session_state[k] = float(np.clip(st.session_state[k], lo, hi))
    return lo, hi, d_min, d_max


def profile_from_state() -> fire.ClientProfile:
    s = st.session_state
    return fire.ClientProfile(age=int(s["age"]), income=float(s["income"]), expenses=float(s["expenses"]),
                              wealth=float(s["wealth"]), target_age=int(s["target_age"]),
                              swr=s["swr"] / 100)


def profile_rows(s, start_vol: Optional[float] = None) -> List[tuple]:
    """(title, text, questionnaire step) for the answers given so far.

    All volatilities are those of real CHF returns, the same basis as the Portfolio tab;
    start_vol is the real volatility of today's portfolio."""
    risk = (f"maximum risk setting {pct(s['sig_max'] / 100, 2)} (real) · "
            f"minimum risk setting {pct(s['sig_min'] / 100, 2)} (real)")
    if start_vol is not None:
        risk += f" · your starting volatility {pct(start_vol, 2)} (real)"
    return [("Age", f"{s['age']} years, goal at {s['target_age']}", 1),
            ("Income and expenses", f"{chf(s['income'])} income, {chf(s['expenses'])} expenses "
                                    f"({chf(s['income'] - s['expenses'])} saved per year)", 2),
            ("Invested wealth", chf(s["wealth"]), 3),
            ("Assumptions", f"withdrawal rate {s['swr']:.2f}%, inflation {s['infl']:.2f}%", 4),
            ("Currency hedging", "yes" if s["hedged"] else "no", 4),
            ("Risk range", risk, 4)]


def show_risk_free(h: int) -> None:
    """Read-only r_f at the initial horizon, with the date of the curve."""
    rf, m = risk_free(), load_market()
    if h <= 0:
        return
    if rf.override is not None:
        st.metric("Risk-free rate used (real)", pct(rf.override, 2))
        st.caption("Set manually, the same for every horizon.")
    else:
        st.metric(f"Risk-free rate for {h} years (real)", pct(rf.real(h), 2))
        st.caption(f"Swiss Confederation curve, {m.curve_date:%B %Y} (month-end): "
                   f"{pct(mk.nominal_rate(h, rf.yields), 2)} nominal, {pct(rf.inflation, 2)} inflation.")


def questionnaire(model: Dict) -> None:
    """One question per screen, then a summary; the results are shown after the last step."""
    s = st.session_state
    step = s["step"]
    st.progress(step / SUMMARY, text=f"Step {step + 1} of {len(STEPS)}  ·  {STEPS[step]}")
    ok = True
    with st.container(border=True):
        if step == 0:
            st.subheader("Welcome", anchor=False)
            st.write("**FIRE** means *Financial Independence, Retire Early*: saving and investing enough "
                     "to live on your investments and stop depending on a salary. Answer a few questions "
                     "and we will show you how likely you are to get there, and how to invest on the way.")
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

        elif step == 4:
            st.subheader("Assumptions", anchor=False)
            st.write("These values are already set to sensible defaults; you can simply continue.")
            field(st.slider, "Safe withdrawal rate (%)", "swr", 2.5, 6.0, step=0.25,
                  help="Share of your wealth you plan to spend each year once independent.")
            hedge_toggle()
            st.info(f"**Portfolio rules (fixed)**  \n"
                    f"At most {opt.CONFIG['max_weight']:.0%} in any single ETF.  \n"
                    f"At least {opt.CONFIG['min_gold']:.0%} in gold.  \n"
                    f"Fully invested, no short selling.")
            lo, hi, d_min, d_max = risk_range(model)
            with st.expander("Advanced: risk range, inflation and simulation"):
                st.caption(f"Feasible volatility with the portfolio rules: **{lo:.2f}% - {hi:.2f}%** "
                           f"(real, yearly).")
                field(st.slider, "Lowest risk (volatility post-FIRE, %)", "sig_min", lo, hi, step=0.05)
                field(st.slider, "Highest risk (volatility early on, %)", "sig_max", lo, hi, step=0.05)
                st.button("Reset to defaults", on_click=reset_risk, args=(d_min, d_max))
                field(st.slider, "Expected inflation (%)", "infl", 0.0, 5.0, step=0.25,
                      help="Turns the nominal risk-free rate into a real one. Past returns are already "
                           "deflated with the Swiss consumer price index.")
                field(st.checkbox, "Set the risk-free rate manually", "rf_manual")
                if s["rf_manual"]:
                    field(st.number_input, "Flat real risk-free rate (%)", "rf_flat", -3.0, 5.0, step=0.1)
                show_risk_free(s["target_age"] - s["age"])
                field(st.number_input, "Random seed", "seed", 0, 2**31 - 1, step=1,
                      help="The same seed always gives the same simulated markets.")
            if s["sig_min"] > s["sig_max"]:
                st.error("The lowest risk cannot be higher than the highest risk.")
                ok = False
            elif fire.is_narrow_risk_range(s["sig_min"] / 100, s["sig_max"] / 100):
                st.warning(fire.NARROW_RANGE_NOTE)

        else:
            st.subheader(f"Summary{', ' + s['name'] if s['name'].strip() else ''}", anchor=False)
            errors, _ = fire.validate_profile(profile_from_state())
            start_vol = None
            if not errors:
                try:
                    start_vol = fire.start_volatility(profile_from_state(), resampled_grid(model)[1], risk_free())
                except Exception:          # the results page reports plan failures
                    pass
            for title, text, target_step in profile_rows(s, start_vol):
                c1, c2, c3 = st.columns([2, 5, 1], vertical_alignment="center")
                c1.markdown(f"**{title}**")
                c2.write(text)
                c3.button("Edit", key=f"edit_{title}", on_click=go_to, args=(target_step,))
            if start_vol is not None and s["sig_max"] - s["sig_min"] > 1e-9:
                st.caption("Your starting volatility is below the maximum risk setting when part of your "
                           "wealth is already invested: the maximum applies only while all of it is still "
                           "future savings.")
            for e in errors:
                st.error(e)
            ok = not errors

    back, nxt, _ = st.columns([1, 1, 4])
    if step > 0:
        back.button("Back", on_click=go_to, args=(step - 1,), width="stretch")
    if step < SUMMARY:
        nxt.button("Next", type="primary", disabled=not ok, on_click=go_to, args=(step + 1,),
                   width="stretch")
    else:
        nxt.button("Show my plan", type="primary", disabled=not ok, on_click=go_to, args=(RESULTS,),
                   width="stretch")


# ----------------------------------------------------------------------------- charts
HOVER_PCT = "Age %{x}<br>%{y:.1f}%<extra>%{fullData.name}</extra>"


def prob_chart(paths: Dict[str, pd.DataFrame], target_ages: Dict[str, int]) -> go.Figure:
    fig = go.Figure()
    for i, (name, path) in enumerate(paths.items()):
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


N_SAMPLE_PATHS = 100
POST_FIRE = "post-FIRE, minimum risk"


def fan_chart(sim: fire.WealthSimulation, K: float, target_age: int) -> go.Figure:
    """Percentiles of simulated wealth by age (5-95 and 25-75 bands, median) and the first
    N_SAMPLE_PATHS simulated paths, coloured by the FIRE condition (W >= K at the target age)."""
    path = sim.path
    fig = go.Figure()
    hover = "%{fullData.name}: CHF %{y:,.0f}<extra></extra>"
    for lo, hi, fill in ((5, 95, ui.FAN_OUTER), (25, 75, ui.FAN_INNER)):
        edge = dict(width=0.5, color=fill)
        fig.add_trace(go.Scatter(x=path["age"], y=path[f"q{hi}"], mode="lines", line=edge, showlegend=False,
                                 name=f"{hi}th percentile", hovertemplate=hover))
        fig.add_trace(go.Scatter(x=path["age"], y=path[f"q{lo}"], mode="lines", line=edge, showlegend=False,
                                 name=f"{lo}th percentile", hovertemplate=hover, fill="tonexty", fillcolor=fill))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{lo}th to {hi}th percentile",
                                 marker=dict(size=12, symbol="square", color=fill)))     # legend key
    sample = sim.wealth[:N_SAMPLE_PATHS]
    ok = sim.wealth_at(target_age)[:N_SAMPLE_PATHS] >= K
    for name, rows, color in (("Sample paths that reach K", sample[ok], ui.REACHED),
                              ("Sample paths that miss K", sample[~ok], ui.MISSED)):
        xs = np.tile(np.append(path["age"].values.astype(float), np.nan), len(rows))   # NaN = pen up
        ys = np.column_stack([rows, np.full(len(rows), np.nan)]).ravel()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.8, color=color), opacity=0.35,
                                 connectgaps=False, showlegend=False, hoverinfo="skip", name=name))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=2, color=color),
                                 name=f"{name} ({len(rows)})"))                        # legend key
    fig.add_trace(go.Scatter(x=path["age"], y=path["q50"], name="Median", mode="lines",
                             line=dict(width=2.5, color=BLUE), hovertemplate=hover))
    fig.add_hline(y=K, line=dict(color=NAVY, width=1, dash="dash"), annotation_text=f"FIRE capital {chf(K)}",
                  annotation_position="top left", annotation_font=dict(color=GREY, size=11))
    fig.add_vline(x=target_age, line=dict(color=NAVY, width=1, dash="dash"),
                  annotation_text=f"target {target_age}", annotation_position="top left",
                  annotation_font=dict(color=GREY, size=11))
    fig.update_layout(xaxis_title="Age", yaxis_title="Today's CHF", yaxis_tickformat=",.0f",
                      height=460, hovermode="x unified")
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


def area_chart(weights: pd.DataFrame, x_title: str, x=None) -> go.Figure:
    fig = go.Figure()
    xs = weights.index if x is None else x
    for t in weights.columns:
        fig.add_trace(go.Scatter(x=xs, y=weights[t] * 100, name=label(t), stackgroup="one",
                                 mode="lines", line=dict(width=0.5, color=COLORS.get(t)),
                                 fillcolor=COLORS.get(t),
                                 hovertemplate=f"{t}  %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(xaxis_title=x_title, yaxis_title="Weight (%)", yaxis_range=[0, 100],
                      height=400, hovermode="x unified", legend=dict(y=-0.25))
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


def risk_chart(path: pd.DataFrame, target_age: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path["age"], y=path["s"] * 100, name="Volatility (real, %)",
                             mode="lines", line=dict(width=2.5, color=BLUE), hovertemplate=HOVER_PCT))
    fig.add_trace(go.Scatter(x=path["age"], y=path["m"] * 100, name="Expected return (real, %)",
                             mode="lines", line=dict(width=2.5, dash="dash", color=STEEL),
                             hovertemplate=HOVER_PCT))
    fig.add_trace(go.Scatter(x=path["age"], y=path["gamma"], name="Risk tolerance γₜ (right axis)",
                             mode="lines", line=dict(width=1.5, color=ui.GOLD), yaxis="y2",
                             hovertemplate="Age %{x}<br>γ %{y:.3f}<extra></extra>"))
    fig.add_vline(x=target_age, line=dict(color=NAVY, width=1, dash="dash"),
                  annotation_text=POST_FIRE, annotation_position="top right",
                  annotation_font=dict(color=GREY, size=11))
    fig.update_layout(xaxis_title="Age", yaxis_title="%", height=380,
                      yaxis2=dict(title=dict(text="γₜ", font=dict(color=GREY, size=12)), overlaying="y",
                                  side="right", showgrid=False, zeroline=False),
                      legend=dict(y=-0.25))
    return fig


def frontier_chart(model: Dict, path: pd.DataFrame, x_today: np.ndarray) -> go.Figure:
    """Risk/return map (real CHF returns), the same basis as the rest of the Portfolio tab."""
    mu, sigma = model["mu"], model["sigma"]

    def stats(W: np.ndarray):
        return zip(*[opt.portfolio_stats(x, mu, sigma) for x in W])

    hover = "Volatility %{x:.1f}% (real)<br>Return %{y:.1f}% (real)<extra>%{fullData.name}</extra>"
    m_pt, s_pt = stats(model["point"])
    m_rs, s_rs = stats(model["resampled"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(s_pt) * 100, y=np.array(m_pt) * 100, mode="lines",
                             name="Constrained efficient frontier", line=dict(color=BLUE, width=2.5),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=np.array(s_rs) * 100, y=np.array(m_rs) * 100, mode="lines",
                             name="Resampled portfolios", line=dict(dash="dash", color=STEEL, width=2.5),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=np.sqrt(np.diag(sigma)) * 100,
                             y=mu * 100, mode="markers+text",
                             text=model["tickers"], textposition="top center", name="Single ETFs",
                             textfont=dict(color=GREY, size=11),
                             marker=dict(size=10, color=[COLORS.get(t) for t in model["tickers"]],
                                         line=dict(color=NAVY, width=0.5)),
                             hovertemplate="%{text}<br>Volatility %{x:.1f}% (real)<br>Return %{y:.1f}% (real)<extra></extra>"))
    m0, s0 = opt.portfolio_stats(x_today, mu, sigma)
    x_end = fire.weights_table(path, model["tickers"]).iloc[-1].values
    m1, s1 = opt.portfolio_stats(x_end, mu, sigma)
    fig.add_trace(go.Scatter(x=[s0 * 100], y=[m0 * 100], mode="markers", name="Your portfolio today",
                             marker=dict(size=15, symbol="star", color=NAVY), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=[s1 * 100], y=[m1 * 100], mode="markers", name=f"Your portfolio {POST_FIRE}",
                             marker=dict(size=12, symbol="diamond", color=ui.GOLD), hovertemplate=hover))
    fig.update_layout(xaxis_title="Volatility (%, real, yearly)", yaxis_title="Expected return (%, real, yearly)",
                      height=480, margin=dict(b=95), legend=dict(y=-0.24), xaxis_title_standoff=4)
    return fig


def curve_chart(rf: mk.RiskFree, h: int) -> go.Figure:
    """Interpolated Confederation curve with its nodes; the initial horizon is marked."""
    T, y = np.array(rf.yields).T
    grid = np.linspace(T[0], T[-1], 200)
    hover = "%{x:.1f} years<br>%{y:.3f}%<extra>%{fullData.name}</extra>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=[mk.nominal_rate(t, rf.yields) * 100 for t in grid], mode="lines",
                             name="Interpolated r(h)", line=dict(width=2.5, color=BLUE), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=T, y=y * 100, mode="markers", name="Nodes (spot rates)",
                             marker=dict(size=8, color=NAVY, line=ui.OUTLINE), hovertemplate=hover))
    fig.add_vline(x=float(np.clip(h, T[0], T[-1])), line=dict(color=NAVY, width=1, dash="dash"),
                  annotation_text=f"your horizon {h}", annotation_position="bottom right",
                  annotation_font=dict(color=GREY, size=11))
    fig.update_layout(xaxis_title="Maturity (years)", yaxis_title="Nominal rate (%)", height=280,
                      legend=dict(y=-0.3))
    return fig


def donut_chart(x: np.ndarray, tickers: List[str]) -> go.Figure:
    fig = go.Figure(go.Pie(labels=tickers, values=x, hole=0.62, sort=False, direction="clockwise",
                           marker=dict(colors=[COLORS.get(t) for t in tickers], line=ui.OUTLINE),
                           textinfo="label", textfont=dict(size=11),
                           customdata=[data.UNIVERSE[t] for t in tickers],
                           hovertemplate="%{customdata} (%{label})<br>%{value:.1%}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    return fig


# ----------------------------------------------------------------------------- tables
COURSE, PROJECT = "Course", "Project layer"
ASSUMPTIONS: List[tuple] = [
    ("Portfolio choice", "γ-problem and σ-problem with long-only weights, 40% cap and 10% gold floor", f"{COURSE} (LN 1)"),
    ("Estimation error", "Resampled frontier (Michaud 2007)", f"{COURSE} (LN 3, slides 11-12)"),
    ("Shrinkage", "Shown only as a comparison", f"{COURSE} (LN 3)"),
    ("Yield curve", "Interpolation linear in the discount factor", f"{COURSE} (Fixed Income and Credit Risk)"),
    ("Spot rates", "The SNB publishes spot rates, so no par-yield conversion is needed", PROJECT),
    ("USD 1-year rate", "Treasury yield from FRED (DGS1), used as the 1-year spot rate", PROJECT),
    ("Monte Carlo on wealth", "Outside the slides; lognormal yearly returns, point estimates treated as known", PROJECT),
    ("Dynamic γ_t", "Human capital rule computed on each path from its own wealth", PROJECT),
    ("Hedging", "1-year forward rolled monthly, no cross-currency basis, 100% of the initial value", PROJECT),
    ("Remaining currency risk", "Only USD/CHF is hedged; EEM keeps its local currencies against the dollar", PROJECT),
    ("Franc drift and hedge cost", "μ contains the average of the sample, hedged or unhedged, not today's value", PROJECT),
    ("Inflation", "Returns deflated with the Swiss CPI; TIPS cover US, not Swiss, inflation", PROJECT),
    ("Expected inflation", "Used only in the real risk-free rate", PROJECT),
    ("Cash flows", "Savings, K and spending constant in today's CHF; savings at the end of the year", PROJECT),
    ("After FIRE", "γ = γ_min; withdrawals are not simulated", PROJECT),
    ("Swiss equities", "SMICHA.SW, SMI total-return index before 2013", PROJECT),
    ("Costs", "No transaction costs and no taxes", PROJECT),
]

PCT1 = "{:.1%}".format


def allocation_table(x: np.ndarray, wealth: float, tickers: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"ETF": tickers, "Asset class": [data.UNIVERSE[t] for t in tickers],
                         "Weight": list(x), "Amount": [w * wealth for w in x]})


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


def tab_profile(profile: fire.ClientProfile, sim: fire.WealthSimulation) -> None:
    s, path = st.session_state, sim.path
    K = fire.fire_target(profile.expenses, profile.swr)
    at_target = path.loc[path["age"] == profile.target_age].iloc[0]

    st.subheader("Client profile", anchor=False)
    for title, text, _ in profile_rows(s, float(path["s"].iloc[0])):
        c1, c2 = st.columns([2, 6])
        c1.markdown(f'<div class="fp-eyebrow" style="margin-top:0.2rem">{title}</div>', unsafe_allow_html=True)
        c2.write(text)
    hedge_toggle()
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
               f"A path reaches the goal if its wealth is at least the FIRE capital K at {profile.target_age}. "
               f"Average of the simulated paths at {profile.target_age}: {chf(at_target['W'])}. After "
               f"{profile.target_age} the chart shows what happens if you keep saving the same amount, with the "
               f"{POST_FIRE} portfolio; spending your wealth in retirement is not simulated.")


def tab_probability(profile: fire.ClientProfile, sim: fire.WealthSimulation, grid: fire.ResampledGrid,
                    summary: Dict) -> None:
    ages, path = summary["ages"], sim.path
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Chance of FIRE by {profile.target_age}", pct(summary["p_target"]))
    for col, lv in zip((c2, c3, c4), fire.PROB_LEVELS):
        col.metric(f"Age with {lv:.0%} chance", str(ages[lv]) if ages[lv] else
                   f"after {int(path['age'].iloc[-1])}")
    st.caption(mc_caption(summary, sim))
    st.divider()
    st.subheader("Chance of being financially independent", anchor=False)
    ui.show(prob_chart({"Your plan": path}, {"Your plan": profile.target_age}))
    st.caption("Each point is the share of simulated paths where your wealth is at least the FIRE capital at "
               "that age, if you keep saving the same amount (in today's CHF) every year.")
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
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        ui.show_table(allocation_table(x_today, profile.wealth, tickers), {"Weight": PCT1, "Amount": chf})
        st.caption(f"Expected return {pct(path['m'].iloc[0], 2)} (real) and volatility "
                   f"{pct(path['s'].iloc[0], 2)} (real) per year, i.e. after inflation.")
    with c2:
        ui.show(donut_chart(x_today, tickers))
    st.divider()

    st.subheader("How the portfolio changes over time", anchor=False)
    st.caption("Press Play to follow the target weights year by year, or drag the slider to a given age.")
    ui.show(glide_chart(W))
    st.caption("While most of your future savings are still ahead, the portfolio takes more risk. "
               "As your wealth grows towards the FIRE capital it becomes more defensive. The actual portfolio "
               "adapts to your wealth each year; the charts show the average over the simulated paths. "
               f"From age {profile.target_age} on you hold the {POST_FIRE} portfolio.")
    ui.show(risk_chart(path, profile.target_age))
    st.divider()

    if len(path) > 1:
        row = path.iloc[1]
        st.subheader(f"Rebalancing after the first year (age {int(row['age'])})", anchor=False)
        ui.show_table(trade_table(row, tickers),
                      {"Weight after one year of market moves": PCT1, "New target weight": PCT1,
                       "Change (pp)": "{:+.1f}".format, "Amount": chf})
        st.caption(f"Trades bring the portfolio back to its new target and invest this year's savings "
                   f"({chf(profile.savings)}); amounts in today's CHF, average of the simulated paths, no transaction costs.")
        st.divider()

    st.subheader("Where your portfolio sits among all possible portfolios", anchor=False)
    ui.show(frontier_chart(model, path, x_today))


def what_if(profile: fire.ClientProfile, sim: fire.WealthSimulation, grid: fire.ResampledGrid,
            summary: Dict) -> None:
    st.subheader("What if", anchor=False)
    st.caption("Change one or two levers and compare with your current plan.")
    base_rate = profile.savings / profile.income if profile.income > 0 else 0.0
    min_age = min(profile.age + 1, 80)
    st.session_state.setdefault("wi_rate", int(round(min(max(base_rate, 0.0), 0.9) * 100)))
    st.session_state.setdefault("wi_age", profile.target_age)
    st.session_state["wi_age"] = int(np.clip(st.session_state["wi_age"], min_age, 80))
    c1, c2 = st.columns(2)
    alt_rate = c1.slider("Savings rate (%)", 0, 90, key="wi_rate",
                         help="Spending less also lowers the capital you need.") / 100
    alt_age = c2.slider("Target FIRE age", min_age, 80, key="wi_age") if min_age < 80 else 80
    alt = profile.with_changes(expenses=max(profile.income * (1 - alt_rate), 1.0), target_age=alt_age)
    alt_path = run_simulation(alt, grid).path
    alt_summary = fire.fire_summary(alt_path, alt.target_age)
    ui.show(prob_chart({"Current plan": sim.path, "Alternative": alt_path},
                       {"Current plan": profile.target_age, "Alternative": alt.target_age}))

    def col(p: fire.ClientProfile, s: Dict) -> List[str]:
        a = s["ages"]
        return [pct(p.savings / p.income if p.income else 0, 0), chf(p.expenses),
                chf(fire.fire_target(p.expenses, p.swr)), str(p.target_age), pct(s["p_target"]),
                f"±{s['se_target'] * 100:.2f} pp"] + \
               [str(a[lv]) if a[lv] else "-" for lv in fire.PROB_LEVELS]

    idx = ["Savings rate", "Yearly expenses", "FIRE capital", "Target age", "Chance by target age",
           "Standard error"] + \
          [f"Age with {lv:.0%} chance" for lv in fire.PROB_LEVELS]
    st.table(pd.DataFrame({"Current plan": col(profile, summary), "Alternative": col(alt, alt_summary)},
                          index=idx))


def tab_technical(model: Dict, profile: fire.ClientProfile, sim: fire.WealthSimulation, gb: Dict) -> None:
    path, rf, m = sim.path, risk_free(), load_market()
    h0 = profile.target_age - profile.age
    tickers, mu, sigma = model["tickers"], model["mu"], model["sigma"]
    sig_minus, sig_plus, gamma_hi = model["bounds"]
    st.caption("Background for the interested reader; nothing here is needed to use the plan.")

    with st.expander("All assumptions: course slides and project layer"):
        st.markdown("| Topic | Assumption | Source |\n|:---|:---|:---|\n"
                    + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in ASSUMPTIONS))

    with st.expander("Data and estimates"):
        st.write(f"Monthly total returns (dividends reinvested) from Yahoo Finance, converted to CHF "
                 f"({'hedged' if model['hedged'] else 'unhedged'}) and deflated with the Swiss CPI, "
                 f"{model['start']:%b %Y} - {model['end']:%b %Y} (T = {model['T']} months, common sample). "
                 f"μ = 12 × monthly mean, Σ = 12 × monthly covariance of these real CHF returns.")
        vol = np.sqrt(np.diag(sigma))
        ui.show_table(pd.DataFrame({"Asset class": [data.UNIVERSE[t] for t in tickers],
                                    "Expected return": mu, "Volatility": vol}, index=tickers),
                      {"Expected return": PCT1, "Volatility": PCT1}, hide_index=False)
        ui.show_table(pd.DataFrame(sigma / np.outer(vol, vol), index=tickers, columns=tickers),
                      {t: "{:.2f}".format for t in tickers}, hide_index=False)
        st.markdown(
            "**Assumptions.** Savings c, FIRE capital K and spending after FIRE are constant in today's CHF. "
            "Savings are added at the end of each year. No transaction costs.  \n"
            "Swiss equities: UBS SMI ETF (SMICHA.SW). Yahoo has no dividends before 2013 for ETFs listed in "
            "Zurich, so before 2013 the SMI total-return index (SMIC.SW) is used.")

    with st.expander("Currency hedging and inflation (project assumptions, not from the course)"):
        st.latex(r"\text{unhedged: }R^{CHF}=(1+R^{USD})\frac{S_{t+1}}{S_t}-1,\qquad "
                 r"\text{hedged: }R^{CHF}=(1+R^{USD})\Big(\frac{1+r^{CHF}_{1y}}{1+r^{USD}_{1y}}\Big)^{1/12}-1")
        cur = model["currency"]
        st.markdown(
            f"S = CHF per USD (FRED DEXSZUS, to {m.usdchf.index[-1]:%d %b %Y}), on the return dates. "
            f"r_CHF = 1-year node of the SNB Confederation curve (month-end, to {m.curve_date:%b %Y}), r_USD = "
            f"1-year Treasury yield (FRED DGS1, to {m.usd_1y.index[-1]:%d %b %Y}), both known at the start of "
            f"the month. CHF-listed assets are never hedged.  \n"
            "**Hedging is an approximation.** It is a 1-year forward rolled every month (covered interest "
            "parity): the cross-currency basis is ignored and the hedge covers 100% of the initial value, not "
            "the gain or loss of the month.  \n"
            f"**Cost of the hedge.** It changed the yearly return of USD assets by {cur['carry_mean']:+.2%} on average "
            f"over the sample; at today's rates it is {cur['carry_last']:+.2%}. μ of the hedged returns contains the "
            "average, not today's value.  \n"
            f"**Drift of the franc in μ.** The dollar moved by {cur['fx_drift']:+.2%} a year against the franc over "
            "the sample. Unhedged estimates of μ for USD assets contain this past drift: they assume that it "
            "continues. A stronger (weaker) franc in the past lowers (raises) them.  \n"
            "**Currency risk that stays.** Only the USD/CHF leg is hedged. Emerging-market equities (EEM) are "
            "traded in USD but keep the risk of their local currencies against the dollar.  \n"
            "**Inflation.** CHF returns are deflated with the actual Swiss CPI "
            f"(SNB cube plkopr, to {m.cpi.index[-1].to_timestamp():%b %Y}): inflation-linked bonds, gold and real estate get "
            "their real characteristics from the data. TIPS protect against US inflation, not Swiss inflation.")

    with st.expander("Risk-free rate (Swiss Confederation curve)"):
        st.latex(r"Z_i=(1+y_i)^{-T_i},\quad Z(h)=Z_{i-1}\frac{\tau}{\Delta}+Z_i\frac{\Delta-\tau}{\Delta},\quad "
                 r"\Delta=T_i-T_{i-1},\ \tau=T_i-h,\quad r(h)=Z(h)^{-1/h}-1,\quad "
                 r"r_f(h)=\frac{1+r(h)}{1+\pi}-1")
        show_risk_free(h0)
        ui.show(curve_chart(rf, h0))
        st.caption(f"SNB spot rates of Confederation bonds (cube {mk.SNB_CURVE_CUBE}, month-end value of "
                   f"{m.curve_date:%B %Y}). Horizons are limited to 1-30 years. The rate is used only to discount "
                   f"future savings (H_t), with the remaining horizon h_t of each year.")

    with st.expander("Portfolio optimisation (γ-problem, LN 1)"):
        st.latex(r"x^\star(\gamma)=\arg\min_x \tfrac12 x^\top\Sigma x-\gamma\,x^\top\mu \quad"
                 r"\text{u.c. } \mathbf 1^\top x=1,\; 0\le x_i\le 0.40,\; x_{GLD}\ge 0.10")
        st.write(f"γ ≥ 0 is the risk tolerance. γ = 0 is the constrained minimum-variance portfolio "
                 f"(σ⁻ = {pct(sig_minus, 2)}); the constrained frontier ends at σ⁺ = {pct(sig_plus, 2)} "
                 f"(reached at γ ≈ {gamma_hi:.3g}). No closed form because of the bounds: solved as a QP with scipy (minimize). "
                 f"Using excess returns μ − r_f·1 (LN 2/3) gives the same x* because 1ᵀx = 1, so r_f "
                 f"does not change the optimal weights.")
        st.write(f"σ-problem: γ such that the resampled portfolio at γ (see below), evaluated with the "
                 f"original μ̂ and Σ̂, has volatility σ*; solved with scipy (fsolve), so the portfolios "
                 f"actually held hit the risk targets exactly. "
                 f"σ_min = {pct(gb['sigma_min'], 2)} → γ_min = {gb['gamma_min']:.4f}; "
                 f"σ_max = {pct(gb['sigma_max'], 2)} → γ₀ = {gb['gamma_0']:.4f}.")

    with st.expander("Resampling (LN 3)"):
        st.markdown(
            f"Estimation error is handled with the resampling procedure of Michaud (2007), slides 11-12.  \n"
            f"**Step 1.** Draw T = {model['T']} monthly returns (the number of historical observations) from a "
            f"joint normal i.i.d. process with the same means, variances and covariances as the data; compute "
            f"the sampled mean vector μ_k and covariance matrix Σ_k (annualised).  \n"
            f"**Step 2.** Compute the constrained frontier (γ-problem) with μ_k and Σ_k on a grid of "
            f"{len(model['gammas'])} γ values, the same for all draws.  \n"
            f"**Step 3.** Repeat for K = {num(model['n_resamples'])} draws.  \n"
            f"**Step 4.** For each γ, average the weights across draws. Risk and return of the averaged "
            f"portfolios are evaluated with the original μ̂ and Σ̂.")
        st.caption("Weights at γₜ are interpolated linearly between grid points; averages and interpolations "
                   "of feasible portfolios satisfy the same linear constraints.")
        def vols(W: np.ndarray) -> List[float]:          # original Σ, also for the resampled portfolios
            return [opt.portfolio_stats(x, mu, sigma)[1] * 100 for x in W]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Constrained Markowitz composition map")
            ui.show(area_chart(pd.DataFrame(model["point"], columns=tickers), "Volatility (%)",
                               x=vols(model["point"])))
        with c2:
            st.markdown("Resampled composition map")
            ui.show(area_chart(pd.DataFrame(model["resampled"], columns=tickers), "Volatility (%)",
                               x=vols(model["resampled"])))

    with st.expander("Dynamic risk tolerance"):
        st.latex(r"H_t=c\,\frac{1-(1+r_f)^{-h_t}}{r_f},\qquad "
                 r"\gamma_t=\gamma_{\min}+(\gamma_0-\gamma_{\min})\frac{H_t}{W_t+H_t},\qquad "
                 r"\gamma_t=\gamma_{\min}\ \text{if}\ W_t\ge K")
        st.write("H_t is the value today of the savings still to come before the target date, discounted at the "
                 "real risk-free rate for the remaining horizon h_t (see above). At and after the target age "
                 "h_t ≤ 0, so H_t = 0 and γ_t = γ_min. Each year, on each simulated path: (W_t, h_t) → r_f → H_t "
                 "→ γ_t → resampled constrained portfolio → trades. The table shows averages over the simulated "
                 "paths.")
        ui.show_table(path[["age", "W", "rf", "H", "gamma", "m", "s"]].rename(
            columns={"W": "W_t", "rf": "r_f (real)", "H": "H_t", "gamma": "γ_t", "m": "μ_p (real)", "s": "σ_p (real)"}),
            {"age": "{:.0f}".format, "W_t": chf, "r_f (real)": "{:.2%}".format, "H_t": chf, "γ_t": "{:.4f}".format,
             "μ_p (real)": PCT1, "σ_p (real)": PCT1})

    with st.expander("Monte Carlo simulation of wealth (project assumption, not from the course)"):
        st.latex(r"W_{t+1}=W_t\,(1+x_t^\top R_t)+c,\qquad x_t=x(\gamma_t(W_t,h_t)),\qquad "
                 r"\hat p=\tfrac1N\sum_{k=1}^N \mathbf 1\{W_T^{(k)}\ge K\},\qquad "
                 r"SE=\sqrt{\hat p(1-\hat p)/N}")
        at = fire.fire_summary(path, profile.target_age)
        st.write(f"Wealth is simulated year by year on N = {num(sim.n_paths)} paths (seed {sim.seed}). "
                 f"γ_t is computed on each path from that path's own wealth, and the resampled weights, computed "
                 f"once, are read at γ_t. The portfolio is rebalanced to its target every year; the savings c "
                 f"enter at the end of the year. T is the target age: the probability is the share of paths with "
                 f"wealth of at least K at that age, not of having crossed K before. "
                 f"Here p̂ = {pct(at['p_target'], 2)}, standard error {at['se_target'] * 100:.2f} pp.")
        st.markdown("**Assumptions.** Yearly gross asset returns are jointly lognormal, with mean and covariance "
                    "matched exactly to the point estimates μ̂ and Σ̂ of the real CHF returns (not the resampled "
                    "ones), independent across years. The moments are treated as known: parameter uncertainty "
                    "enters only through the resampled weights. Savings are constant in real terms. From the target "
                    f"age on γ = γ_min ({POST_FIRE}); nothing is claimed about this portfolio preserving K, and "
                    "withdrawals are not simulated: later points assume that savings continue. No transaction "
                    "costs or taxes. The standard error measures simulation noise only, not model or "
                    "estimation error.")
        st.caption("The same seed gives the same simulated markets. The current plan and the What if alternative "
                   "share them (common random numbers), so their difference is not blurred by simulation noise.")

    with st.expander("Covariance shrinkage comparison (LN 3, Ledoit & Wolf 2003)"):
        delta = model["delta"]
        st.write(f"Σ̂ = δF + (1−δ)S with the constant-correlation target F; δ estimated from the data with the "
                 f"Ledoit-Wolf formula: **δ̂ = {delta:.3f}**. Comparison only: the plan uses the sample covariance "
                 f"with resampling. Note (Ledoit & Wolf, footnote 4): a constant-correlation target may be less "
                 f"appropriate when assets come from different asset classes, as here.")
        g_today = float(path["gamma"].iloc[0])
        rows = {}
        for name, S in (("Sample Σ", sigma), ("Shrunk Σ", model["shrunk"])):
            rows[f"Min. variance - {name}"] = opt.solve_gamma(mu, S, 0.0, tickers)
            rows[f"γ today = {g_today:.3f} - {name}"] = opt.solve_gamma(mu, S, g_today, tickers)
        ui.show_table(pd.DataFrame(rows, index=tickers).T, {t: PCT1 for t in tickers}, hide_index=False)


# ----------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    ui.header("When can you stop working?",
              f"A plan that invests your savings in {len(TICKERS)} ETFs "
              "and lowers its risk as you get closer to financial independence.")

    try:
        opt.check_feasibility(TICKERS)
    except ValueError as e:
        st.error(f"The investment universe cannot satisfy the portfolio rules: {e}")
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
    try:
        with st.spinner("Estimating portfolios. This takes about a minute on first run."):
            model = build_model(rets, bool(st.session_state["hedged"]), opt.CONFIG["n_resamples"], opt.CONFIG["seed"])
    except Exception as e:
        st.error(f"The portfolio optimisation failed ({e}). Please reload the page.")
        st.stop()

    model["currency"] = data.currency_facts(month_end, market.usdchf, market.usd_1y, market.chf_curve[1])

    risk_range(model)
    if st.session_state["step"] < RESULTS:
        questionnaire(model)
        return                    # not st.stop(): it would skip the clean-up that `field` relies on

    name = st.session_state["name"].strip()
    head, top1, top2 = st.columns([4, 1, 1], vertical_alignment="bottom")
    head.markdown(f'<div class="fp-eyebrow">Your plan</div>', unsafe_allow_html=True)
    if name:
        head.subheader(f"{name}, here is your plan", anchor=False)
    top1.button("Change my answers", on_click=go_to, args=(SUMMARY,), width="stretch")
    top2.button("Start over", on_click=go_to, args=(0,), width="stretch")

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

    t1, t2, t3, t4 = st.tabs(["Client profile", "Portfolio", "Probability of FIRE", "Technical details"])
    with t1:
        tab_profile(profile, sim)
    with t2:
        tab_portfolio(model, profile, sim.path)
    with t3:
        tab_probability(profile, sim, grid, summary)
    with t4:
        tab_technical(model, profile, sim, gb)


main()
