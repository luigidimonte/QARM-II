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
DEFAULTS: Dict = dict(PRESETS["Mid career (40)"], name="", swr=4.0, rf=1.0, infl=1.5)


# ----------------------------------------------------------------------------- formatting
def chf(x: float) -> str:
    return f"CHF {x:,.0f}".replace(",", "'")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def label(t: str) -> str:
    return f"{data.UNIVERSE[t]} ({t})"


# ----------------------------------------------------------------------------- cached model
@st.cache_data(ttl=86_400, show_spinner=False)
def load_returns(tickers: tuple) -> pd.DataFrame:
    """Fresh monthly returns (cached for one day)."""
    return data.monthly_returns(data.download_prices(list(tickers)))


@st.cache_data(show_spinner=False)
def build_model(rets: pd.DataFrame) -> Dict:
    """Estimates, constrained frontier, gamma grid and resampled weights (depends on data only)."""
    tickers = list(rets.columns)
    mu, sigma, T = data.estimate_moments(rets)
    mu_v, sig_v = mu.values, sigma.values
    bounds = opt.frontier_bounds(mu_v, sig_v, tickers)
    gammas = opt.gamma_grid(mu_v, sig_v, tickers, bounds=bounds)
    point = opt.frontier_weights(mu_v, sig_v, gammas, tickers)
    resampled = opt.resample_weights((mu / 12).values, (sigma / 12).values, T, gammas, tickers)
    shrunk, delta = data.shrunk_covariance(rets, sigma)
    return {"tickers": tickers, "mu": mu_v, "sigma": sig_v, "T": T, "bounds": bounds,
            "gammas": gammas, "point": point, "resampled": resampled,
            "shrunk": shrunk.values, "delta": delta,
            "start": rets.index[0], "end": rets.index[-1]}


@st.cache_data(show_spinner=False)
def gamma_bounds(_model: Dict, model_key: str, sigma_min: float, sigma_max: float) -> Dict:
    return fire.gamma_bounds_from_sigma(_model["mu"], _model["sigma"], _model["tickers"],
                                        _model["bounds"], sigma_min, sigma_max)


def run_path(model: Dict, profile: fire.ClientProfile, gb: Dict) -> pd.DataFrame:
    return fire.simulate_glide_path(profile, model["mu"], model["sigma"], model["tickers"],
                                    model["gammas"], model["resampled"], gb["gamma_0"], gb["gamma_min"])


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
    """Feasible volatility range and defaults; keeps stored values inside the range."""
    sig_minus, sig_plus, _ = model["bounds"]
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
                              swr=s["swr"] / 100, rf_nominal=s["rf"] / 100, inflation=s["infl"] / 100)


def profile_rows(s) -> List[tuple]:
    """(title, text, questionnaire step) for the answers given so far."""
    return [("Age", f"{s['age']} years, goal at {s['target_age']}", 1),
            ("Income and expenses", f"{chf(s['income'])} income, {chf(s['expenses'])} expenses "
                                    f"({chf(s['income'] - s['expenses'])} saved per year)", 2),
            ("Invested wealth", chf(s["wealth"]), 3),
            ("Assumptions", f"withdrawal rate {s['swr']:.2f}%, inflation {s['infl']:.2f}%, "
                            f"risk-free rate {s['rf']:.2f}%", 4),
            ("Risk range", f"volatility {s['sig_max']:.2f}% early on, falling to {s['sig_min']:.2f}% "
                           f"at financial independence (nominal, yearly)", 4)]


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
            field(st.slider, "Expected inflation (%)", "infl", 0.0, 5.0, step=0.25)
            field(st.slider, "Risk-free rate (%, nominal)", "rf", 0.0, 5.0, step=0.25)
            st.info(f"**Portfolio rules (fixed)**  \n"
                    f"At most {opt.CONFIG['max_weight']:.0%} in any single ETF.  \n"
                    f"At least {opt.CONFIG['min_gold']:.0%} in gold.  \n"
                    f"Fully invested, no short selling.")
            lo, hi, d_min, d_max = risk_range(model)
            with st.expander("Advanced: risk range"):
                st.caption(f"Feasible volatility with the portfolio rules: **{lo:.2f}% - {hi:.2f}%** "
                           f"(nominal, yearly).")
                field(st.slider, "Lowest risk (volatility at / after FIRE, %)", "sig_min", lo, hi, step=0.05)
                field(st.slider, "Highest risk (volatility early on, %)", "sig_max", lo, hi, step=0.05)
                st.button("Reset to defaults", on_click=reset_risk, args=(d_min, d_max))
            if s["sig_min"] > s["sig_max"]:
                st.error("The lowest risk cannot be higher than the highest risk.")
                ok = False
            elif fire.is_narrow_risk_range(s["sig_min"] / 100, s["sig_max"] / 100):
                st.warning(fire.NARROW_RANGE_NOTE)

        else:
            st.subheader(f"Summary{', ' + s['name'] if s['name'].strip() else ''}", anchor=False)
            for title, text, target_step in profile_rows(s):
                c1, c2, c3 = st.columns([2, 5, 1], vertical_alignment="center")
                c1.markdown(f"**{title}**")
                c2.write(text)
                c3.button("Edit", key=f"edit_{title}", on_click=go_to, args=(target_step,))
            errors, _ = fire.validate_profile(profile_from_state())
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
                      yaxis_range=[0, 102], height=420)
    return fig


def wealth_chart(path: pd.DataFrame, K: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path["age"], y=path["M1"], name="Expected wealth", mode="lines",
                             line=dict(width=2.5, color=BLUE), fill="tozeroy",
                             fillcolor="rgba(10, 61, 145, 0.06)",
                             hovertemplate="Age %{x}<br>CHF %{y:,.0f}<extra></extra>"))
    fig.add_hline(y=K, line=dict(color=NAVY, width=1, dash="dash"), annotation_text=f"FIRE capital {chf(K)}",
                  annotation_font=dict(color=GREY, size=11))
    fig.update_layout(xaxis_title="Age", yaxis_title="CHF (today's money)", height=380, showlegend=False)
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


def risk_chart(path: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path["age"], y=path["s"] * 100, name="Expected volatility (real, %)",
                             mode="lines", line=dict(width=2.5, color=BLUE), hovertemplate=HOVER_PCT))
    fig.add_trace(go.Scatter(x=path["age"], y=path["m"] * 100, name="Expected return (real, %)",
                             mode="lines", line=dict(width=2.5, dash="dash", color=STEEL),
                             hovertemplate=HOVER_PCT))
    fig.add_trace(go.Scatter(x=path["age"], y=path["gamma"], name="Risk tolerance γₜ (right axis)",
                             mode="lines", line=dict(width=1.5, color=ui.GOLD), yaxis="y2",
                             hovertemplate="Age %{x}<br>γ %{y:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="Age", yaxis_title="%", height=380,
                      yaxis2=dict(title=dict(text="γₜ", font=dict(color=GREY, size=12)), overlaying="y",
                                  side="right", showgrid=False, zeroline=False),
                      legend=dict(y=-0.25))
    return fig


def frontier_chart(model: Dict, path: pd.DataFrame, x_today: np.ndarray) -> go.Figure:
    mu, sigma = model["mu"], model["sigma"]

    def stats(W: np.ndarray):
        return zip(*[opt.portfolio_stats(x, mu, sigma) for x in W])

    hover = "Volatility %{x:.1f}%<br>Return %{y:.1f}%<extra>%{fullData.name}</extra>"
    m_pt, s_pt = stats(model["point"])
    m_rs, s_rs = stats(model["resampled"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(s_pt) * 100, y=np.array(m_pt) * 100, mode="lines",
                             name="Constrained efficient frontier", line=dict(color=BLUE, width=2.5),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=np.array(s_rs) * 100, y=np.array(m_rs) * 100, mode="lines",
                             name="Resampled portfolios", line=dict(dash="dash", color=STEEL, width=2.5),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=np.sqrt(np.diag(sigma)) * 100, y=mu * 100, mode="markers+text",
                             text=model["tickers"], textposition="top center", name="Single ETFs",
                             textfont=dict(color=GREY, size=11),
                             marker=dict(size=10, color=[COLORS.get(t) for t in model["tickers"]],
                                         line=dict(color=NAVY, width=0.5)),
                             hovertemplate="%{text}<br>Volatility %{x:.1f}%<br>Return %{y:.1f}%<extra></extra>"))
    m0, s0 = opt.portfolio_stats(x_today, mu, sigma)
    x_end = fire.weights_table(path, model["tickers"]).iloc[-1].values
    m1, s1 = opt.portfolio_stats(x_end, mu, sigma)
    fig.add_trace(go.Scatter(x=[s0 * 100], y=[m0 * 100], mode="markers", name="Your portfolio today",
                             marker=dict(size=15, symbol="star", color=NAVY), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=[s1 * 100], y=[m1 * 100], mode="markers", name="Your portfolio at the end",
                             marker=dict(size=12, symbol="diamond", color=ui.GOLD), hovertemplate=hover))
    fig.update_layout(xaxis_title="Volatility (%, nominal, yearly)", yaxis_title="Expected return (%, nominal, yearly)",
                      height=450)
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
def tab_profile(profile: fire.ClientProfile, path: pd.DataFrame) -> None:
    s = st.session_state
    K = fire.fire_target(profile.expenses, profile.swr)
    w_target = float(path.loc[path["age"] == profile.target_age, "M1"].iloc[0])

    st.subheader("Client profile", anchor=False)
    for title, text, _ in profile_rows(s):
        c1, c2 = st.columns([2, 6])
        c1.markdown(f'<div class="fp-eyebrow" style="margin-top:0.2rem">{title}</div>', unsafe_allow_html=True)
        c2.write(text)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital needed for FIRE", chf(K), help="Yearly expenses / safe withdrawal rate")
    c2.metric("Yearly savings", chf(profile.savings))
    c3.metric("Savings rate", pct(profile.savings / profile.income, 0) if profile.income > 0 else "-")
    c4.metric(f"Expected wealth at {profile.target_age}", chf(w_target),
              delta=chf(w_target - K) + " vs target")
    st.divider()

    st.subheader("Expected wealth", anchor=False)
    ui.show(wealth_chart(path, K))
    st.caption("Expected wealth in today's money, assuming the same yearly savings until the target age.")


def tab_probability(model: Dict, profile: fire.ClientProfile, path: pd.DataFrame, gb: Dict,
                    summary: Dict) -> None:
    ages = summary["ages"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Chance of FIRE by {profile.target_age}", pct(summary["p_target"], 0))
    for col, lv in zip((c2, c3, c4), fire.PROB_LEVELS):
        col.metric(f"Age with {lv:.0%} chance", str(ages[lv]) if ages[lv] else
                   f"after {int(path['age'].iloc[-1])}")
    st.divider()
    st.subheader("Chance of being financially independent", anchor=False)
    ui.show(prob_chart({"Your plan": path}, {"Your plan": profile.target_age}))
    st.caption("Each point is the chance that your wealth is at least the FIRE capital at that age, "
               "if you keep saving the same amount (in today's money) every year.")
    st.divider()
    what_if(model, profile, path, gb, summary)


def tab_portfolio(model: Dict, profile: fire.ClientProfile, path: pd.DataFrame) -> None:
    tickers = model["tickers"]
    W = fire.weights_table(path, tickers)
    x_today = W.iloc[0].values
    st.subheader("Your portfolio today", anchor=False)
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        ui.show_table(allocation_table(x_today, profile.wealth, tickers), {"Weight": PCT1, "Amount": chf})
        st.caption(f"Expected return {pct(path['m'].iloc[0])} and volatility {pct(path['s'].iloc[0])} "
                   f"per year, after inflation.")
    with c2:
        ui.show(donut_chart(x_today, tickers))
    st.divider()

    st.subheader("How the portfolio changes over time", anchor=False)
    st.caption("Press Play to follow the target weights year by year, or drag the slider to a given age.")
    ui.show(glide_chart(W))
    st.caption("While most of your future savings are still ahead, the portfolio takes more risk. "
               "As your wealth grows towards the FIRE capital it becomes more defensive.")
    ui.show(risk_chart(path))
    st.divider()

    if len(path) > 1:
        row = path.iloc[1]
        st.subheader(f"Rebalancing after the first year (age {int(row['age'])})", anchor=False)
        ui.show_table(trade_table(row, tickers),
                      {"Weight after one year of market moves": PCT1, "New target weight": PCT1,
                       "Change (pp)": "{:+.1f}".format, "Amount": chf})
        st.caption(f"Trades bring the portfolio back to its new target and invest this year's savings "
                   f"({chf(profile.savings)}); amounts in today's money, on the expected path, no transaction costs.")
        st.divider()

    st.subheader("Where your portfolio sits among all possible portfolios", anchor=False)
    ui.show(frontier_chart(model, path, x_today))


def what_if(model: Dict, profile: fire.ClientProfile, path: pd.DataFrame, gb: Dict, summary: Dict) -> None:
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
    alt_path = run_path(model, alt, gb)
    alt_summary = fire.fire_summary(alt_path, alt.target_age)
    ui.show(prob_chart({"Current plan": path, "Alternative": alt_path},
                       {"Current plan": profile.target_age, "Alternative": alt.target_age}))

    def col(p: fire.ClientProfile, s: Dict) -> List[str]:
        a = s["ages"]
        return [pct(p.savings / p.income if p.income else 0, 0), chf(p.expenses),
                chf(fire.fire_target(p.expenses, p.swr)), str(p.target_age), pct(s["p_target"], 0)] + \
               [str(a[lv]) if a[lv] else "-" for lv in fire.PROB_LEVELS]

    idx = ["Savings rate", "Yearly expenses", "FIRE capital", "Target age", "Chance by target age"] + \
          [f"Age with {lv:.0%} chance" for lv in fire.PROB_LEVELS]
    st.table(pd.DataFrame({"Current plan": col(profile, summary), "Alternative": col(alt, alt_summary)},
                          index=idx))


def tab_technical(model: Dict, profile: fire.ClientProfile, path: pd.DataFrame, gb: Dict) -> None:
    tickers, mu, sigma = model["tickers"], model["mu"], model["sigma"]
    sig_minus, sig_plus, gamma_hi = model["bounds"]
    st.caption("Background for the interested reader; nothing here is needed to use the plan.")

    with st.expander("Data and estimates"):
        st.write(f"Monthly total returns (dividends reinvested) from Yahoo Finance, "
                 f"{model['start']:%b %Y} - {model['end']:%b %Y} (T = {model['T']} months, common sample). "
                 f"μ = 12 × monthly mean, Σ = 12 × monthly covariance (nominal).")
        vol = np.sqrt(np.diag(sigma))
        ui.show_table(pd.DataFrame({"Asset class": [data.UNIVERSE[t] for t in tickers],
                                    "Expected return": mu, "Volatility": vol}, index=tickers),
                      {"Expected return": PCT1, "Volatility": PCT1}, hide_index=False)
        ui.show_table(pd.DataFrame(sigma / np.outer(vol, vol), index=tickers, columns=tickers),
                      {t: "{:.2f}".format for t in tickers}, hide_index=False)
        st.markdown("**Assumptions.** Single currency, FX ignored. Real terms with the exact Fisher relation "
                    "(1+R)/(1+π) − 1, also applied to r_f. Savings constant in real terms, added at the end of "
                    "each year. No transaction costs.")

    with st.expander("Portfolio optimisation (γ-problem, LN 1)"):
        st.latex(r"x^\star(\gamma)=\arg\min_x \tfrac12 x^\top\Sigma x-\gamma\,x^\top\mu \quad"
                 r"\text{u.c. } \mathbf 1^\top x=1,\; 0\le x_i\le 0.40,\; x_{GLD}\ge 0.10")
        st.write(f"γ ≥ 0 is the risk tolerance. γ = 0 is the constrained minimum-variance portfolio "
                 f"(σ⁻ = {pct(sig_minus, 2)}); the constrained frontier ends at σ⁺ = {pct(sig_plus, 2)} "
                 f"(reached at γ ≈ {gamma_hi:.3g}). No closed form because of the bounds: solved as a QP with cvxpy. "
                 f"Using excess returns μ − r_f·1 (LN 2/3) gives the same x* because 1ᵀx = 1, so r_f and inflation "
                 f"do not change the optimal weights.")
        st.write(f"σ-problem: γ such that σ(x*(γ)) = σ*, solved with Brent's method. "
                 f"σ_min = {pct(gb['sigma_min'], 2)} → γ_min = {gb['gamma_min']:.4f}; "
                 f"σ_max = {pct(gb['sigma_max'], 2)} → γ₀ = {gb['gamma_0']:.4f}.")

    with st.expander("Resampling (LN 3)"):
        st.write(f"Estimation error is handled with Michaud resampling: {opt.CONFIG['n_resamples']} samples of "
                 f"T = {model['T']} monthly returns drawn from N(μ̂, Σ̂), re-estimation, the same constrained "
                 f"γ-problem solved on a grid of {len(model['gammas'])} γ values (evenly spaced in volatility), "
                 f"weights averaged. Weights at γₜ are interpolated linearly between grid points; averages and "
                 f"interpolations of feasible portfolios satisfy the same linear constraints.")
        vols = [opt.portfolio_stats(x, mu, sigma)[1] * 100 for x in model["point"]]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Constrained Markowitz composition map")
            ui.show(area_chart(pd.DataFrame(model["point"], columns=tickers), "Volatility (%)", x=vols))
        with c2:
            st.markdown("Resampled composition map")
            ui.show(area_chart(pd.DataFrame(model["resampled"], columns=tickers), "Volatility (%)", x=vols))

    with st.expander("Dynamic risk tolerance"):
        st.latex(r"H_t=c\,\frac{1-(1+r_f)^{-h_t}}{r_f},\qquad "
                 r"\gamma_t=\gamma_{\min}+(\gamma_0-\gamma_{\min})\frac{H_t}{W_t+H_t},\qquad "
                 r"\gamma_t=\gamma_{\min}\ \text{if}\ W_t\ge K")
        st.write(f"H_t is the value today of the savings still to come before the target date (real r_f = "
                 f"{pct(profile.rf_real, 2)}). W_t is the expected wealth on the planned path. Each year: "
                 f"(W_t, h_t) → H_t → γ_t → resampled constrained portfolio → trades.")
        ui.show_table(path[["age", "W", "H", "gamma", "m", "s"]].rename(
            columns={"W": "W_t", "H": "H_t", "gamma": "γ_t", "m": "μ_p (real)", "s": "σ_p (real)"}),
            {"age": "{:.0f}".format, "W_t": chf, "H_t": chf, "γ_t": "{:.4f}".format,
             "μ_p (real)": PCT1, "σ_p (real)": PCT1})

    with st.expander("Probability of FIRE (project assumption, not from the course)"):
        st.latex(r"M_1'=M_1(1+m_t)+c,\quad M_2'=M_2\big((1+m_t)^2+s_t^2\big)+2c(1+m_t)M_1+c^2")
        st.latex(r"s^2=\ln\frac{M_2}{M_1^2},\quad m=\ln M_1-\tfrac{s^2}{2},\quad "
                 r"P(W_t\ge K)=1-\Phi\Big(\frac{\ln K-m}{s}\Big)")
        st.write("No Monte Carlo: the first two moments of wealth are propagated with the mean and variance of "
                 "the chosen portfolio each year (returns independent across years), then a lognormal is fitted. "
                 "The portfolio path itself is fixed by the expected wealth path. The curve is the probability of "
                 "being above K at each age, not of having crossed it before.")

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
    ui.header("FIRE Planner",
              "When can you stop depending on a salary? A plan that invests your savings in six ETFs "
              "and lowers its risk as you get closer to financial independence.")

    try:
        opt.check_feasibility(TICKERS)
    except ValueError as e:
        st.error(f"The investment universe cannot satisfy the portfolio rules: {e}")
        st.stop()
    try:
        with st.spinner("Retrieving market data"):
            rets = load_returns(tuple(TICKERS))
    except Exception as e:  # network / provider problems
        st.error(f"Market data could not be downloaded right now ({type(e).__name__}). Please try again in a minute.")
        st.stop()
    problems = data.validate_returns(rets, TICKERS)
    if problems:
        st.error("The market data is not usable: " + " ".join(problems))
        st.stop()
    try:
        with st.spinner("Estimating portfolios. This takes a few seconds on first run."):
            model = build_model(rets)
    except Exception as e:
        st.error(f"The portfolio optimisation failed ({e}). Please reload the page.")
        st.stop()

    risk_range(model)
    if st.session_state["step"] < RESULTS:
        questionnaire(model)
        st.stop()

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

    s = st.session_state
    key = f"{model['start']}-{model['end']}-{model['T']}"
    try:
        gb = gamma_bounds(model, key, s["sig_min"] / 100, s["sig_max"] / 100)
        path = run_path(model, profile, gb)
    except Exception as e:
        st.error(f"The plan could not be computed ({e}).")
        st.stop()
    for note in gb["notes"]:
        st.warning(note)
    summary = fire.fire_summary(path, profile.target_age)

    t1, t2, t3, t4 = st.tabs(["Client profile", "Portfolio", "Probability of FIRE", "Technical details"])
    with t1:
        tab_profile(profile, path)
    with t2:
        tab_portfolio(model, profile, path)
    with t3:
        tab_probability(model, profile, path, gb, summary)
    with t4:
        tab_technical(model, profile, path, gb)


main()
