"""Smoke test of the Streamlit app (synthetic prices, no network)."""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data  # noqa: E402
import market as mk  # noqa: E402
import optimizer as opt  # noqa: E402

PREFS, ASSUMPTIONS, SUMMARY, RESULTS = 4, 5, 6, 7          # questionnaire steps of app_fire.py
with open(os.path.join(ROOT, "app_fire.py"), encoding="utf8") as _f:   # app_fire runs the app on import
    N_SAMPLE_PATHS = int(re.search(r"^N_SAMPLE_PATHS = (\d+)", _f.read(), re.M).group(1))


def fake_prices(tickers, start=None):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2008-01-01", "2024-12-31")
    mu = np.array([0.10, 0.08, 0.09, 0.08, 0.03, 0.04, 0.03, 0.07]) / 252
    vol = np.array([0.15, 0.14, 0.22, 0.20, 0.07, 0.08, 0.06, 0.17]) / np.sqrt(252)
    common = rng.normal(size=(len(days), 1))
    rets = mu + vol * (0.5 * common + np.sqrt(0.75) * rng.normal(size=(len(days), len(tickers))))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=list(tickers))


def offline(url):
    raise ConnectionError("no network")


@pytest.fixture
def app(monkeypatch):
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setattr(data, "download_prices", fake_prices)
    monkeypatch.setattr(mk, "_download", offline)            # local copies in data/
    monkeypatch.setitem(opt.CONFIG, "n_resamples", 5)
    return AppTest.from_file(os.path.join(ROOT, "app_fire.py"), default_timeout=120)


def charts(at):
    return " ".join(c.proto.spec for c in at.get("plotly_chart"))


def test_app_starts(app):
    app.run()
    assert not app.exception and not app.error


def test_results_show_probability_fan_chart_and_standard_error(app):
    app.session_state["step"] = RESULTS
    app.run()
    assert not app.exception and not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Chance of FIRE by 55"].endswith("%")
    captions = " ".join(c.value for c in app.caption)
    assert "Monte Carlo standard error" in captions and "10'000 simulated paths, seed 42" in captions
    assert "10th to 90th percentile" in charts(app) and "FIRE capital reached" in charts(app)
    lasts = metrics["Chance your money lasts until 95"]
    assert lasts.endswith("%") and 0 <= float(lasts.rstrip("%")) <= 100
    assert what_if_table(app).loc["Chance your money lasts until 95", "Current plan"] == lasts
    app.slider(key="wi_rate").set_value(60).run()                    # What if reacts
    assert not app.exception
    chance = what_if_table(app).loc["Chance by target age"]
    assert chance["Alternative"] != chance["Current plan"]
    lasts = what_if_table(app).loc["Chance your money lasts until 95"]
    assert float(lasts["Alternative"].rstrip("%")) > float(lasts["Current plan"].rstrip("%"))  # spends less


def traces(at):
    return [t for c in at.get("plotly_chart") for t in json.loads(c.proto.spec)["data"]]


def test_sample_paths_and_histogram(app):
    app.session_state["step"] = RESULTS
    app.run()
    names = [t.get("name") or "" for t in traces(app)]
    keys = [n for n in names if n.startswith("Sample paths that") and n.endswith(")")]
    assert len(keys) == 2
    assert sum(int(n.split("(")[1].rstrip(")")) for n in keys) == N_SAMPLE_PATHS   # every path is in one group
    chance = {m.label: m.value for m in app.metric}["Chance of FIRE by 55"]
    assert f"FIRE capital reached ({chance})" in names                         # same share as the probability
    spec = charts(app)
    assert "FIRE capital CHF" in spec and "target 55" in spec
    assert "today's money" not in spec


def test_fallback_warnings_are_shown(app):
    app.run()
    text = " ".join(w.value for w in app.warning)
    assert "using the local copy up to Jul 2025" in text and "DEXSZUS" in text


def set_hedging(at, value: bool):
    """Hedging is answered in the questionnaire (Assumptions step); then the plan is shown again."""
    at.session_state["step"] = ASSUMPTIONS
    at.run()
    at.toggle(key="_hedged").set_value(value).run()
    at.session_state["step"] = RESULTS
    at.run()
    assert not at.exception and not at.error


def test_hedging_is_a_questionnaire_answer_and_read_only_in_the_plan(app):
    app.session_state["step"] = RESULTS
    app.run()
    assert not app.get("toggle")                                     # read-only after submit
    assert any("fixed for this plan" in c.value for c in app.caption)

    def state():
        return allocation(app), {m.label: m.value for m in app.metric}
    x_h, metrics_h = state()
    set_hedging(app, False)
    x_u, metrics_u = state()
    assert not np.allclose(x_h.values, x_u.values)                   # new estimates, new portfolios
    assert metrics_h["Median wealth at 55"] != metrics_u["Median wealth at 55"]
    assert any(m.value == "no" for m in app.markdown)                # profile row: Currency hedging no
    set_hedging(app, True)
    assert state()[1] == metrics_h


def test_advanced_settings(app):
    app.session_state["step"] = ASSUMPTIONS
    app.run()
    assert {n.label for n in app.number_input} == {"Random seed"}    # number of paths is fixed
    rate = next(m for m in app.metric if m.label.startswith("Risk-free rate"))
    assert rate.label == "Risk-free rate for 15 years (real)"
    assert rate.value == f"{1.00545 / 1.01 - 1:.2%}"                 # 15J node of July 2025, inflation 1%
    app.checkbox(key="_rf_manual").set_value(True).run()
    app.number_input(key="_rf_flat").set_value(0.5).run()
    assert not app.exception
    assert any(m.label == "Risk-free rate used (real)" and m.value == "0.50%" for m in app.metric)


def test_every_questionnaire_step_renders(app):
    for step in range(RESULTS):
        app.session_state["step"] = step
        app.run()
        assert not app.exception


def test_widgets_are_restored_when_a_step_is_shown_again(app):
    """Regression: st.stop() after the questionnaire skipped Streamlit's clean-up of hidden widgets,
    so the browser showed default values (e.g. hedging off) instead of the stored answers."""
    app.session_state["step"] = ASSUMPTIONS
    app.run()
    app.slider(key="_swr").set_value(4.25).run()
    next(b for b in app.button if b.label == "Next").click().run()
    assert "_swr" not in app.session_state and "_hedged" not in app.session_state
    app.session_state["step"] = ASSUMPTIONS
    app.run()
    assert app.slider(key="_swr").value == 4.25 and app.toggle(key="_hedged").value is True


# ----------------------------------------------------------------------------- investment preferences
def what_if_table(at) -> pd.DataFrame:
    """The What if comparison (several st.table on the page: ui.show_table renders tables with it)."""
    return next(t.value for t in at.table if "Chance by target age" in t.value.index)


def allocation(at) -> pd.Series:
    df = next(t.value for t in at.table if "Weight" in t.value.columns and "Amount" in t.value.columns)
    return df["Weight"]                  # ui.show_table: st.table with the ETF column as row labels


def count_resampling(monkeypatch):
    calls = []
    real = opt.resample_weights
    monkeypatch.setattr(opt, "resample_weights", lambda *a, **k: calls.append(a[-1]) or real(*a, **k))
    return calls


def next_button(at):
    return next(b for b in at.button if b.label == "Next")


def test_no_preference_uses_default_rules(app):
    app.session_state["step"] = PREFS
    app.run()
    assert app.radio(key="_prefs").value == "No preference"
    assert any("At most 30% in any asset class, at most 15% in gold" in i.value for i in app.info)
    assert not next_button(app).disabled and not app.error
    app.session_state["step"] = RESULTS
    app.run()
    x = allocation(app)
    assert (x <= 0.30 + 1e-6).all() and x["GLD"] <= 0.15 + 1e-6 and abs(x.sum() - 1) < 1e-6
    text = " ".join(m.value for m in app.markdown)
    assert "Default rules" in text


def test_custom_table_excludes_and_forces_and_recomputes(app, monkeypatch):
    calls = count_resampling(monkeypatch)
    app.session_state["step"] = RESULTS
    app.run()
    default_x, n_default = allocation(app), len(calls)
    assert n_default == 1
    app.session_state["step"] = PREFS
    app.run()
    app.radio(key="_prefs").set_value("Yes").run()
    assert app.number_input(key="_hi_SPY").value == 30 and app.number_input(key="_hi_GLD").value == 15
    app.number_input(key="_hi_EEM").set_value(0.0).run()          # exclude
    app.number_input(key="_lo_IEF").set_value(20.0).run()         # forced minimum
    app.number_input(key="_hi_SPY").set_value(50.0).run()         # looser cap
    assert not app.error and len(calls) == n_default               # nothing computed while editing
    next_button(app).click().run()
    assert app.session_state["step"] == ASSUMPTIONS and len(calls) == n_default   # not before submit
    assert not app.exception and not app.error
    app.session_state["step"] = RESULTS
    app.run()
    assert len(calls) == n_default + 1                                          # new cache key
    assert not app.exception and not app.error
    x = allocation(app)
    assert x["EEM"] == 0.0 and x["IEF"] >= 0.20 - 1e-6 and abs(x.sum() - 1) < 1e-6
    assert not np.allclose(x.values, default_x.values)
    text = " ".join(m.value for m in app.markdown)
    assert "Your own rules" in text and "Emerging markets equities excluded" in text
    profile = " ".join(m.value for m in app.markdown if "own rules" in m.value)
    assert "Sovereign bonds (US Treasuries 7-10y) 20% - 30%" in profile
    # back to the defaults: the first cache entry is used again, nothing recomputed
    app.session_state["step"] = PREFS
    app.run()
    app.radio(key="_prefs").set_value("No preference").run()
    next_button(app).click().run()
    assert app.session_state["rules"] is None and len(calls) == n_default + 1


@pytest.mark.parametrize("key, value, msg", [
    ("_lo_SPY", 40.0, "US equities: the minimum weight (40%) is higher than the maximum (30%)"),
    ("_hi_GLD", 0.0, None),                                                      # still feasible
])
def test_infeasible_rules_block_the_run(app, monkeypatch, key, value, msg):
    calls = count_resampling(monkeypatch)
    app.session_state["step"] = PREFS
    app.run()
    app.radio(key="_prefs").set_value("Yes").run()
    app.number_input(key=key).set_value(value).run()
    errors = [e.value for e in app.error]
    if msg:
        assert any(msg in e for e in errors) and next_button(app).disabled
    else:
        assert not errors and not next_button(app).disabled
    assert calls == []                                                           # nothing before submit


def test_sum_of_maxima_below_100_is_blocked(app):
    app.session_state["step"] = PREFS
    app.run()
    app.radio(key="_prefs").set_value("Yes").run()
    for t in data.UNIVERSE:
        app.number_input(key=f"_hi_{t}").set_value(10.0).run()
    assert any("maximum weights add up to only 80%" in e.value for e in app.error)
    assert next_button(app).disabled


def test_infeasible_stored_rules_never_reach_the_optimiser(app, monkeypatch):
    calls = count_resampling(monkeypatch)
    n = len(data.UNIVERSE)
    app.session_state["rules"] = ((0.0,) * n, (0.1,) * n)
    app.session_state["step"] = RESULTS
    app.run()
    assert not app.exception and calls == []
    assert any("The investment rules cannot be met" in e.value for e in app.error)


# ----------------------------------------------------------------------------- expected return adjustment
def results(at) -> dict:
    """Probability of FIRE and today's expected return (Portfolio tab caption)."""
    cap = next(c.value for c in at.caption if c.value.startswith("Expected return "))
    return {"P": float({m.label: m.value for m in at.metric}["Chance of FIRE by 55"].rstrip("%")),
            "m": float(cap.split()[2].rstrip("%")), "caption": cap}


def set_adjustment(at, value):
    at.session_state["step"] = ASSUMPTIONS
    at.run()
    at.slider(key="_mu_adj").set_value(value).run()
    assert not at.exception and not at.error
    at.session_state["step"] = RESULTS
    at.run()
    assert not at.exception and not at.error


def test_historical_estimates_are_declared(app):
    app.session_state["step"] = RESULTS
    app.run()
    assert "(real, historical estimate)" in results(app)["caption"]
    assert "an average of past returns (Feb 2008 - Dec 2024)" in results(app)["caption"]   # sample of fake_prices


def test_adjustment_lowers_returns_and_probability_and_changes_the_cache_key(app, monkeypatch):
    calls = count_resampling(monkeypatch)
    app.session_state["step"] = ASSUMPTIONS
    app.run()
    assert app.slider(key="_mu_adj").value == 0.0                         # default: historical means
    caps = " ".join(c.value for c in app.caption)
    assert "By default (0) the expected returns are the historical means of" in caps and "optimistic" in caps
    app.session_state["step"] = RESULTS
    app.run()
    base, n, x0 = results(app), len(calls), allocation(app)
    set_adjustment(app, 2.0)
    low = results(app)
    assert len(calls) == n + 1                                             # new cache key
    assert low["m"] < base["m"] and low["P"] < base["P"]
    assert "historical estimate minus 2.0 pp" in low["caption"]
    np.testing.assert_allclose(allocation(app), x0, atol=1e-6)          # same shift for all: same weights
    set_adjustment(app, 0.0)
    assert results(app) == base and len(calls) == n + 1                    # 0: identical, cached


# ----------------------------------------------------------------------------- UI cleanup
def test_three_tabs_and_no_removed_charts(app):
    app.session_state["step"] = RESULTS
    app.run()
    assert [t.label for t in app.tabs] == ["Client profile", "Portfolio", "Probability of FIRE"]
    subheaders = [h.value for h in app.subheader]
    assert "How the portfolio changes over time" in subheaders
    assert any(h.startswith("Rebalancing after the first year") for h in subheaders)
    assert "What if" in subheaders
    for gone in ("Where your portfolio sits among all possible portfolios",
                 "Chance of being financially independent"):
        assert gone not in subheaders
    names = {t.get("name") for t in traces(app)}
    assert not names & {"Volatility (real, %)", "Constrained efficient frontier", "Your plan"}
    assert {"Current plan", "Alternative"} <= names                      # What if chart stays


def test_what_if_reset_restores_the_questionnaire_answers(app):
    app.session_state["step"] = RESULTS
    app.run()
    rate, age = app.slider(key="wi_rate").value, app.slider(key="wi_age").value
    assert (rate, age) == (43, 55)                                       # 65'000 / 150'000, target 55
    app.slider(key="wi_rate").set_value(70).run()
    app.slider(key="wi_age").set_value(60).run()
    changed = what_if_table(app).loc["Chance by target age", "Alternative"]
    next(b for b in app.button if b.label == "Reset to initial answers").click().run()
    assert not app.exception
    assert (app.slider(key="wi_rate").value, app.slider(key="wi_age").value) == (rate, age)
    chance = what_if_table(app).loc["Chance by target age"]
    assert chance["Alternative"] == chance["Current plan"] != changed


# ----------------------------------------------------------------------------- compute once at submit
def test_resampling_runs_once_at_submit_and_never_for_other_answers(app, monkeypatch):
    calls = count_resampling(monkeypatch)
    for step in range(RESULTS):                                          # the whole questionnaire
        app.session_state["step"] = step
        app.run()
        assert not app.exception
    assert calls == []
    next(b for b in app.button if b.label == "Show my plan").click().run()
    assert app.session_state["step"] == RESULTS and len(calls) == 1 and not app.exception
    p0 = what_if_table(app).loc["Chance by target age", "Current plan"]
    for key, value in (("wi_rate", 60), ("wi_age", 58)):                 # What if
        app.slider(key=key).set_value(value).run()
    next(b for b in app.button if b.label == "Reset to initial answers").click().run()
    assert len(calls) == 1
    # answers that do not enter the resampling: age, income, expenses, wealth, FIRE age, SWR,
    # inflation, seed, risk range
    for key, value in (("age", 35), ("income", 160_000), ("expenses", 70_000), ("wealth", 500_000),
                       ("target_age", 52), ("swr", 3.5), ("infl", 2.0), ("seed", 7), ("risk_manual", True)):
        app.session_state[key] = value
        if "_" + key in app.session_state:                               # widget shows the new answer
            del app.session_state["_" + key]
        app.run()
        assert not app.exception and not app.error
    assert len(calls) == 1
    assert what_if_table(app).loc["Chance by target age", "Current plan"] != p0   # the plan did change
    set_hedging(app, False)                                              # inputs of the resampling
    assert len(calls) == 2
    set_adjustment(app, 1.0)
    assert len(calls) == 3


def test_manual_risk_range(app, monkeypatch):
    calls = count_resampling(monkeypatch)
    app.session_state["step"] = ASSUMPTIONS
    app.run()
    assert not app.get("slider") or "_sig_min" not in [w.key for w in app.slider]   # default: no sliders
    app.checkbox(key="_risk_manual").set_value(True).run()
    lo, hi = app.slider(key="_sig_min").min, app.slider(key="_sig_max").max
    assert lo < hi and calls == []                                       # point range: no resampling
    app.slider(key="_sig_min").set_value(round(lo + 1.0, 2)).run()
    app.slider(key="_sig_max").set_value(round(lo + 2.0, 2)).run()
    app.slider(key="_sig_min").set_value(round(lo + 3.0, 2)).run()       # lowest above highest
    assert any("cannot be higher" in e.value for e in app.error)
    assert next(b for b in app.button if b.label == "Next").disabled
    app.slider(key="_sig_min").set_value(round(lo + 1.0, 2)).run()
    app.session_state["step"] = RESULTS
    app.run()
    assert not app.exception and len(calls) == 1
    row = next(m.value for m in app.markdown if "set manually" in m.value)
    assert f"minimum risk setting {lo + 1.0:.2f}%" in row and f"maximum risk setting {lo + 2.0:.2f}%" in row
