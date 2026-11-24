"""Smoke test of the Streamlit app (synthetic prices, no network)."""
from __future__ import annotations

import json
import os
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
    app.session_state["step"] = 6
    app.run()
    assert not app.exception and not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Chance of FIRE by 55"].endswith("%")
    captions = " ".join(c.value for c in app.caption)
    assert "Monte Carlo standard error" in captions and "10'000 simulated paths, seed 42" in captions
    assert "5th to 95th percentile" in charts(app) and "FIRE capital reached" in charts(app)
    app.slider(key="wi_rate").set_value(60).run()                    # What if reacts
    assert not app.exception
    chance = app.table[0].value.loc["Chance by target age"]
    assert chance["Alternative"] != chance["Current plan"]


def traces(at):
    return [t for c in at.get("plotly_chart") for t in json.loads(c.proto.spec)["data"]]


def test_sample_paths_and_histogram(app):
    app.session_state["step"] = 6
    app.run()
    names = [t.get("name") or "" for t in traces(app)]
    keys = [n for n in names if n.startswith("Sample paths that") and n.endswith(")")]
    assert len(keys) == 2
    assert sum(int(n.split("(")[1].rstrip(")")) for n in keys) == 100          # every path is in one group
    chance = {m.label: m.value for m in app.metric}["Chance of FIRE by 55"]
    assert f"FIRE capital reached ({chance})" in names                         # same share as the probability
    spec = charts(app)
    assert "FIRE capital CHF" in spec and "target 55" in spec and "post-FIRE, minimum risk" in spec
    assert "today's money" not in spec


def test_technical_details_list_all_assumptions(app):
    app.session_state["step"] = 6
    app.run()
    text = next(m.value for m in app.markdown if m.value.startswith("| Topic"))
    for word in ("Monte Carlo", "forward", "EEM", "Franc drift", "TIPS", "spot rates", "FRED", "No transaction costs"):
        assert word in text
    body = " ".join(m.value for m in app.markdown)
    assert "Cost of the hedge" in body and "Drift of the franc" in body


def test_fallback_warnings_are_shown(app):
    app.run()
    text = " ".join(w.value for w in app.warning)
    assert "using the local copy up to Jul 2025" in text and "DEXSZUS" in text


def test_hedging_toggle_changes_the_estimates(app):
    app.session_state["step"] = 6
    app.run()
    assert app.toggle(key="_hedged").value is True                   # default: hedged

    def state():
        mu = next(d.value for d in app.dataframe if "Expected return" in d.value.columns)["Expected return"]
        return mu, {m.label: m.value for m in app.metric}, app.session_state["sig_max"]
    mu_h, metrics_h, sig_h = state()
    app.toggle(key="_hedged").set_value(False).run()
    assert not app.exception and not app.error
    mu_u, metrics_u, sig_u = state()
    assert np.abs(mu_h - mu_u).drop("SMICHA.SW").min() > 1e-4        # every USD asset
    assert mu_h["SMICHA.SW"] == mu_u["SMICHA.SW"]                    # CHF asset never hedged
    assert sig_h != sig_u and metrics_h["Median wealth at 55"] != metrics_u["Median wealth at 55"]
    assert any("unhedged" in m.value for m in app.markdown)
    app.toggle(key="_hedged").set_value(True).run()
    assert state()[1] == metrics_h


def test_advanced_settings(app):
    app.session_state["step"] = 4
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
    for step in range(6):
        app.session_state["step"] = step
        app.run()
        assert not app.exception


def test_widgets_are_restored_when_a_step_is_shown_again(app):
    """Regression: st.stop() after the questionnaire skipped Streamlit's clean-up of hidden widgets,
    so the browser showed default values (e.g. hedging off) instead of the stored answers."""
    app.session_state["step"] = 4
    app.run()
    app.slider(key="_swr").set_value(4.25).run()
    next(b for b in app.button if b.label == "Next").click().run()
    assert "_swr" not in app.session_state and "_hedged" not in app.session_state
    app.session_state["step"] = 4
    app.run()
    assert app.slider(key="_swr").value == 4.25 and app.toggle(key="_hedged").value is True
