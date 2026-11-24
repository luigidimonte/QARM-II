"""Tests for market.py (parsing, fallback, Confederation curve); no network."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market as mk  # noqa: E402

NODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
UP = tuple((float(T), 0.002 + 0.001 * T) for T in NODES)       # upward sloping
FLAT = tuple((float(T), 0.02) for T in NODES)


@pytest.mark.parametrize("T, y", UP)
def test_node_returns_its_own_yield(T, y):
    assert mk.nominal_rate(T, UP) == y
    assert mk.zero_coupon(T, UP) == pytest.approx((1 + y) ** -T)


@pytest.mark.parametrize("h, lo, hi", [(1.5, 1, 2), (6.3, 6, 7), (12, 10, 15), (17.5, 15, 20), (26, 20, 30)])
def test_between_nodes_lies_between_the_yields(h, lo, hi):
    y = dict(UP)
    assert y[lo] < mk.nominal_rate(h, UP) < y[hi]


def test_interpolation_by_hand():
    """tau = T_i - h: closer to the lower node -> more weight on Z_{i-1}."""
    Z10, Z15 = 1.012 ** -10, 1.017 ** -15
    Z = Z10 * 4 / 5 + Z15 * 1 / 5                               # h = 11: tau = 4, Delta = 5
    assert mk.zero_coupon(11, UP) == pytest.approx(Z)
    assert mk.nominal_rate(11, UP) == pytest.approx(Z ** (-1 / 11) - 1)


def test_clamping():
    assert mk.nominal_rate(0.2, UP) == mk.nominal_rate(1, UP)
    assert mk.nominal_rate(45, UP) == mk.nominal_rate(30, UP)


def test_flat_curve_gives_flat_rate_at_the_nodes_and_nearly_between():
    for h in np.linspace(1, 30, 59):
        # the interpolation is linear in Z, so between nodes a flat curve is flat only up to convexity
        assert mk.nominal_rate(h, FLAT) == pytest.approx(0.02, abs=3e-4)
    for T in NODES:
        assert mk.RiskFree(FLAT, 0.01).real(T) == pytest.approx(1.02 / 1.01 - 1)
    zero = tuple((float(T), 0.0) for T in NODES)
    for h in (1, 2.5, 12, 27, 40):
        assert mk.nominal_rate(h, zero) == pytest.approx(0.0, abs=1e-15)


def test_discount_factor_decreases_on_an_upward_curve():
    Z = [mk.zero_coupon(h, UP) for h in np.linspace(1, 30, 291)]
    assert np.all(np.diff(Z) < 0)


def test_real_rate_and_override():
    assert mk.RiskFree(UP, 0.01).real(7) == pytest.approx(1.009 / 1.01 - 1)
    assert mk.RiskFree(UP, 0.0).real(7) == pytest.approx(0.009)
    manual = mk.RiskFree(UP, 0.01, override=-0.002)
    assert manual.real(2) == manual.real(25) == -0.002


# ----------------------------------------------------------------------------- parsing and fallback
OLD = '"CubeId";"rendoblim"\r\n"PublishingDate";"x"\r\n\r\n"Date";"D0";"Value"\r\n"1996-12";"1J";"1.5"\r\n' \
      '"1996-12";"30J";\r\n"1997-12";"1J";"2"\r\n"1997-12";"30J";"4.445"\r\n'
NEW = '"CubeId";"rendeiduebm"\n"PublishingDate";"x"\n\n"Date";"D0";"D1";"Value"\n"1997-12";"CHF";"1J";"2"\n' \
      '"1997-12";"CHF";"30J";"4.445"\n"1997-12";"EUR";"30J";"9"\n'


def test_parse_both_snb_cubes():
    old, new = mk.parse_snb_curve(OLD), mk.parse_snb_curve(NEW)
    assert np.isnan(old.loc["1996-12", 30])
    for t in (old, new):
        assert t.loc["1997-12", 1] == pytest.approx(0.02) and t.loc["1997-12", 30] == pytest.approx(0.04445)


def test_parse_fred_fills_holidays():
    x = mk.parse_fred("observation_date,DGS1\n2020-01-01,\n2020-01-02,1.5\n2020-01-03,.\n2020-01-06,1.7\n", "DGS1")
    assert list(x) == [1.5, 1.5, 1.7]


def test_fallback_to_local_copies_with_warnings(monkeypatch):
    def fail(url):
        raise ConnectionError("no network")
    monkeypatch.setattr(mk, "_download", fail)
    m = mk.load_market()
    assert len(m.warnings) == 4 and all("local copy up to" in w for w in m.warnings)
    assert any("Jul 2025" in w for w in m.warnings)             # the date of the data is shown
    assert [T for T, _ in m.yields] == NODES
    assert m.usdchf.loc["2000-01-03"] == 1.5808
    assert m.usd_1y.between(0, 0.10).all() and (m.cpi > 0).all()


def test_no_warning_when_downloads_work(monkeypatch):
    files = {"DEXSZUS": "DEXSZUS.csv", "DGS1": "DGS1.csv", mk.SNB_CURVE_CUBE: "snb_rendoblim.csv",
             mk.SNB_CPI_CUBE: "snb_plkopr.csv"}

    def local(url):
        name = next(f for k, f in files.items() if k in url)
        with open(os.path.join(mk.CACHE_DIR, name), encoding="utf-8-sig") as f:
            return f.read()
    monkeypatch.setattr(mk, "_download", local)
    assert mk.load_market().warnings == ()


def test_empty_download_also_falls_back(monkeypatch):
    monkeypatch.setattr(mk, "_download", lambda url: "<html>maintenance</html>")
    assert len(mk.load_market().warnings) == 4
