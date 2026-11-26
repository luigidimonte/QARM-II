"""Checks of the header facts and the font configuration (no network)."""
from __future__ import annotations

import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ui_style as ui  # noqa: E402


def test_fire_facts_have_title_and_text():
    assert [f["title"] for f in ui.FIRE_FACTS] == ["Where it started", "A blog boom", "Many flavours",
                                                   "Going mainstream", "Savings rate is the lever"]
    for fact in ui.FIRE_FACTS:
        assert set(fact) == {"title", "text"}
        assert isinstance(fact["title"], str) and fact["title"].strip()
        assert isinstance(fact["text"], str) and fact["text"].strip()


def test_rotator_contains_the_lead_and_every_fact():
    page = ui.rotator_html("<strong>FIRE</strong> intro")
    assert page.count('class="slide"') == page.count('class="dot"') == len(ui.FIRE_FACTS) + 1
    assert "<strong>FIRE</strong> intro" in page
    for fact in ui.FIRE_FACTS:
        assert html.escape(fact["title"]) in page and html.escape(fact["text"]) in page
    assert ui.FONT_URL in page and ui.FONT_STACK in page and "__" not in page.replace("__init__", "")


def test_theme_font_matches_font_stack():
    """.streamlit/config.toml (used by st.dataframe) must use the same fonts as ui_style.FONT_STACK."""
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), encoding="utf8") as f:
        line = next(x for x in f if x.startswith("font = "))
    families = [x.strip() for x in line.split("=", 1)[1].strip().strip('"').split(",")]
    first, url = families[0].split(":", 1)
    assert url == ui.FONT_URL
    stack = [x.strip().strip("'\"") for x in ui.FONT_STACK.split(",")]
    assert [first] + families[1:] == stack
