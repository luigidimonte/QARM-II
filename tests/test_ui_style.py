"""Checks of the header facts and the font configuration (no network)."""
from __future__ import annotations

import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ui_style as ui  # noqa: E402

CM_FAMILY = "'Computer Modern Serif'"                                    # header (rotator) and theme font
CM_REPO = "https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@"
CM_CSS = "/font/Serif/cmun-serif.css"


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
    # the header is set in Computer Modern (its own stylesheet link), not in the body font FONT_STACK
    assert CM_FAMILY in page and CM_CSS in page and "__" not in page.replace("__init__", "")


def test_theme_font_matches_font_stack():
    """.streamlit/config.toml (theme font, used where the CSS does not reach, e.g. st.dataframe) loads
    Computer Modern from the same jsDelivr stylesheet as the header and the CSS, and the body font
    FONT_STACK / FONT_URL that the CSS loads are both defined."""
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), encoding="utf8") as f:
        line = next(x for x in f if x.startswith("font = "))
    families = [x.strip() for x in line.split("=", 1)[1].strip().strip('"').split(",")]
    first, url = families[0].split(":", 1)
    assert first == CM_FAMILY.strip("'")
    assert url.startswith(CM_REPO) and url.endswith(CM_CSS)
    assert families[-1] == "serif"
    assert f"@import url('{ui.FONT_URL}')" in ui.CSS and f"font-family: {ui.FONT_STACK}" in ui.CSS
    assert CM_REPO in ui.CSS and CM_CSS in ui.CSS
