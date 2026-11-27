"""Presentation layer for app_fire.py: CSS, colour palette, Plotly template and table helpers.
Nothing in here computes anything; it only decides how results look.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import html
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
# ----------------------------------------------------------------------------- palette (single source)

NAVY = "#111111" # editorial black: primary text and controls
BLUE = "#7897C5" # restrained institutional blue accent
STEEL = "#6A7078" # muted steel grey
MIST = "#F4F4F6" # pale card surface
LINE = "#DADCE0" # soft borders and gridlines
GREY = "#73777D" # secondary captions and labels
GOLD = "#A88A4B" # muted gold accent (chart colorway)
RED = "#111111" # retained API constant; monochrome visual treatment
FAN_OUTER = "rgba(17, 17, 17, 0.07)"
FAN_INNER = "rgba(17, 17, 17, 0.16)"
REACHED = "#245A49" # forest green for paths reaching FIRE
MISSED = "#A24735" # muted red for paths missing FIRE
# header band: change the burgundy here, in one place
ACCENT_BURGUNDY = "#7A1F2B" # header background; white text on it: 10.2:1 contrast (WCAG AA)
HERO_TEXT = "#FFFFFF" # title and "+" box on the burgundy band
HERO_LEAD = "#F3E6E8"
HERO_ACCENT_LINE = "#8BBCE0" # clear alpine sky blue accent # muted alpine/slate blue hairline separating header from page
SLATE_BLUE = "#8BBCE0" # clear alpine sky blue for progress bar and interactive controls
SWISS_RED = "#D52B1E" # official Swiss flag red # intro paragraph: 8.4:1 on the burgundy
TABLE_ROW = "#FAFBFC" # table body rows (as before, with st.dataframe)
TABLE_GRID = "#E2E4E7" # lines between table cells
TABLE_BORDER = "#E2E4E7" # outer table border
TABLE_RADIUS = "0.5rem" # rounded table corners; they also clip the burgundy header row
BUTTON_RADIUS = "999px" # corner radius of every button: "999px" = pill, e.g. "10px" for softer corners
# one hue per asset

ASSET_COLORS: Dict[str, str] = {
    "SPY": "#1D3557",        # US equities: deep institutional navy
    "SMICHA.SW": "#B82534",  # Swiss equities: Swiss crimson red
    "EEM": "#2A7B88",        # Emerging markets: muted teal
    "VNQ": "#C46D4E",        # US real estate: refined terracotta
    "IEF": "#2D6A4F",        # Sovereign bonds: forest green
    "LQD": "#6FA8DC",        # Corporate bonds: alpine blue
    "TIP": "#7A6382",        # Inflation-linked bonds: muted plum
    "GLD": "#D9A726",        # Gold: luminous refined gold
}
OUTLINE = dict(color="#FFFFFF", width=1)
# Computer Modern (LaTeX look) for the whole app, in one place. Web font: Computer Modern Unicode
# Serif (SIL Open Font License) from jsDelivr, pinned to a commit of dreampulse/computer-modern-web-font.
# Only regular, bold, italic and bold italic exist: weights up to 500 render as 400, 600+ as bold.
# .streamlit/config.toml repeats this stack for st.dataframe, which draws on a canvas (CSS cannot reach
# it); tests/test_ui_style.py checks that the two stay the same.
FONT_URL = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
            "&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap")
FONT_STACK = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_SERIF = "'Source Serif 4', Georgia, 'Times New Roman', serif"
# ----------------------------------------------------------------------------- Plotly template

pio.templates["fire"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT_STACK, size=14, color=NAVY),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    colorway=[BLUE, STEEL, NAVY, "#9CA5B0", GOLD, MIST],
    xaxis=dict(automargin=True, showgrid=False, zeroline=False, showline=True, linecolor=LINE, linewidth=1,
            ticks="outside", tickcolor=LINE, ticklen=4,
            title=dict(font=dict(family=FONT_STACK, size=12, color=GREY))),
    yaxis=dict(automargin=True, showgrid=True, gridcolor="#ECEDEF", gridwidth=1, zeroline=False, showline=False,
            title=dict(font=dict(family=FONT_STACK, size=12, color=GREY))),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
            font=dict(family=FONT_STACK, size=12, color=NAVY), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(family=FONT_STACK, size=12, color=NAVY)),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="closest",
    separators=".'",
))
pio.templates.default = "fire"

def show(fig: go.Figure, **kwargs) -> None:
    """Render a figure with our own template (Streamlit's theme would override the palette)."""
    st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **kwargs)

# ----------------------------------------------------------------------------- tables

def show_table(df: pd.DataFrame, formats: Optional[Dict[str, Callable]] = None,
                hide_index: bool = True, align: Optional[str] = None) -> None:
    """Static table (st.table, HTML), so that it gets the table style of the CSS below like every
    other table: burgundy header row, black text (st.dataframe draws on a canvas CSS cannot style).
    Cells are aligned by type (text left, numbers right) unless `align` ("left"/"right") is given,
    e.g. for numbers already formatted as text."""
    formats = dict(formats or {})
    if hide_index:                       # st.table ignores Styler.hide: show the first column as the
        first = df.columns[0]            # row labels instead (header kept, no 0..n column)
        df = df.set_index(first)
        first_format = formats.pop(first, None)
    sty = df.style.format(formats, na_rep="-")
    if hide_index and first_format is not None:
        sty = sty.format_index(first_format, axis="index")
    if align:
        sty = sty.set_properties(**{"text-align": f"{align} !important"})   # beats Streamlit's inline style
    st.table(sty)

# ----------------------------------------------------------------------------- CSS

CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/font/Serif/cmun-serif.css');
@import url('{FONT_URL}');
:root {{
--ink: {NAVY};
--accent: {BLUE};
--slate: {STEEL};
--surface: {MIST};
--line: {LINE};
--muted: {GREY};
--radius: 0px;
--ease: cubic-bezier(0.22, 1, 0.36, 1);
--shadow-hover: 0 12px 30px rgba(17, 17, 17, 0.075);
}}
@keyframes fpFadeUp {{
from {{ opacity: 0; transform: translateY(12px); }}
to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fpLineIn {{
from {{ transform: scaleX(0); }}
to {{ transform: scaleX(1); }}
}}
html, body, [class*="st-"], .stMarkdown, .stText, button, input, textarea, select, label {{
font-family: {FONT_STACK};
-webkit-font-smoothing: antialiased;
}}
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: #FFFFFF; color: var(--ink); overflow-x: hidden; }}
/* Streamlit chrome */
header[data-testid="stHeader"] {{ background: {ACCENT_BURGUNDY} !important; }}
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] span,
header[data-testid="stHeader"] svg {{ color: {HERO_TEXT} !important; fill: {HERO_TEXT} !important; }}
header[data-testid="stHeader"] button:hover {{ background: rgba(255,255,255,0.12) !important; }}
/* Layout: wide, editorial, with generous white space */
[data-testid="stMainBlockContainer"], .block-container {{
max-width: 1080px;
padding: 0 2.5rem 5rem 2.5rem;
}}
@media (max-width: 720px) {{
[data-testid="stMainBlockContainer"], .block-container {{ padding: 0 1rem 3rem 1rem; }}
}}
/* Hero: keep the existing content, change only its presentation */
.st-key-fp_hero {{
background: {ACCENT_BURGUNDY};
width: 100vw;
position: relative;
left: 50%;
right: 50%;
margin-left: -50vw;
margin-right: -50vw;
padding: 3.6rem max(2.5rem, calc((100vw - 1080px) / 2 + 2.5rem)) 1.4rem;
gap: 0 !important;
margin-top: -1rem;
margin-bottom: 1.6rem;
overflow: hidden;
box-shadow: 0 -4rem 0 0 {ACCENT_BURGUNDY};
}}
.st-key-fp_hero::after {{
content: "";
position: absolute;
left: 0;
right: 0;
bottom: 0;
height: 4px;
background: {HERO_ACCENT_LINE};
transform-origin: left center;
animation: fpLineIn 700ms var(--ease) both;
}}
.st-key-fp_hero > div {{ animation: fpFadeUp 520ms var(--ease) both; }}
.st-key-fp_hero {{ width: 100vw !important; max-width: 100vw !important; box-sizing: border-box; }}
.st-key-fp_hero iframe {{ display: block; }}
/* the rotating-text iframe sizes itself to its tallest slide: let its wrapper follow */
.st-key-fp_hero [data-testid="stElementContainer"]:has(iframe) {{ height: auto !important; flex: 0 0 auto !important; }}
@media (max-width: 720px) {{
.st-key-fp_hero {{ padding: 3.2rem 1rem 1.2rem 1rem; }}
}}
.fp-hero-title-row {{
display: flex;
flex-wrap: wrap;
align-items: center;
justify-content: center;
text-align: center;
gap: 0.9rem;
margin-bottom: 1.15rem;
}}
.fp-hero-swiss-cross {{
display: inline-flex;
align-items: center;
justify-content: center;
flex-shrink: 0;
width: clamp(32px, 3.2vw, 44px);
height: clamp(32px, 3.2vw, 44px);
border: 1.5px solid {HERO_TEXT};
box-sizing: border-box;
line-height: 0;
}}
.fp-hero-swiss-cross svg {{
width: 100%;
height: 100%;
display: block;
}}
.st-key-fp_hero .fp-hero-title {{
font-family: 'Times New Roman', Times, Georgia, serif !important;
font-size: clamp(2.7rem, 4.6vw, 4.3rem) !important;
font-weight: 400 !important;
color: {HERO_TEXT} !important;
margin: 0 !important;
letter-spacing: -0.015em;
overflow-wrap: anywhere;
line-height: 1.02;
}}
.st-key-fp_quiz {{
width: 100%;
max-width: 820px;
margin: 0 auto;
}}
/* Typography */
h1, h2, h3, h1 *, h2 *, h3 * {{ color: var(--ink) !important; }}
h1, h2 {{ font-family: {FONT_SERIF} !important; }}
h2 {{
font-size: 2rem !important;
font-weight: 600 !important;
letter-spacing: -0.02em;
line-height: 1.2 !important;
}}
h3 {{
font-family: {FONT_SERIF} !important;
font-size: 1.45rem !important;
font-weight: 600 !important;
letter-spacing: -0.01em;
padding-top: 0.2rem !important;
}}
p, li {{ font-size: 1.02rem; line-height: 1.6; color: #1F1F1F; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: #52565C !important; font-size: 0.9rem; }}
[data-testid="stCaptionContainer"] p {{ font-size: 0.9rem !important; color: #52565C !important; }}
.fp-eyebrow {{
display: inline-block;
font-size: 0.8rem;
font-weight: 600;
letter-spacing: 0.1em;
text-transform: uppercase;
color: var(--muted);
margin-bottom: 0.45rem;
}}
/* Main content appears softly, without changing the DOM/content */
[data-testid="stMainBlockContainer"] > div > div > div:not(.st-key-fp_hero) {{
animation: fpFadeUp 480ms 80ms var(--ease) both;
}}
/* Cards / bordered Streamlit containers: pale grey, flat, institutional */
[data-testid="stVerticalBlockBorderWrapper"] {{
border: 0 !important;
border-radius: var(--radius) !important;
background: #F4F4F6;
box-shadow: none;
padding: 0.35rem;
transition: transform 220ms var(--ease), box-shadow 260ms var(--ease), background-color 220ms ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
transform: translateY(-2px);
box-shadow: var(--shadow-hover);
background: #F2F2F4;
}}
/* Metric cards stay quieter and more analytical */
[data-testid="stMetric"] {{
position: relative;
overflow: hidden;
background: #FFFFFF;
border: 1px solid var(--line);
border-radius: var(--radius);
padding: 0.9rem 0.95rem;
box-shadow: none;
transition: transform 180ms var(--ease), box-shadow 220ms var(--ease), border-color 180ms ease;
}}
[data-testid="stMetric"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 3px;
background: {SLATE_BLUE} !important;
transform: scaleX(0);
transform-origin: left;
transition: transform 240ms var(--ease);
}}
[data-testid="stMetric"]:hover {{
transform: translateY(-2px);
box-shadow: 0 9px 22px rgba(17,17,17,0.065);
border-color: {SLATE_BLUE};
}}
[data-testid="stMetric"]:hover::before {{ transform: scaleX(1); }}
[data-testid="stMetricLabel"] p {{
font-size: 0.9rem !important;
font-weight: 500;
letter-spacing: 0.01em;
color: var(--muted) !important;
}}
[data-testid="stMetricValue"] {{
color: var(--ink) !important;
font-family: {FONT_SERIF} !important;
font-weight: 600;
font-size: clamp(1.2rem, 1.5vw, 1.48rem) !important;
font-variant-numeric: lining-nums tabular-nums;
}}
/* Progress */
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] p {{
font-size: 0.96rem !important;
color: var(--ink) !important;
}}
[data-testid="stProgress"] [role="progressbar"]:has(> div:empty),
[data-testid="stProgress"] [role="progressbar"] div:has(> div:empty) {{
background: #ECEEF2 !important;
border-radius: 0 !important;
}}
[data-testid="stProgress"] [role="progressbar"] div:empty {{
background: {SLATE_BLUE} !important;
border-radius: 0 !important;
transition: width 500ms var(--ease);
}}
/* Buttons: restrained black/white system with subtle motion */
.stButton > button, [data-testid^="stBaseButton"] {{
border-radius: 0 !important;
min-height: 2.75rem;
padding-left: 1.1rem !important;
padding-right: 1.1rem !important;
font-weight: 500;
font-size: 1.08rem;
letter-spacing: 0;
transition: transform 150ms var(--ease), box-shadow 200ms var(--ease), background-color 180ms ease,
color 180ms ease, border-color 180ms ease;
}}
.stButton > button p, [data-testid^="stBaseButton"] p {{
font-size: 0.98rem !important;
}}
.stButton > button:hover, [data-testid^="stBaseButton"]:hover {{
transform: translateY(-1px);
}}
[data-testid="stBaseButton-primary"] {{
background: #FFFFFF !important;
border: 1.5px solid {ACCENT_BURGUNDY} !important;
color: var(--ink) !important;
font-weight: 600 !important;
}}
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] div,
[data-testid="stBaseButton-primary"] span {{
color: var(--ink) !important;
font-weight: 600 !important;
}}
[data-testid="stBaseButton-primary"]:hover {{
background: rgba(122, 31, 43, 0.06) !important;
border-color: {ACCENT_BURGUNDY} !important;
box-shadow: 0 6px 16px rgba(122, 31, 43, 0.08);
}}
[data-testid="stBaseButton-secondary"] {{
background: transparent !important;
border: 1px solid #CFCFD2 !important;
color: var(--ink) !important;
}}
/* Rounded buttons: only the corner radius changes (BUTTON_RADIUS). button[...] has the specificity of
   ".stButton > button" above and comes later, so it wins; toolbar and chart/table icons stay square. */
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-tertiary"],
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button,
button[data-testid^="stBaseButton-pills"],
button[data-testid^="stBaseButton-segmented_control"] {{ border-radius: {BUTTON_RADIUS} !important; }}
[data-testid="stBaseButton-secondary"]:hover {{
border-color: {SLATE_BLUE} !important;
background: #FFFFFF !important;
box-shadow: 0 6px 16px rgba(17,17,17,0.055);
}}
/* Inputs */
[data-testid="stWidgetLabel"] p {{
font-size: 0.98rem !important;
color: var(--ink) !important;
}}
[data-baseweb="input"], [data-baseweb="select"] > div {{
border-radius: 0 !important;
background: #FFFFFF !important;
}}
[data-baseweb="input"] input, [data-baseweb="select"] {{
font-size: 0.98rem !important;
}}
[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within {{
box-shadow: 0 0 0 1.5px {SLATE_BLUE} !important;
}}
input, textarea {{ caret-color: var(--accent); }}
/* Sliders / radio / checkbox accents */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
background: {SLATE_BLUE} !important;
border: 2px solid #FFFFFF !important;
box-shadow: 0 0 0 1px {SLATE_BLUE} !important;
transition: transform 120ms ease;
}}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover {{ transform: scale(1.08); }}
.stSlider [data-baseweb="slider"] > div > div > div {{ background: {SLATE_BLUE} !important; }}
/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{
border-bottom: 1px solid var(--line);
gap: 1.6rem;
}}
[data-testid="stTabs"] [role="tab"] p {{
font-size: 1rem;
font-weight: 500;
color: var(--muted);
transition: color 160ms ease;
}}
[data-testid="stTabs"] [role="tab"]:hover p {{ color: var(--ink); }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: var(--ink); font-weight: 600; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
background-color: {ACCENT_BURGUNDY} !important;
height: 2px;
}}
/* Dividers */
hr, [data-testid="stDivider"] {{ border-color: var(--line) !important; }}
/* Tables */
/* every table (ui.show_table and st.table): burgundy header row, black body text, light rows */
[data-testid="stTable"] {{
border: 1px solid {TABLE_BORDER};
border-radius: {TABLE_RADIUS};
overflow: hidden;
background: {TABLE_ROW};
}}
[data-testid="stTable"] table {{ border-collapse: separate; border-spacing: 0; width: 100%; border: none !important; }}
[data-testid="stTable"] th, [data-testid="stTable"] td {{
background: {TABLE_ROW} !important;
color: #000000 !important;
border: none !important;
border-right: 1px solid {TABLE_GRID} !important;
border-bottom: 1px solid {TABLE_GRID} !important;
padding: 0.7rem 0.85rem;
font-size: 0.95rem;
}}
[data-testid="stTable"] tr > :last-child {{ border-right: none !important; }}
[data-testid="stTable"] tbody tr:last-child > * {{ border-bottom: none !important; }}
[data-testid="stTable"] p {{ color: inherit !important; font-size: inherit !important; }}
[data-testid="stTable"] thead th {{
background: {ACCENT_BURGUNDY} !important;
color: {HERO_TEXT} !important;
text-align: left !important;
border-right-color: rgba(255,255,255,0.18) !important;
border-bottom: none !important;
}}
[data-testid="stTable"] td {{
font-family: {FONT_STACK};
font-size: 1.08rem;
font-variant-numeric: lining-nums tabular-nums;
}}
[data-testid="stTable"] th {{ font-weight: 500; }}
[data-testid="stTable"] tbody th {{ text-align: left !important; }}
[data-testid="stTable"] tbody tr > * {{ transition: background-color 140ms ease; }}
[data-testid="stTable"] tbody tr:hover > * {{ background: #F2F3F5 !important; }}
[data-testid="stDataFrame"] {{
border: 1px solid var(--line);
border-radius: 0;
overflow: hidden;
background: #FFFFFF;
}}
/* Expanders */
[data-testid="stExpander"] details {{
border: 1px solid var(--line) !important;
border-radius: 0 !important;
background: #FFFFFF;
transition: border-color 160ms ease, box-shadow 200ms ease;
}}
[data-testid="stExpander"] details:hover {{
border-color: #BFC2C7 !important;
box-shadow: 0 6px 16px rgba(17,17,17,0.04);
}}
[data-testid="stExpander"] summary p {{ font-size: 1rem; font-weight: 500; color: var(--ink); }}
/* Alerts */
[data-testid="stAlert"] {{ border-radius: 0 !important; }}
/* Accessibility */
@media (prefers-reduced-motion: reduce) {{
*, ::before, ::after {{
animation-duration: 0.001ms !important;
animation-iteration-count: 1 !important;
transition-duration: 0.001ms !important;
scroll-behavior: auto !important;
}}
}}
</style>
"""

def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- header with rotating facts

# Slides 2..N of the header (slide 1 is the intro paragraph). Edit the texts here, in one place.
FIRE_FACTS = [
    {"title": "Where it started",
     "text": "The ideas behind FIRE go back to the 1992 book “Your Money or Your Life” by Vicki Robin and "
             "Joe Dominguez. Dominguez had left a Wall Street career in his early thirties."},
    {"title": "A blog boom",
     "text": "Jacob Lund Fisker’s “Early Retirement Extreme” (2010) and the Mr. Money Mustache blog "
             "(started in 2011) helped turn the idea into an online movement."},
    {"title": "Many flavours",
     "text": "Lean FIRE (frugal), Fat FIRE (comfortable spending), Barista FIRE (part-time work covers part "
             "of the costs) and Coast FIRE (save early, then let it grow until the traditional retirement age)."},
    {"title": "Going mainstream",
     "text": "The movement reached a wide audience around 2018. In a late-2020 survey of US adults by Shopify "
             "and the Angus Reid Group, 9% of non-retired adults said they were actively pursuing FIRE and "
             "39% were considering it."},
    {"title": "Savings rate is the lever",
     "text": "FIRE followers often aim to save 50–70% of their income. Spending less both raises what you "
             "save and lowers the capital you need."},
]
ROTATE_SECONDS = 20

# Client-side carousel in its own iframe (st.components.v1.html), so it keeps rotating while Python
# computes. The slides are stacked in one grid cell: the widget is as tall as its longest slide at the
# current width and never changes height between slides; the (same-origin) iframe sets its own height.
_ROTATOR = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/font/Serif/cmun-serif.css">
<style>
html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
body { font-family: 'Computer Modern Serif', Georgia, serif; color: __LEAD__; -webkit-font-smoothing: antialiased; }
.rot { max-width: 60rem; margin: 0 auto; text-align: center; }
.slides { display: grid; }
.slide { grid-area: 1 / 1; margin: 0; font-size: 1.05rem; line-height: 1.62; opacity: 0; visibility: hidden;
         transition: opacity 400ms ease, visibility 0s linear 400ms; }
.slide.on { opacity: 1; visibility: visible; transition: opacity 400ms ease, visibility 0s; }
.slide strong, .title { color: __TEXT__; font-weight: 700; }
.slide em { font-style: italic; }
.title { display: block; margin-bottom: 0.4rem; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.88; }
.nav { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 0.1rem; margin-top: 0.75rem; }
button { font: inherit; color: __TEXT__; background: none; border: 0; padding: 0; margin: 0; cursor: pointer;
         opacity: 0.7; transition: opacity 150ms ease; }
button:hover, button:focus-visible, .dot.on { opacity: 1; }
button:focus-visible { outline: 1px solid __TEXT__; outline-offset: 2px; }
.arrow { width: 2rem; height: 2rem; font-size: 1.5rem; line-height: 1; }
.dot { width: 1.3rem; height: 2rem; display: inline-flex; align-items: center; justify-content: center; }
.dot::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: __TEXT__; opacity: 0.55; }
.dot.on::before { opacity: 1; transform: scale(1.25); }
@media (prefers-reduced-motion: reduce) { .slide, .slide.on { transition: none; } }
</style></head><body>
<div class="rot" id="rot" role="region" aria-roledescription="carousel" aria-label="About FIRE">
  <div class="slides">__SLIDES__</div>
  <div class="nav">
    <button class="arrow" id="prev" type="button" aria-label="Previous">&lsaquo;</button>__DOTS__
    <button class="arrow" id="next" type="button" aria-label="Next">&rsaquo;</button>
  </div>
</div>
<script>
(function () {
  var slides = document.querySelectorAll(".slide"), dots = document.querySelectorAll(".dot");
  var n = slides.length, DELAY = __DELAY__, TICK = 250, K_IDX = "fp_header_slide", K_LAST = "fp_header_last";
  function load(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
  function save(k, v) { try { sessionStorage.setItem(k, String(v)); } catch (e) {} }
  var i = parseInt(load(K_IDX), 10), last = parseInt(load(K_LAST), 10), now = Date.now();
  if (!(i >= 0 && i < n)) i = 0;
  if (!(last <= now && now - last < 2 * DELAY)) last = now;
  var hoverUntil = 0;
  function restart() { last = Date.now(); save(K_LAST, last); }
  function show(k) {
    i = (k + n) % n;
    for (var j = 0; j < n; j++) {
      slides[j].classList.toggle("on", j === i); slides[j].setAttribute("aria-hidden", j === i ? "false" : "true");
      dots[j].classList.toggle("on", j === i); dots[j].setAttribute("aria-current", j === i ? "true" : "false");
    }
    save(K_IDX, i);
  }
  function go(k) { show(k); restart(); }
  function tick() {
    var t = Date.now();
    if (t >= hoverUntil && t - last >= DELAY) go(i + 1);
    setTimeout(tick, TICK);
  }
  function hovering(e) { if (e.pointerType === "mouse") hoverUntil = Date.now() + DELAY; }
  function leave() { if (hoverUntil) { hoverUntil = 0; restart(); } }
  document.getElementById("prev").onclick = function () { go(i - 1); };
  document.getElementById("next").onclick = function () { go(i + 1); };
  for (var j = 0; j < n; j++) (function (k) { dots[k].onclick = function () { go(k); }; })(j);
  var rot = document.getElementById("rot");
  rot.addEventListener("pointerenter", hovering);
  rot.addEventListener("pointermove", hovering);
  rot.addEventListener("pointerleave", function (e) { if (e.pointerType === "mouse") leave(); });
  document.documentElement.addEventListener("mouseleave", leave);
  window.addEventListener("blur", leave);
  document.addEventListener("visibilitychange", function () { if (document.hidden) leave(); });
  function fit() {
    try {
      var f = window.frameElement, h = Math.ceil(rot.getBoundingClientRect().height) + 2;
      if (f && Math.abs(f.getBoundingClientRect().height - h) > 1) f.style.height = h + "px";
    } catch (e) {}
  }
  if (window.ResizeObserver) new ResizeObserver(fit).observe(rot);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
  window.addEventListener("resize", fit);
  show(i); save(K_LAST, last); fit();
  setTimeout(tick, TICK);
})();
</script></body></html>"""


def rotator_html(lead: str) -> str:
    """Header carousel: slide 1 = lead (HTML written in app_fire.py), then FIRE_FACTS (escaped)."""
    slides = [lead] + [f'<span class="title">{html.escape(f["title"])}</span>{html.escape(f["text"])}'
                       for f in FIRE_FACTS]
    body = "".join(f'<p class="slide" role="group" aria-label="{k + 1} of {len(slides)}">{s}</p>'
                   for k, s in enumerate(slides))
    dots = "".join(f'<button class="dot" type="button" aria-label="Show {k + 1} of {len(slides)}"></button>'
                   for k in range(len(slides)))
    out = _ROTATOR
    for key, val in (("__FONT_URL__", FONT_URL), ("__FONT__", FONT_STACK), ("__LEAD__", HERO_LEAD),
                     ("__TEXT__", HERO_TEXT), ("__DELAY__", json.dumps(ROTATE_SECONDS * 1000)),
                     ("__SLIDES__", body), ("__DOTS__", dots)):
        out = out.replace(key, val)
    return out


def header(title: str, lead: str) -> None:
    """Burgundy header: title, then the rotating intro paragraph / FIRE facts."""
    swiss_flag_svg = (
        f'<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Switzerland">'
        f'<rect width="32" height="32" fill="{SWISS_RED}"/>'
        f'<rect x="13" y="6" width="6" height="20" fill="{HERO_TEXT}"/>'
        f'<rect x="6" y="13" width="20" height="6" fill="{HERO_TEXT}"/>'
        f'</svg>'
    )
    with st.container(key="fp_hero"):
        st.markdown(
            f'<div class="fp-hero-title-row">'
            f'<div class="fp-hero-swiss-cross">{swiss_flag_svg}</div>'
            f'<div class="fp-hero-title">{title}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        components.html(rotator_html(lead), height=160)