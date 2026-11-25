"""Presentation layer for app_fire.py: CSS, colour palette, Plotly template and table helpers.

Nothing in here computes anything; it only decides how results look.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------------- palette (single source)
NAVY = "#0B192C"        # deep Swiss institutional navy: primary text, headers, hero
BLUE = "#1E3A5F"        # dark navy accent
STEEL = "#5C6B73"       # muted steel grey
MIST = "#F4F6F8"        # cool alabaster surface
LINE = "#D8DCE3"        # crisp thin borders and gridlines
GREY = "#64748B"        # secondary captions and labels
GOLD = "#C59B27"        # muted gold accent (γ line)
RED = "#C8102E"         # Swiss crimson red accent
FAN_OUTER = "rgba(11, 25, 44, 0.08)"
FAN_INNER = "rgba(11, 25, 44, 0.18)"
REACHED = "#1B5E4B"     # forest green for paths reaching FIRE
MISSED = "#B43B20"      # dark crimson for paths missing FIRE

# one hue per asset
ASSET_COLORS: Dict[str, str] = {
    "SPY": "#0B192C",
    "SMICHA.SW": "#C8102E",
    "EEM": "#5B21B6",
    "VNQ": "#BE185D",
    "IEF": "#1B5E4B",
    "LQD": "#0F766E",
    "TIP": "#4D7C0F",
    "GLD": "#C59B27",
}
OUTLINE = dict(color="#FFFFFF", width=1)

FONT_URL = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
            "&family=JetBrains+Mono:wght@500&display=swap")
SANS = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif'
SERIF = '"Times New Roman", Times, "Liberation Serif", Georgia, serif'
MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

# ----------------------------------------------------------------------------- Plotly template
pio.templates["fire"] = go.layout.Template(layout=go.Layout(
    font=dict(family=SANS, size=13, color=NAVY),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    colorway=[BLUE, STEEL, NAVY, "#8FA3BC", GOLD, MIST],
    xaxis=dict(automargin=True, showgrid=False, zeroline=False, showline=True, linecolor=LINE, linewidth=1,
               ticks="outside", tickcolor=LINE, ticklen=4,
               title=dict(font=dict(family=SANS, size=12, color=GREY))),
    yaxis=dict(automargin=True, showgrid=True, gridcolor="#ECEFF4", gridwidth=1, zeroline=False, showline=False,
               title=dict(font=dict(family=SANS, size=12, color=GREY))),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                font=dict(family=SANS, size=12, color=NAVY), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(family=SANS, size=12, color=NAVY)),
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
               hide_index: bool = True) -> None:
    sty = df.style.format(formats or {}, na_rep="-")
    st.dataframe(sty, hide_index=hide_index, width="stretch")


# ----------------------------------------------------------------------------- CSS
CSS = f"""
<style>
@import url('{FONT_URL}');
:root {{
  --navy: {NAVY}; --crimson: {RED}; --accent: {BLUE}; --slate: {STEEL}; --surface: {MIST}; --line: {LINE}; --muted: {GREY};
  --radius: 4px; --ease: cubic-bezier(0.22, 1, 0.36, 1);
  --shadow: 0 1px 3px rgba(11, 25, 44, 0.05);
  --shadow-hover: 0 6px 18px rgba(11, 25, 44, 0.08);
}}
html, body, [class*="st-"], .stMarkdown, .stText, button, input, textarea, select, label {{
  font-family: {SANS}; -webkit-font-smoothing: antialiased;
}}
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: #F8F9FB; color: var(--navy); overflow-x: hidden; }}

/* layout */
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1140px; padding: 0rem 2rem 4rem 2rem;
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: 0rem 1rem 3rem 1rem; }}
}}
header[data-testid="stHeader"] {{ background: var(--navy) !important; }}
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] span,
header[data-testid="stHeader"] svg {{ color: #E2E8F0 !important; fill: #E2E8F0 !important; }}

/* Full-bleed solid navy hero banner */
.fp-hero {{
  background-color: var(--navy);
  width: 100vw;
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  padding: 4rem 0 2.6rem 0;
  border-bottom: 3px solid var(--crimson);
  margin-bottom: 2rem;
}}
.fp-hero-inner {{
  max-width: 1140px;
  margin: 0 auto;
  padding: 0 2rem;
}}
@media (max-width: 640px) {{
  .fp-hero {{ padding: 3.5rem 0 2rem 0; }}
  .fp-hero-inner {{ padding: 0 1rem; }}
}}
.fp-hero-title-row {{
  display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1rem;
}}
.fp-hero-swiss-cross {{
  display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
  width: 34px; height: 34px; background-color: var(--crimson);
  color: #FFFFFF; font-family: {SANS}; font-weight: 700; font-size: 1.25rem; border-radius: 4px;
  line-height: 1;
}}
.fp-hero .fp-hero-title {{
  font-family: {SERIF} !important;
  font-size: 3.5rem !important;
  font-weight: 700 !important;
  color: #FFFFFF !important;
  margin: 0 !important;
  letter-spacing: -0.01em;
  line-height: 1.1;
}}
@media (max-width: 640px) {{
  .fp-hero .fp-hero-title {{ font-size: 2.3rem !important; }}
}}
.fp-hero-lead {{
  font-family: {SANS};
  font-size: 1.02rem;
  color: #D8E0EA;
  max-width: 50rem;
  line-height: 1.65;
  margin: 0;
}}
.fp-hero-lead strong {{ color: #FFFFFF; font-weight: 600; }}
.fp-hero-lead em {{ color: #F1F5F9; font-style: italic; }}

/* typography for the rest of the app (Inter) */
h1, h2, h3, h1 *, h2 *, h3 * {{ font-family: {SANS} !important; color: var(--navy) !important; }}
h2 {{ font-size: 1.45rem !important; font-weight: 600 !important; letter-spacing: -0.015em; }}
h3 {{ font-size: 1.15rem !important; font-weight: 600 !important; padding-top: 0.3rem !important; }}
p, li {{ font-size: 0.98rem; line-height: 1.6; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--muted) !important; font-size: 0.86rem; }}
.fp-eyebrow {{
  display: inline-block; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 0.35rem;
}}

/* cards with subtle Navy top border accent */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color: var(--line) !important; border-radius: var(--radius) !important;
  border-top: 3px solid var(--navy) !important;
  background: #FFFFFF; box-shadow: var(--shadow);
}}
[data-testid="stMetric"] {{
  position: relative; overflow: hidden; background: #FFFFFF;
  border: 1px solid var(--line); border-top: 3px solid var(--navy); border-radius: var(--radius);
  padding: 1rem 1.2rem; box-shadow: var(--shadow);
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-hover); border-color: #B2B8C2; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.82rem !important; font-weight: 500; color: var(--muted) !important; }}
[data-testid="stMetricValue"] {{
  color: var(--navy) !important; font-family: {SANS} !important; font-weight: 600; font-size: 1.6rem !important;
  font-variant-numeric: tabular-nums;
}}

/* progress bar */
[data-testid="stProgress"] > div > div > div > div {{
  background: var(--navy) !important; border-radius: var(--radius);
}}

/* buttons */
.stButton > button, [data-testid^="stBaseButton"] {{
  border-radius: var(--radius) !important; font-weight: 500; font-size: 0.94rem;
  transition: transform 0.15s var(--ease), box-shadow 0.2s var(--ease), background-color 0.2s ease;
}}
.stButton > button:hover, [data-testid^="stBaseButton"]:hover {{ transform: translateY(-1px); }}
[data-testid="stBaseButton-primary"] {{
  background: var(--navy) !important; border: 1px solid var(--navy) !important; color: #FFFFFF !important;
  border-top: 2px solid var(--crimson) !important;
}}
[data-testid="stBaseButton-primary"]:hover {{
  background: #15263D !important; border-color: #15263D !important;
  box-shadow: 0 4px 12px rgba(11, 25, 44, 0.16);
}}
[data-testid="stBaseButton-secondary"] {{
  background: #FFFFFF !important; border: 1px solid var(--line) !important;
  border-top: 2px solid var(--navy) !important; color: var(--navy) !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--navy) !important; background: #F8FAFC !important; }}

/* inputs */
[data-baseweb="input"], [data-baseweb="select"] > div {{ border-radius: var(--radius) !important; }}
[data-baseweb="input"]:focus-within {{ box-shadow: 0 0 0 2px rgba(11, 25, 44, 0.12); }}

/* tabs */
[data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid var(--line); gap: 1.5rem; }}
[data-testid="stTabs"] [role="tab"] p {{ font-size: 0.94rem; font-weight: 500; color: var(--muted); }}
[data-testid="stTabs"] [role="tab"]:hover p {{ color: var(--navy); }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: var(--navy); font-weight: 600; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: var(--crimson) !important; height: 2px; }}

/* tables */
[data-testid="stTable"] table {{ border-collapse: collapse; width: 100%; }}
[data-testid="stTable"] th, [data-testid="stTable"] td {{
  background: transparent !important; border: none !important; border-bottom: 1px solid var(--line) !important;
  padding: 0.55rem 0.75rem; font-size: 0.9rem;
}}
[data-testid="stTable"] td {{ text-align: right !important; font-family: {MONO}; font-size: 0.85rem; }}
[data-testid="stTable"] th {{ color: var(--muted); font-weight: 500; text-align: right !important; }}
[data-testid="stTable"] tbody th {{ text-align: left !important; color: var(--navy); }}
[data-testid="stTable"] tbody tr:hover {{ background: var(--surface); }}
[data-testid="stDataFrame"] {{
  border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden;
}}

/* expanders */
[data-testid="stExpander"] details {{
  border: 1px solid var(--line) !important; border-radius: var(--radius) !important; background: #FFFFFF;
}}
[data-testid="stExpander"] summary p {{ font-size: 0.95rem; font-weight: 500; color: var(--navy); }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, lead: str, eyebrow: str = "") -> None:
    st.markdown(
        f'<div class="fp-hero"><div class="fp-hero-inner">'
        f'  <div class="fp-hero-title-row">'
        f'    <div class="fp-hero-swiss-cross">✚</div>'
        f'    <div class="fp-hero-title">{title}</div>'
        f'  </div>'
        f'  <p class="fp-hero-lead">{lead}</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )