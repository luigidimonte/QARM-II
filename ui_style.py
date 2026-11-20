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
NAVY = "#0B1F3A"        # text, headings
BLUE = "#0A3D91"        # primary
STEEL = "#5B7FA8"       # secondary
MIST = "#DCE6F2"        # light blue-grey
LINE = "#D9E0EA"        # thin rules and gridlines
GREY = "#6B7A90"        # captions, secondary marks
GOLD = "#B39B63"        # muted accent, gold only

ASSET_COLORS: Dict[str, str] = {
    "SPY": NAVY,
    "EWL": BLUE,
    "EEM": STEEL,
    "IEF": "#8FA3BC",
    "LQD": MIST,
    "GLD": GOLD,
}
# thin outline so the lightest fills stay visible on white
OUTLINE = dict(color="#FFFFFF", width=1)

SANS = ('-apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Helvetica, Arial, '
        'sans-serif')
SERIF = 'Georgia, "Times New Roman", Times, serif'

# ----------------------------------------------------------------------------- Plotly template
pio.templates["fire"] = go.layout.Template(layout=go.Layout(
    font=dict(family=SANS, size=13, color=NAVY),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    colorway=[BLUE, STEEL, NAVY, "#8FA3BC", GOLD, MIST],
    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=LINE, linewidth=1,
               ticks="outside", tickcolor=LINE, ticklen=4, title=dict(font=dict(size=12, color=GREY))),
    yaxis=dict(showgrid=True, gridcolor="#EEF2F7", gridwidth=1, zeroline=False, showline=False,
               title=dict(font=dict(size=12, color=GREY))),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                font=dict(size=12, color=NAVY), bgcolor="rgba(0,0,0,0)"),
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
    """Numeric columns stay numeric (so they are right-aligned); the Styler only sets the display text."""
    sty = df.style.format(formats or {}, na_rep="-")
    st.dataframe(sty, hide_index=hide_index, width="stretch")


# ----------------------------------------------------------------------------- CSS
CSS = f"""
<style>
:root {{
  --navy: {NAVY}; --blue: {BLUE}; --steel: {STEEL}; --mist: {MIST}; --line: {LINE}; --grey: {GREY};
}}
html, body, [class*="st-"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: {SANS};
}}
.stApp {{ background: #FFFFFF; color: var(--navy); }}

/* layout: centred column, generous white space */
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1180px; padding: 3.5rem 2.5rem 5rem 2.5rem;
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: 2rem 1rem 3rem 1rem; }}
}}
header[data-testid="stHeader"] {{ background: rgba(255,255,255,0.92); }}

/* typography */
h1, h2, h3, h1 *, h2 *, h3 * {{
  font-family: {SERIF} !important; color: var(--navy) !important; font-weight: 400 !important;
}}
h1 {{ font-size: 2.5rem !important; letter-spacing: 0.04em; padding-bottom: 0.2rem !important; }}
h2 {{ font-size: 1.75rem !important; letter-spacing: 0.03em; }}
h3 {{ font-size: 1.35rem !important; letter-spacing: 0.025em; padding-top: 0.6rem !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--grey) !important; font-size: 0.88rem; }}
.fp-eyebrow {{
  font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--steel);
  margin-bottom: 0.25rem;
}}
.fp-lead {{ font-size: 1.05rem; color: var(--navy); max-width: 46rem; line-height: 1.6; }}
hr, [data-testid="stDivider"] hr {{ border: none; border-top: 1px solid var(--line) !important; margin: 1.6rem 0; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: var(--mist); border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label p {{
  font-family: {SANS} !important; font-size: 0.72rem !important; text-transform: uppercase;
  letter-spacing: 0.14em; color: var(--navy) !important; font-weight: 600 !important;
}}

/* containers with a border: thin line, no heavy box */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--line) !important; border-radius: 2px !important; }}

/* metrics */
[data-testid="stMetric"] {{
  border: 1px solid var(--line); border-radius: 2px; padding: 1.1rem 1.25rem; background: #FFFFFF;
  transition: transform 0.2s ease, border-color 0.2s ease;
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-2px); border-color: #B9C7DA; }}
[data-testid="stMetricLabel"] p {{
  font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.12em; color: var(--grey) !important;
}}
[data-testid="stMetricValue"] {{ color: var(--navy) !important; font-family: {SERIF}; font-weight: 400; font-size: 1.7rem !important; }}

/* buttons: flat and rectangular */
.stButton > button, [data-testid^="stBaseButton"] {{
  border-radius: 2px !important; box-shadow: none !important; font-weight: 500;
  letter-spacing: 0.03em; transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}}
[data-testid="stBaseButton-primary"] {{ background: var(--blue) !important; border: 1px solid var(--blue) !important; color: #FFFFFF !important; }}
[data-testid="stBaseButton-primary"]:hover {{ background: #082F70 !important; border-color: #082F70 !important; }}
[data-testid="stBaseButton-secondary"] {{ background: #FFFFFF !important; border: 1px solid var(--line) !important; color: var(--navy) !important; }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--blue) !important; color: var(--blue) !important; }}

/* tabs: quiet underline */
[data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid var(--line); gap: 2rem; }}
[data-testid="stTabs"] [role="tab"] p {{ font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: var(--blue); }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top: 1.6rem; }}

/* progress bar and alerts */
[data-testid="stProgress"] > div > div > div > div {{ background-color: var(--blue) !important; }}
[data-testid="stAlert"] {{ border-radius: 2px; }}

/* tables: no zebra, right-aligned cells in static tables */
[data-testid="stTable"] table {{ border-collapse: collapse; width: 100%; }}
[data-testid="stTable"] th, [data-testid="stTable"] td {{
  background: #FFFFFF !important; border: none !important; border-bottom: 1px solid var(--line) !important;
  padding: 0.5rem 0.75rem; font-size: 0.9rem;
}}
[data-testid="stTable"] td {{ text-align: right !important; font-variant-numeric: tabular-nums; }}
[data-testid="stTable"] th {{ color: var(--grey); font-weight: 500; text-align: right !important; }}
[data-testid="stTable"] tbody th {{ text-align: left !important; color: var(--navy); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 2px; }}

/* expanders */
[data-testid="stExpander"] details {{ border: none !important; border-bottom: 1px solid var(--line) !important; border-radius: 0 !important; }}
[data-testid="stExpander"] summary p {{ font-size: 0.95rem; letter-spacing: 0.02em; color: var(--navy); }}

/* gentle fade-in for the main blocks */
@keyframes fp-fade {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
[data-testid="stTabs"] [role="tabpanel"], [data-testid="stMetric"], [data-testid="stPlotlyChart"],
[data-testid="stVerticalBlockBorderWrapper"], h1, .fp-lead {{
  animation: fp-fade 0.5s ease-out both;
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
  [data-testid="stMetric"]:hover {{ transform: none; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, lead: str, eyebrow: str = "Financial independence planning") -> None:
    st.markdown(f'<div class="fp-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<p class="fp-lead">{lead}</p>', unsafe_allow_html=True)
    st.divider()
