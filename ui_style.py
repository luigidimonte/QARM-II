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
# Names kept from the first version so app_fire.py does not change; values are the new look.
NAVY = "#0F172A"        # ink: text, headings, primary buttons
BLUE = "#2563EB"        # accent: main lines, selection, focus
STEEL = "#64748B"       # secondary series
MIST = "#F1F5F9"        # light surface
LINE = "#E2E8F0"        # thin rules and gridlines
GREY = "#64748B"        # captions, secondary marks
GOLD = "#D97706"        # warm highlight (γ line, markers)
FAN_OUTER = "rgba(37, 99, 235, 0.12)"
FAN_INNER = "rgba(37, 99, 235, 0.22)"
REACHED = "#0F766E"     # simulated paths that satisfy / miss the FIRE condition
MISSED = "#EA580C"

# one hue per asset: equities blue / red / violet, real estate pink, bonds teal (dark / light) and
# green (inflation-linked), gold yellow
ASSET_COLORS: Dict[str, str] = {
    "SPY": "#1E40AF",
    "SMICHA.SW": "#DC2626",
    "EEM": "#7C3AED",
    "VNQ": "#F9A8D4",
    "IEF": "#0F766E",
    "LQD": "#5EEAD4",
    "TIP": "#65A30D",
    "GLD": "#EAB308",
}
# thin outline so the lightest fills stay visible on white
OUTLINE = dict(color="#FFFFFF", width=1)

FONT_URL = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
            "&family=JetBrains+Mono:wght@500&display=swap")
SANS = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif'
MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

# ----------------------------------------------------------------------------- Plotly template
pio.templates["fire"] = go.layout.Template(layout=go.Layout(
    font=dict(family=SANS, size=13, color=NAVY),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    colorway=[BLUE, STEEL, NAVY, "#8FA3BC", GOLD, MIST],
    xaxis=dict(automargin=True, showgrid=False, zeroline=False, showline=True, linecolor=LINE, linewidth=1,
               ticks="outside", tickcolor=LINE, ticklen=4, title=dict(font=dict(size=12, color=GREY))),
    yaxis=dict(automargin=True, showgrid=True, gridcolor="#EEF2F7", gridwidth=1, zeroline=False, showline=False,
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
@import url('{FONT_URL}');
:root {{
  --ink: {NAVY}; --accent: {BLUE}; --slate: {STEEL}; --surface: {MIST}; --line: {LINE}; --muted: {GREY};
  --radius: 10px; --ease: cubic-bezier(0.22, 1, 0.36, 1);
  --shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.06);
  --shadow-hover: 0 8px 24px rgba(15, 23, 42, 0.08);
}}
html, body, [class*="st-"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: {SANS}; -webkit-font-smoothing: antialiased;
}}
/* the rule above also matches the icon glyphs, which need their own font */
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: #FAFBFC; color: var(--ink); }}

/* layout */
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1160px; padding: 2.5rem 2.5rem 5rem 2.5rem;
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: 1.5rem 1rem 3rem 1rem; }}
}}
header[data-testid="stHeader"] {{ background: rgba(250, 251, 252, 0.85); backdrop-filter: blur(8px); }}

/* typography: one sans family, weight and tracking do the hierarchy */
h1, h2, h3, h1 *, h2 *, h3 * {{ font-family: {SANS} !important; color: var(--ink) !important; }}
h1 {{ font-size: 2.25rem !important; font-weight: 700 !important; letter-spacing: -0.025em; }}
h2 {{ font-size: 1.5rem !important; font-weight: 600 !important; letter-spacing: -0.02em; }}
h3 {{ font-size: 1.15rem !important; font-weight: 600 !important; letter-spacing: -0.01em; padding-top: 0.5rem !important; }}
p, li {{ line-height: 1.6; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--muted) !important; font-size: 0.86rem; }}
.fp-eyebrow {{
  display: inline-block; font-size: 0.75rem; font-weight: 500; color: var(--accent);
  background: rgba(37, 99, 235, 0.08); border-radius: 999px; padding: 0.2rem 0.65rem; margin-bottom: 0.5rem;
}}
.fp-lead {{ font-size: 1.05rem; color: #334155; max-width: 44rem; line-height: 1.6; }}
hr, [data-testid="stDivider"] hr {{ border: none; border-top: 1px solid var(--line) !important; margin: 1.5rem 0; }}

/* brand bar above the title */
.fp-brand {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.25rem; }}
.fp-logo {{
  width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center;
  color: #FFFFFF; font-weight: 700; font-size: 0.95rem;
  background: linear-gradient(135deg, var(--accent), #0F766E 60%, var(--accent));
  background-size: 200% 200%; animation: fp-sheen 8s ease-in-out infinite;
}}
.fp-brand-name {{ font-weight: 600; font-size: 0.95rem; color: var(--ink); letter-spacing: -0.01em; }}
.fp-rule {{
  height: 2px; border-radius: 2px; margin: 1.25rem 0 1.75rem 0;
  background: linear-gradient(90deg, var(--accent), #0F766E, transparent);
  transform-origin: left; animation: fp-grow 0.9s var(--ease) both;
}}

/* sidebar */
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid var(--line); }}

/* cards: bordered containers and metrics */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color: var(--line) !important; border-radius: var(--radius) !important;
  background: #FFFFFF; box-shadow: var(--shadow);
}}
[data-testid="stMetric"] {{
  position: relative; overflow: hidden; background: #FFFFFF;
  border: 1px solid var(--line); border-radius: var(--radius); padding: 1rem 1.2rem; box-shadow: var(--shadow);
  transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease), border-color 0.25s var(--ease);
}}
[data-testid="stMetric"]::before {{
  content: ""; position: absolute; left: 0; top: 0; height: 3px; width: 100%;
  background: linear-gradient(90deg, var(--accent), #0F766E);
  transform: scaleX(0); transform-origin: left; transition: transform 0.35s var(--ease);
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-hover); border-color: #CBD5E1; }}
[data-testid="stMetric"]:hover::before {{ transform: scaleX(1); }}
[data-testid="stMetricLabel"] p {{ font-size: 0.8rem !important; font-weight: 500; color: var(--muted) !important; }}
[data-testid="stMetricValue"] {{
  color: var(--ink) !important; font-family: {SANS}; font-weight: 600; font-size: 1.6rem !important;
  letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
}}

/* buttons */
.stButton > button, [data-testid^="stBaseButton"] {{
  border-radius: 8px !important; font-weight: 500; letter-spacing: 0;
  transition: transform 0.15s var(--ease), box-shadow 0.2s var(--ease), background-color 0.2s ease,
              border-color 0.2s ease, color 0.2s ease;
}}
.stButton > button:hover, [data-testid^="stBaseButton"]:hover {{ transform: translateY(-1px); }}
.stButton > button:active, [data-testid^="stBaseButton"]:active {{ transform: translateY(0) scale(0.98); }}
[data-testid="stBaseButton-primary"] {{
  background: var(--ink) !important; border: 1px solid var(--ink) !important; color: #FFFFFF !important;
}}
[data-testid="stBaseButton-primary"]:hover {{
  background: #1E293B !important; border-color: #1E293B !important; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.18);
}}
[data-testid="stBaseButton-secondary"] {{
  background: #FFFFFF !important; border: 1px solid var(--line) !important; color: var(--ink) !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--accent) !important; color: var(--accent) !important; }}

/* inputs: accent focus ring */
[data-baseweb="input"], [data-baseweb="select"] > div {{ border-radius: 8px !important; transition: box-shadow 0.2s ease; }}
[data-baseweb="input"]:focus-within {{ box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15); }}

/* tabs: pill-like labels, accent underline */
[data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid var(--line); gap: 1.5rem; }}
[data-testid="stTabs"] [role="tab"] p {{ font-size: 0.92rem; font-weight: 500; color: var(--muted); transition: color 0.2s ease; }}
[data-testid="stTabs"] [role="tab"]:hover p {{ color: var(--ink); }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: var(--ink); font-weight: 600; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; height: 2px; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top: 1.5rem; }}

/* progress bar: accent fill that slides to its new width */
[data-testid="stProgress"] > div > div > div > div {{
  background: linear-gradient(90deg, var(--accent), #0F766E) !important; transition: width 0.6s var(--ease);
}}
[data-testid="stAlert"] {{ border-radius: var(--radius); }}

/* tables: numbers in a mono face, right-aligned */
[data-testid="stTable"] table {{ border-collapse: collapse; width: 100%; }}
[data-testid="stTable"] th, [data-testid="stTable"] td {{
  background: transparent !important; border: none !important; border-bottom: 1px solid var(--line) !important;
  padding: 0.55rem 0.75rem; font-size: 0.9rem;
}}
[data-testid="stTable"] td {{ text-align: right !important; font-family: {MONO}; font-size: 0.85rem; }}
[data-testid="stTable"] th {{ color: var(--muted); font-weight: 500; text-align: right !important; }}
[data-testid="stTable"] tbody th {{ text-align: left !important; color: var(--ink); }}
[data-testid="stTable"] tbody tr {{ transition: background-color 0.15s ease; }}
[data-testid="stTable"] tbody tr:hover {{ background: var(--surface); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }}

/* expanders */
[data-testid="stExpander"] details {{
  border: 1px solid var(--line) !important; border-radius: var(--radius) !important; background: #FFFFFF;
  transition: box-shadow 0.2s ease;
}}
[data-testid="stExpander"] details:hover {{ box-shadow: var(--shadow); }}
[data-testid="stExpander"] summary p {{ font-size: 0.95rem; font-weight: 500; color: var(--ink); }}

/* motion: blocks rise in, metrics in a short cascade */
@keyframes fp-rise {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes fp-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes fp-grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes fp-sheen {{ 0%, 100% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} }}
h1, .fp-lead, .fp-brand, [data-testid="stVerticalBlockBorderWrapper"] {{ animation: fp-rise 0.6s var(--ease) both; }}
.fp-lead {{ animation-delay: 0.08s; }}
[data-testid="stTabs"] [role="tabpanel"] {{ animation: fp-fade 0.4s ease-out both; }}
[data-testid="stPlotlyChart"] {{ animation: fp-fade 0.6s ease-out both; animation-delay: 0.1s; }}
[data-testid="stMetric"] {{ animation: fp-rise 0.55s var(--ease) both; }}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {{ animation-delay: 0.07s; }}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {{ animation-delay: 0.14s; }}
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {{ animation-delay: 0.21s; }}
[data-testid="stColumn"]:nth-child(5) [data-testid="stMetric"] {{ animation-delay: 0.28s; }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, lead: str, eyebrow: str = "Financial independence planning") -> None:
    st.markdown(f'<div class="fp-brand"><div class="fp-logo">F</div>'
                f'<div class="fp-brand-name">FIRE Planner</div></div>'
                f'<div class="fp-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<p class="fp-lead">{lead}</p><div class="fp-rule"></div>', unsafe_allow_html=True)
