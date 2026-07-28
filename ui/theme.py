"""Design system for the AI Data Analyst workspace.

Relies natively on Streamlit's `.streamlit/config.toml` but adds premium HTML injections
for hiding the sidebar, layout enhancements, and injecting high-end SVG icons.
"""

from __future__ import annotations
from typing import Any

FONT_STACK = (
    '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, sans-serif'
)

# ── Chart series (distinct, legible on dark background) ──────────────────
SERIES = [
    "#5B82FF", "#2DD4BF", "#FBBF24", "#F87171", "#A78BFA",
    "#4ADE80", "#F472B6", "#38BDF8", "#A3E635", "#C084FC",
]
ACCENT = SERIES[0]

def chart_series() -> list[str]:
    return SERIES


def _recolor_traces(figure: Any, series: list[str], on_bg: str) -> None:
    index = 0
    for trace in figure.data:
        ttype = getattr(trace, "type", "")
        if ttype in ("heatmap", "heatmapgl", "contour", "image"):
            continue
        colour = series[index % len(series)]
        index += 1
        if ttype == "pie":
            trace.marker.colors = series
            trace.marker.line = dict(color=on_bg, width=1)
            continue
        if ttype in ("scatter", "scattergl"):
            trace.line.color = colour
            trace.marker.color = colour
            continue
        try:
            trace.marker.color = colour
        except (ValueError, AttributeError):
            pass


def style_figure(figure: Any) -> Any:
    """Style Plotly figures to match the native Streamlit dark theme."""
    bg_color = "rgba(0,0,0,0)"
    text_color = "#F8FAFC"
    muted_color = "#94A3B8"
    grid_color = "#334155"

    _recolor_traces(figure, SERIES, bg_color)
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family=FONT_STACK, size=12, color=muted_color),
        title=dict(font=dict(size=14, color=text_color, weight=700), x=0, xanchor="left", pad=dict(b=14)),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(bgcolor=bg_color, font=dict(size=11, color=muted_color),
                    orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#475569",
                         font=dict(family=FONT_STACK, size=12, color=text_color)),
        colorway=SERIES, height=390, separators=".,",
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, linecolor=grid_color,
                   tickfont=dict(size=11, color=muted_color), title_font=dict(size=11.5, color=muted_color)),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, linecolor=grid_color,
                   tickfont=dict(size=11, color=muted_color), title_font=dict(size=11.5, color=muted_color)),
    )
    return figure


def inject_premium_css() -> str:
    """CSS injection to hide sidebar and style the premium layout."""
    return """
    <style>
        /* Premium sidebar navigation */
        section[data-testid="stSidebar"] {
            background: #0B1120;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 10px 14px;
            border-radius: 10px;
            margin-bottom: 2px;
            font-weight: 600;
            transition: background 120ms ease;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,0.05);
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
            background: rgba(99,102,241,0.18);
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 2px !important; }

        /* Premium Typography & Spacing */
        h1, h2, h3 { font-family: "Inter", sans-serif !important; letter-spacing: -0.02em !important; }
        .hero-title { 
            font-size: 4rem !important; 
            font-weight: 800; 
            text-align: center; 
            background: linear-gradient(to right, #22D3EE, #6366F1, #EC4899);
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1.25rem;
            color: #94A3B8;
            margin-top: -10px;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Premium Card SVGs and alignment */
        .icon-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        .icon-header svg {
            width: 28px;
            height: 28px;
        }
        
        /* Adjust tabs to look cleaner without emojis */
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            padding-top: 1rem;
            padding-bottom: 1rem;
            font-size: 1.05rem;
        }
    </style>
    """

# ── SVG ICON SET ────────────────────────────────────────────────────────
SVG_CHAT = """<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#3B82F6"><path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>"""
SVG_DASHBOARD = """<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#10B981"><path stroke-linecap="round" stroke-linejoin="round" d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" /></svg>"""
SVG_INSIGHTS = """<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#F59E0B"><path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>"""
SVG_QUALITY = """<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#EF4444"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>"""
SVG_EXPLORE = """<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#8B5CF6"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>"""
SVG_WORKSPACE = """<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#64748B"><path stroke-linecap="round" stroke-linejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>"""

def render_header(title: str, svg: str) -> str:
    """Returns HTML for a premium header with an SVG icon next to it."""
    return f"""
    <div class="icon-header">
        {svg}
        <h3 style="margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: #F1F5F9;">{title}</h3>
    </div>
    """
