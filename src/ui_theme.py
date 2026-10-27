"""Visual theme for the dashboard: palette, CSS, Plotly styling, sensor metadata."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

C = {
    "bg": "#0b1220",
    "panel": "#111a2e",
    "text": "#e6edf7",
    "muted": "#8fa3c0",
    "cyan": "#38bdf8",
    "green": "#34d399",
    "amber": "#fbbf24",
    "red": "#f87171",
    "violet": "#a78bfa",
}

STATUS_STYLE = {
    "Healthy": {
        "color": C["green"],
        "zone": "> 30 cycles",
        "headline": "Operating normally",
        "advice": "No action needed. Keep up routine monitoring.",
    },
    "Warning": {
        "color": C["amber"],
        "zone": "16 – 30 cycles",
        "headline": "Degradation detected",
        "advice": "Schedule an inspection within the next 15–30 cycles and keep an eye on the flagged sensors.",
    },
    "Critical": {
        "color": C["red"],
        "zone": "≤ 15 cycles",
        "headline": "Failure approaching",
        "advice": "Plan maintenance immediately. Remaining useful life is 15 cycles or fewer.",
    },
}

# NASA C-MAPSS sensor descriptions: id -> (symbol, description)
SENSOR_INFO = {
    "s_1": ("T2", "Fan inlet temperature"),
    "s_2": ("T24", "LPC outlet temperature"),
    "s_3": ("T30", "HPC outlet temperature"),
    "s_4": ("T50", "LPT outlet temperature"),
    "s_5": ("P2", "Fan inlet pressure"),
    "s_6": ("P15", "Bypass-duct pressure"),
    "s_7": ("P30", "HPC outlet pressure"),
    "s_8": ("Nf", "Physical fan speed"),
    "s_9": ("Nc", "Physical core speed"),
    "s_10": ("epr", "Engine pressure ratio"),
    "s_11": ("Ps30", "HPC outlet static pressure"),
    "s_12": ("phi", "Fuel flow / Ps30 ratio"),
    "s_13": ("NRf", "Corrected fan speed"),
    "s_14": ("NRc", "Corrected core speed"),
    "s_15": ("BPR", "Bypass ratio"),
    "s_16": ("farB", "Burner fuel-air ratio"),
    "s_17": ("htBleed", "Bleed enthalpy"),
    "s_18": ("Nf_dmd", "Demanded fan speed"),
    "s_19": ("PCNfR_dmd", "Demanded corrected fan speed"),
    "s_20": ("W31", "HPT coolant bleed"),
    "s_21": ("W32", "LPT coolant bleed"),
}


def sensor_label(sensor: str, long: bool = False) -> str:
    """Return a human readable label such as ``s_11 · Ps30``."""
    if sensor in SENSOR_INFO:
        sym, desc = SENSOR_INFO[sensor]
        return f"{sensor} · {sym} ({desc})" if long else f"{sensor} · {sym}"
    return sensor


def rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba()`` string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def style_fig(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    """Apply the dark, transparent house style to a Plotly figure."""
    base = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C["panel"], font_size=13),
    )
    base.update(layout)  # caller overrides win
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    return fig


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(56,189,248,.16), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(167,139,250,.14), transparent 60%),
    #0b1220;
}
.block-container { padding-top: 1.6rem; max-width: 1350px; }
footer { visibility: hidden; }

/* ---------- hero ---------- */
.hero {
  padding: 22px 28px; border-radius: 20px; margin-bottom: 18px;
  background: linear-gradient(120deg, rgba(56,189,248,.14), rgba(167,139,250,.12));
  border: 1px solid rgba(148,163,184,.18);
  animation: fadeUp .6s ease both;
}
.hero h1 {
  margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, #e6edf7, #38bdf8 60%, #a78bfa);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 6px 0 0 0; color: #8fa3c0; font-size: .98rem; }
.pill {
  display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .75rem;
  font-weight: 600; margin-right: 8px; border: 1px solid rgba(148,163,184,.3); color: #cbd5e1;
  background: rgba(15,23,42,.5);
}
.pill.live { color: #fca5a5; border-color: rgba(248,113,113,.5); }
.pill.live::before {
  content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #f87171; margin-right: 7px; animation: pulse 1.2s infinite;
}

/* ---------- cards ---------- */
.card {
  --accent: #38bdf8;
  background: rgba(17,26,46,.72); backdrop-filter: blur(8px);
  border: 1px solid rgba(148,163,184,.16); border-top: 3px solid var(--accent);
  border-radius: 16px; padding: 16px 18px; min-height: 130px;
  transition: transform .2s ease, box-shadow .2s ease;
  animation: fadeUp .5s ease both;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 10px 30px -10px var(--accent); }
.card .label { font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; color: #8fa3c0; font-weight: 600; }
.card .value { font-size: 1.9rem; font-weight: 800; margin: 6px 0 2px 0; line-height: 1.1; }
.card .sub { font-size: .82rem; color: #8fa3c0; }
.card.glow-Critical { animation: fadeUp .5s ease both, glowRed 1.6s ease-in-out infinite; }
.dot {
  display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 10px;
  background: var(--accent); animation: pulse 1.5s infinite; vertical-align: middle;
}
.advice {
  border-left: 4px solid var(--accent); background: rgba(17,26,46,.6);
  border-radius: 0 12px 12px 0; padding: 12px 16px; margin: 6px 0 4px 0; color: #dbe5f3;
  animation: fadeUp .5s ease both;
}
.advice b { color: var(--accent); }

/* ---------- widgets ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1a30 0%, #0b1220 100%);
  border-right: 1px solid rgba(148,163,184,.12);
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
  height: 42px; padding: 0 18px; border-radius: 12px 12px 0 0;
  background: rgba(17,26,46,.55); color: #8fa3c0; font-weight: 600;
  transition: background .2s ease, color .2s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(56,189,248,.12); color: #e6edf7; }
.stTabs [aria-selected="true"] { background: rgba(56,189,248,.18) !important; color: #38bdf8 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #38bdf8; height: 3px; }
div[data-testid="stMetric"] {
  background: rgba(17,26,46,.65); border: 1px solid rgba(148,163,184,.15);
  border-radius: 14px; padding: 12px 14px; transition: transform .2s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-3px); }
.stButton > button { border-radius: 12px; font-weight: 600; transition: transform .15s ease; }
.stButton > button:hover { transform: translateY(-2px); }
div[data-testid="stPlotlyChart"] {
  background: rgba(17,26,46,.45); border: 1px solid rgba(148,163,184,.12);
  border-radius: 16px; padding: 6px;
}

@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 var(--accent, #f87171); } 70% { box-shadow: 0 0 0 9px transparent; } 100% { box-shadow: 0 0 0 0 transparent; } }
@keyframes glowRed { 0%,100% { box-shadow: 0 0 0 0 rgba(248,113,113,0); } 50% { box-shadow: 0 0 26px 2px rgba(248,113,113,.45); } }
"""


def inject_css() -> None:
    """Inject the global stylesheet."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
