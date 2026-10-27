"""Turbofan Predictive Maintenance dashboard (Streamlit)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from src import ui_components as ui
from src.ui_data import (
    FEATURES, KEY_SENSORS, SENSORS, assess, hazard_from_rul, load_artifacts, load_background,
    load_presets, load_test_engines, score_engine,
)
from src.ui_theme import STATUS_STYLE, inject_css, sensor_label

st.set_page_config(page_title="Turbofan Health Monitor", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")
inject_css()

scaler, pca, model = art = load_artifacts()
presets = load_presets()
bg = load_background(scaler, pca)

MODES = ["🛰️ Fleet replay", "⚡ Quick presets", "🎛️ What-if lab"]
SPEEDS = {"Slow": (0.8, 1), "Normal": (0.4, 1), "Fast": (0.4, 3)}  # (seconds per tick, cycles per tick)
DEFAULT_ENGINE = 31


# ----------------------------------------------------------------------------- what-if helpers
def _clamp(feature: str, value: float) -> float:
    lo, hi = bg["ranges"].loc[feature, "min"], bg["ranges"].loc[feature, "max"]
    pad = 0.03 * (hi - lo)
    return float(min(max(value, lo - pad), hi + pad))


def _load_preset(name: str) -> None:
    row = presets[presets["Preset"] == name].iloc[0]
    for f in FEATURES:
        st.session_state[f"wi_{f}"] = _clamp(f, float(row[f]))
    st.session_state["dial"] = {"Normal": 0, "Degrading": 55, "Critical": 100}[name]


def _apply_dial() -> None:
    t = st.session_state["dial"] / 100
    h, c = bg["raw_mean"]["Healthy"], bg["raw_mean"]["Critical"]
    for i, f in enumerate(FEATURES):
        if f.startswith("s_"):
            st.session_state[f"wi_{f}"] = _clamp(f, h[i] + t * (c[i] - h[i]))


# ----------------------------------------------------------------------------- replay transport
def _controls(min_c: int, max_c: int, interval: float, step: int) -> None:
    """Play / pause / restart / end buttons. Runs as a fragment; timer ticks advance the shared cycle."""
    playing = st.session_state.get("playing", False)
    now = time.monotonic()

    # Handle clicks BEFORE the tick logic: a tick calls st.rerun(), which would otherwise swallow a click.
    b1, b2, b3 = st.columns(3)
    if b1.button("⏸" if playing else "▶", key="btn_play", type="primary", width="stretch", help="Play / pause the replay"):
        if not playing and st.session_state["cycle"] >= max_c:
            st.session_state["cycle"] = min_c
        st.session_state["playing"] = not playing
        st.session_state["_tick"] = now
        st.rerun()
    if b2.button("⏮", key="btn_start", width="stretch", help="Back to the first cycle"):
        st.session_state.update(cycle=min_c, playing=False)
        st.rerun()
    if b3.button("⏭", key="btn_end", width="stretch", help="Jump to the latest cycle"):
        st.session_state.update(cycle=max_c, playing=False)
        st.rerun()

    if playing and now - st.session_state.get("_tick", 0.0) >= interval * 0.8:
        st.session_state["_tick"] = now
        nxt = min(max_c, st.session_state["cycle"] + step)
        st.session_state["cycle"] = nxt
        if nxt >= max_c:
            st.session_state["playing"] = False
        st.rerun()  # redraw the whole page for the new cycle


# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🧭 Control panel")
mode = st.sidebar.radio("Data source", MODES, key="mode", label_visibility="collapsed")

traj = scaled_df = None
true_status = None
prev_risk = None
cycle = None

if mode == MODES[0]:
    te = load_test_engines()
    engines = sorted(int(e) for e in te["engine_id"].unique())
    engine = st.sidebar.selectbox("Engine", engines, index=engines.index(DEFAULT_ENGINE), format_func=lambda e: f"Engine #{e}", key="engine")
    speed = st.sidebar.segmented_control("Playback speed", list(SPEEDS), default="Normal", key="speed") or "Normal"
    interval, step = SPEEDS[speed]
    traj, scaled_df = score_engine(engine, scaler, pca, model)
    min_c, max_c = int(traj["cycle"].min()), int(traj["cycle"].max())
    st.sidebar.caption(f"{max_c} recorded cycles · true RUL at last cycle: {int(traj['RUL'].iloc[-1])}")

    ctx = ("replay", engine)
    if st.session_state.get("_ctx") != ctx:
        st.session_state.update(_ctx=ctx, cycle=max_c, playing=False, last_status=None)

    ui.hero(mode, live=st.session_state.get("playing", False))
    st.markdown("##### 🎬 Flight recorder")
    t1, t2 = st.columns([1.4, 5], vertical_alignment="bottom")
    with t1:
        st.fragment(run_every=interval if st.session_state.get("playing", False) else None)(_controls)(min_c, max_c, interval, step)
    with t2:
        cycle = st.slider("Flight cycle", min_c, max_c, key="cycle")

    idx = int(np.where(traj["cycle"].to_numpy() == cycle)[0][0])
    vector = traj.loc[idx, FEATURES].to_numpy(float)
    true_status = str(traj.loc[idx, "Status"])
    prev_risk = float(traj.loc[idx - 1, "Risk"]) if idx > 0 else None

elif mode == MODES[1]:
    choice = st.sidebar.selectbox("Engine snapshot", ["Normal", "Degrading", "Critical"], key="preset")
    row = presets[presets["Preset"] == choice].iloc[0]
    vector = row[FEATURES].to_numpy(float)
    true_status = str(hazard_from_rul([row["RUL"]])[0])
    st.sidebar.caption(f"Real test-set cycle · true RUL: {int(row['RUL'])}")
    if st.session_state.get("_ctx") != ("preset", choice):
        st.session_state.update(_ctx=("preset", choice), last_status=None)
    ui.hero(mode)

else:
    normal = presets[presets["Preset"] == "Normal"].iloc[0]
    for f in FEATURES:
        st.session_state.setdefault(f"wi_{f}", _clamp(f, float(normal[f])))
    st.session_state.setdefault("dial", 0)

    st.sidebar.markdown("**Degradation dial**")
    st.sidebar.slider("Healthy → Critical", 0, 100, key="dial", format="%d%%", on_change=_apply_dial, help="Blends all sensors between the average healthy and average critical engine.")
    st.sidebar.markdown("**Load a real snapshot**")
    b = st.sidebar.columns(3)
    for col, (label, name) in zip(b, (("Normal", "Normal"), ("Warning", "Degrading"), ("Critical", "Critical"))):
        col.button(label, key=f"load_{name}", on_click=_load_preset, args=(name,), width="stretch")

    st.sidebar.markdown("**Fine-tune sensors**")

    def _slider(f: str) -> None:
        lo, hi = float(bg["ranges"].loc[f, "min"]), float(bg["ranges"].loc[f, "max"])
        pad = 0.03 * (hi - lo)
        st.sidebar.slider(sensor_label(f, long=True), lo - pad, hi + pad, step=(hi - lo) / 400, key=f"wi_{f}", format="%.2f")

    for f in KEY_SENSORS:
        _slider(f)
    with st.sidebar.expander("More sensors"):
        for f in [s for s in SENSORS if s not in KEY_SENSORS]:
            lo, hi = float(bg["ranges"].loc[f, "min"]), float(bg["ranges"].loc[f, "max"])
            pad = 0.03 * (hi - lo)
            st.slider(sensor_label(f), lo - pad, hi + pad, step=(hi - lo) / 400, key=f"wi_{f}", format="%.3f")

    vector = np.array([st.session_state[f"wi_{f}"] for f in FEATURES], dtype=float)
    if st.session_state.get("_ctx") != ("whatif",):
        st.session_state.update(_ctx=("whatif",), last_status=None)
    ui.hero(mode)

# ----------------------------------------------------------------------------- assess + toast
a = assess(vector, art, bg)
if st.session_state.get("last_status") not in (None, a["status"]):
    st.toast(f"Status changed: {st.session_state['last_status']} → {a['status']}", icon="🚨" if a["status"] == "Critical" else "⚠️" if a["status"] == "Warning" else "✅")
st.session_state["last_status"] = a["status"]

ui.status_cards(a, prev_risk, true_status)
st.write("")

# Lazy tabs: only the selected tab is built each rerun, which keeps replay playback smooth.
tab_live, tab_why, tab_space, tab_lab = st.tabs(
    ["📡 Live monitor", "🔍 Why this prediction", "🗺️ Sensor space", "🏁 Model lab"], key="main_tabs", on_change="rerun"
)

with tab_live:
    if tab_live.open:
        g, p = st.columns([1, 1.25])
        with g:
            st.plotly_chart(ui.risk_gauge(a, prev_risk), key="gauge", config=ui.PLOT_CFG)
        with p:
            st.plotly_chart(ui.probability_bars(a), key="proba", config=ui.PLOT_CFG)
            ui.advice_box(a["status"])
        if traj is not None:
            st.plotly_chart(ui.trajectory_chart(traj, cycle), key="timeline", config=ui.PLOT_CFG)
            picked = st.multiselect("Sensors to trace", SENSORS, default=["s_2", "s_11", "s_12", "s_21"], format_func=sensor_label, max_selections=6, key="trend_sel")
            if picked:
                st.plotly_chart(ui.sensor_trends(traj, scaled_df, picked, cycle), key="trends", config=ui.PLOT_CFG)
        else:
            st.markdown("##### Key sensors vs healthy average")
            ui.key_sensor_metrics(a, bg)

with tab_why:
    if tab_why.open:
        top_n = st.slider("Sensors to show", 3, 10, 6, key="topn")
        l, r = st.columns([1.1, 1])
        with l:
            st.plotly_chart(ui.contribution_bars(a, top_n), key="contrib", config=ui.PLOT_CFG)
            st.caption("Score = how far the sensor sits from the healthy average, weighted by how much the Random Forest relies on it (via the PCA loadings).")
        with r:
            st.plotly_chart(ui.fingerprint_radar(a, bg), key="radar", config=ui.PLOT_CFG)
        with st.expander("Sensor glossary for the top sensors"):
            g = a["contrib"].head(top_n).copy()
            g["Sensor"] = g["Feature"].map(lambda s: sensor_label(s, long=True))
            g["Shift vs healthy"] = g["DeltaPct"].map(lambda v: f"{v:+.2f}%")
            st.dataframe(g[["Sensor", "Raw", "HealthyRaw", "Shift vs healthy", "Score"]].rename(columns={"Raw": "Current", "HealthyRaw": "Healthy avg"}), hide_index=True)

with tab_space:
    if tab_space.open:
        m, v = st.columns([1.6, 1])
        with m:
            st.plotly_chart(ui.pca_map(bg, pca, a, traj, cycle), key="pca_map", config=ui.PLOT_CFG)
        with v:
            st.plotly_chart(ui.variance_chart(pca), key="pca_var", config=ui.PLOT_CFG)
            st.caption("Each dot is a training-set reading squeezed from 18 features down to 2. Degradation shows up mostly along PC1: healthy readings sit on the left and drift right through Warning into Critical. The horizontal banding in PC2 comes from the quantized operating-setting channels rather than from wear. Click legend items to isolate a class.")

with tab_lab:
    if tab_lab.open:
        ui.model_lab()

st.caption("Built on NASA C-MAPSS FD001 · RUL classes: Healthy > 30, Warning 16–30, Critical ≤ 15 cycles · the risk index is a simple summary of the class probabilities, not a calibrated failure probability.")
