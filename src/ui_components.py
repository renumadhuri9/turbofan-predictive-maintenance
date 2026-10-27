"""Reusable Streamlit / Plotly components for the dashboard."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.ui_data import CLASSES, FEATURES, KEY_SENSORS, SENSORS
from src.ui_theme import C, STATUS_STYLE, rgba, sensor_label, style_fig

PLOT_CFG = {"displayModeBar": False}


def _card(label: str, value: str, sub: str = "", color: str = C["cyan"], extra: str = "") -> str:
    return (
        f'<div class="card {extra}" style="--accent:{color}">'
        f'<div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def hero(mode: str, live: bool = False) -> None:
    """Page header with mode pill and optional LIVE badge."""
    live_pill = '<span class="pill live">LIVE REPLAY</span>' if live else ""
    st.markdown(
        '<div class="hero"><h1>✈️ Turbofan Health Monitor</h1>'
        f'<p>{live_pill}<span class="pill">{mode}</span>'
        "NASA C-MAPSS FD001 · PCA + Random Forest · explainable remaining-life triage</p></div>",
        unsafe_allow_html=True,
    )


def status_cards(a: Dict[str, Any], prev_risk: Optional[float] = None, true_status: Optional[str] = None) -> None:
    """Four headline KPI cards."""
    s = STATUS_STYLE[a["status"]]
    col = s["color"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        _card("Health status", f'<span class="dot"></span>{a["status"]}', s["headline"], col, f"glow-{a['status']}"),
        unsafe_allow_html=True,
    )
    sub = "Predicted RUL zone"
    if true_status is not None:
        ok = "✅ matches" if true_status == a["status"] else f"⚠️ actual: {true_status}"
        sub = f"Predicted zone · {ok}"
    c2.markdown(_card("Remaining useful life", s["zone"], sub, col), unsafe_allow_html=True)
    c3.markdown(
        _card("Model confidence", f'{a["confidence"]:.1%}', "Top class probability", C["cyan"]),
        unsafe_allow_html=True,
    )
    arrow = ""
    if prev_risk is not None:
        d = a["risk"] - prev_risk
        arrow = f"{'▲' if d > 0.05 else '▼' if d < -0.05 else '■'} {d:+.1f} vs previous cycle"
    c4.markdown(
        _card("Failure risk index", f'{a["risk"]:.0f}<span style="font-size:1rem"> / 100</span>', arrow or "0 = healthy · 100 = critical", C["violet"]),
        unsafe_allow_html=True,
    )


def advice_box(status: str) -> None:
    """Colour-coded recommendation under the gauge."""
    s = STATUS_STYLE[status]
    st.markdown(
        f'<div class="advice" style="--accent:{s["color"]}"><b>{s["headline"]}.</b> {s["advice"]}</div>',
        unsafe_allow_html=True,
    )


def risk_gauge(a: Dict[str, Any], prev_risk: Optional[float] = None) -> go.Figure:
    """Semi-circular gauge for the failure-risk index."""
    col = STATUS_STYLE[a["status"]]["color"]
    mode = "gauge+number" + ("+delta" if prev_risk is not None else "")
    fig = go.Figure(
        go.Indicator(
            mode=mode,
            value=a["risk"],
            number={"font": {"size": 54, "color": col}, "valueformat": ".0f"},
            delta={"reference": prev_risk or 0, "increasing": {"color": C["red"]}, "decreasing": {"color": C["green"]}, "valueformat": ".1f"},
            title={"text": "Failure risk index", "font": {"size": 15, "color": C["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": C["muted"]},
                "bar": {"color": col, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": rgba(C["green"], 0.18)},
                    {"range": [25, 60], "color": rgba(C["amber"], 0.18)},
                    {"range": [60, 100], "color": rgba(C["red"], 0.2)},
                ],
            },
        )
    )
    return style_fig(fig, 300, margin=dict(l=45, r=45, t=60, b=10))


def probability_bars(a: Dict[str, Any]) -> go.Figure:
    """Class-probability bars."""
    p = a["proba"]
    colors = [STATUS_STYLE[c]["color"] for c in CLASSES]
    fig = go.Figure(
        go.Bar(
            y=CLASSES,
            x=[p[c] * 100 for c in CLASSES],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{p[c]:.1%}" for c in CLASSES],
            textposition="auto",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 100], title="Probability (%)")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 200, title=dict(text="Class probabilities", font=dict(size=15, color=C["muted"])), margin=dict(l=10, r=10, t=45, b=35))


def trajectory_chart(traj: pd.DataFrame, cycle: int) -> go.Figure:
    """Stacked class-probability timeline with an 'actual status' strip and a moving cursor."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.04)
    for cls in CLASSES:
        col = STATUS_STYLE[cls]["color"]
        fig.add_trace(
            go.Scatter(
                x=traj["cycle"], y=traj[cls] * 100, name=cls, mode="lines", stackgroup="p",
                line=dict(width=0.6, color=col), fillcolor=rgba(col, 0.6),
                hovertemplate=f"{cls}: " + "%{y:.1f}%<extra></extra>",
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Bar(
            x=traj["cycle"], y=np.ones(len(traj)), showlegend=False,
            marker=dict(color=[STATUS_STYLE[s]["color"] for s in traj["Status"]], line=dict(width=0)),
            customdata=np.stack([traj["Status"], traj["RUL"]], axis=-1),
            hovertemplate="Actual: %{customdata[0]} (RUL %{customdata[1]})<extra></extra>",
        ),
        row=2, col=1,
    )
    last = int(traj["cycle"].max())
    if cycle < last:
        fig.add_vrect(x0=cycle, x1=last, fillcolor="rgba(11,18,32,.6)", line_width=0, layer="above", row="all", col="all")
    fig.add_vline(x=cycle, line=dict(color="white", width=2, dash="dot"), row="all", col="all")
    fig.update_yaxes(title="Probability (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title="Actual", showticklabels=False, range=[0, 1], row=2, col=1)
    fig.update_xaxes(title="Flight cycle", row=2, col=1)
    fig.update_layout(bargap=0, hovermode="x unified")
    return style_fig(fig, 400, title=dict(text="Health timeline — predicted probabilities vs actual status", font=dict(size=15, color=C["muted"])))


def sensor_trends(traj: pd.DataFrame, scaled: pd.DataFrame, sensors: list[str], cycle: int) -> go.Figure:
    """Scaled sensor traces that 'grow' as the replay advances."""
    palette = [C["cyan"], C["violet"], C["amber"], C["green"], C["red"], "#f472b6"]
    mask = traj["cycle"] <= cycle
    fig = go.Figure()
    for i, s in enumerate(sensors):
        col = palette[i % len(palette)]
        y = scaled.loc[mask, s]
        fig.add_trace(go.Scatter(x=traj.loc[mask, "cycle"], y=y, name=sensor_label(s), mode="lines", line=dict(width=2.2, color=col)))
        fig.add_trace(go.Scatter(x=[cycle], y=[y.iloc[-1]], mode="markers", showlegend=False, marker=dict(size=10, color=col, line=dict(color="white", width=1.5)), hoverinfo="skip"))
    fig.update_xaxes(range=[traj["cycle"].min(), traj["cycle"].max()], title="Flight cycle")
    fig.update_yaxes(range=[-0.02, 1.02], title="Scaled value (0–1)")
    return style_fig(fig, 340, title=dict(text="Sensor drift", font=dict(size=15, color=C["muted"])))


def contribution_bars(a: Dict[str, Any], top_n: int) -> go.Figure:
    """Horizontal bars: which sensors push the prediction away from 'healthy'."""
    df = a["contrib"].head(top_n).iloc[::-1]
    top = float(a["contrib"]["Score"].max())
    fig = go.Figure(
        go.Bar(
            y=[sensor_label(s) for s in df["Feature"]],
            x=df["Score"],
            orientation="h",
            marker=dict(color=df["Score"], colorscale=[[0, C["cyan"]], [0.5, C["violet"]], [1, C["red"]]], cmin=0, cmax=max(top, 0.1), line=dict(width=0)),
            text=[f"{'↑' if d > 0 else '↓'} {abs(p):.1f}%" for d, p in zip(df["Direction"], df["DeltaPct"])],
            textposition="outside",
            customdata=np.stack([df["Raw"], df["HealthyRaw"]], axis=-1),
            hovertemplate="%{y}<br>Score: %{x:.3f}<br>Value: %{customdata[0]:.2f} (healthy avg %{customdata[1]:.2f})<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, max(top * 1.3, 0.13)], title="Fault contribution score")
    return style_fig(fig, max(260, 52 * top_n), title=dict(text="Top fault-contributing sensors (label = shift vs healthy average)", font=dict(size=15, color=C["muted"])))


def fingerprint_radar(a: Dict[str, Any], bg: Dict[str, Any]) -> go.Figure:
    """Radar comparing the current reading with the average Healthy and Critical engine."""
    idx = [FEATURES.index(s) for s in SENSORS]
    labels = [sensor_label(s) for s in SENSORS]

    def trace(vals, name, color, fill, dash=None, width=2):
        v = list(np.clip(vals, 0, 1)) + [float(np.clip(vals[0], 0, 1))]
        return go.Scatterpolar(r=v, theta=labels + [labels[0]], name=name, fill=fill, fillcolor=rgba(color, 0.18), line=dict(color=color, width=width, dash=dash))

    fig = go.Figure()
    fig.add_trace(trace(bg["scaled_mean"]["Healthy"][idx], "Healthy average", C["green"], "toself", "dot"))
    fig.add_trace(trace(bg["scaled_mean"]["Critical"][idx], "Critical average", C["red"], "none", "dash"))
    fig.add_trace(trace(a["scaled"][idx], "Current reading", C["cyan"], "toself", None, 3))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,.18)", tickfont=dict(size=9)), angularaxis=dict(gridcolor="rgba(148,163,184,.18)", tickfont=dict(size=11))))
    return style_fig(fig, 420, title=dict(text="Sensor fingerprint", font=dict(size=15, color=C["muted"])), margin=dict(l=50, r=50, t=50, b=40))


def key_sensor_metrics(a: Dict[str, Any], bg: Dict[str, Any]) -> None:
    """st.metric row for the five key sensors, with delta vs the healthy average."""
    cols = st.columns(len(KEY_SENSORS))
    for col, s in zip(cols, KEY_SENSORS):
        i = FEATURES.index(s)
        val, base = a["vector"][i], bg["raw_mean"]["Healthy"][i]
        col.metric(sensor_label(s), f"{val:,.1f}", f"{100 * (val - base) / abs(base):+.2f}%", delta_color="off")


def pca_map(bg: Dict[str, Any], pca, a: Dict[str, Any], traj: Optional[pd.DataFrame] = None, cycle: Optional[int] = None) -> go.Figure:
    """Training data in PC1/PC2 space, with the current sample (and replay trail)."""
    ev = pca.explained_variance_ratio_ * 100
    fig = go.Figure()
    for cls, alpha, size in (("Healthy", 0.22, 5), ("Warning", 0.5, 6), ("Critical", 0.6, 6)):
        d = bg["points"][bg["points"]["Status"] == cls]
        fig.add_trace(go.Scattergl(x=d["PC1"], y=d["PC2"], mode="markers", name=f"{cls} (training)", marker=dict(size=size, color=rgba(STATUS_STYLE[cls]["color"], alpha)), hoverinfo="skip"))
    if traj is not None and cycle is not None:
        t = traj[traj["cycle"] <= cycle]
        fig.add_trace(go.Scatter(x=t["PC1"], y=t["PC2"], mode="lines", name="Engine trail", line=dict(color="rgba(255,255,255,.55)", width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[a["pcs"][0]], y=[a["pcs"][1]], mode="markers", name="Current", marker=dict(symbol="star", size=22, color="white", line=dict(color=STATUS_STYLE[a["status"]]["color"], width=3))))
    fig.update_xaxes(title=f"PC1 ({ev[0]:.1f}% of variance)")
    fig.update_yaxes(title=f"PC2 ({ev[1]:.1f}% of variance)")
    return style_fig(fig, 470, title=dict(text="Where this engine sits in sensor space", font=dict(size=15, color=C["muted"])))


def variance_chart(pca) -> go.Figure:
    """Explained-variance bars + cumulative line (interactive scree plot)."""
    ev = pca.explained_variance_ratio_ * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, name="Per component", marker=dict(color=C["cyan"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(ev))], y=np.cumsum(ev), name="Cumulative", mode="lines+markers", line=dict(color=C["amber"], width=3)), secondary_y=True)
    fig.update_yaxes(title="Variance (%)", secondary_y=False)
    fig.update_yaxes(title="Cumulative (%)", range=[0, 100], secondary_y=True, showgrid=False)
    return style_fig(fig, 340, title=dict(text="PCA variance explained", font=dict(size=15, color=C["muted"])))


def model_lab() -> None:
    """Interactive benchmark comparison + confusion matrices from ``artifacts/``."""
    try:
        df = pd.read_csv("artifacts/benchmark_metrics.csv")
    except Exception as exc:
        st.warning(f"Benchmark file not found: {exc}")
        return
    lat = [c for c in df.columns if "Latency" in c][0]
    df = df.rename(columns={lat: "Latency (ms / 1k samples)"})
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    metrics = [c for c in ("Test Accuracy", "Macro F1-Score", "Precision", "Recall") if c in df.columns]

    left, right = st.columns([1, 1])
    with left:
        metric = st.segmented_control("Compare models by", metrics, default="Macro F1-Score", key="lab_metric") or metrics[0]
        d = df.sort_values(metric)
        best = d[metric].idxmax()
        fig = go.Figure(go.Bar(y=d["Model Name"], x=d[metric], orientation="h", marker=dict(color=[C["cyan"] if i == best else rgba(C["muted"], 0.55) for i in d.index]), text=[f"{v:.3f}" for v in d[metric]], textposition="outside"))
        fig.update_xaxes(range=[0, 1.08])
        st.plotly_chart(style_fig(fig, 360, title=dict(text=f"{metric} by model", font=dict(size=15, color=C["muted"]))), key="lab_bar", config=PLOT_CFG)
    with right:
        st.markdown("&nbsp;")
        fig2 = go.Figure(
            go.Scatter(
                x=df["Latency (ms / 1k samples)"], y=df["Macro F1-Score"], mode="markers+text", text=df["Model Name"], textposition="top center",
                marker=dict(size=12 + 16 * (df["Test Accuracy"] - df["Test Accuracy"].min()) / max(df["Test Accuracy"].max() - df["Test Accuracy"].min(), 1e-9), color=C["violet"], opacity=0.75, line=dict(color="white", width=1)),
                hovertemplate="%{text}<br>Latency %{x:.2f} ms<br>Macro F1 %{y:.3f}<extra></extra>",
            )
        )
        fig2.update_xaxes(type="log", title="Inference latency per 1,000 samples (ms, log scale)")
        fig2.update_yaxes(title="Macro F1")
        st.plotly_chart(style_fig(fig2, 360, title=dict(text="Speed vs quality (bubble size = accuracy)", font=dict(size=15, color=C["muted"]))), key="lab_scatter", config=PLOT_CFG)

    st.caption("Accuracy sits around 98% for almost every model because ~85% of training rows are *Healthy*. Macro F1 treats all three classes equally, so it is the fairer yardstick here.")
    st.dataframe(
        df, hide_index=True,
        column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.3f") for c in ("Train Accuracy", "Test Accuracy", "Precision", "Recall", "Macro F1-Score") if c in df.columns},
    )
    with st.expander("Confusion matrices"):
        c1, c2 = st.columns(2)
        for col, (title, path) in zip((c1, c2), (("Random Forest", "artifacts/random_forest_confusion_matrix.png"), ("MLP", "artifacts/mlp_confusion_matrix.png"))):
            try:
                col.image(path, caption=title, width="stretch")
            except Exception:
                col.info(f"{title} confusion matrix not found.")
