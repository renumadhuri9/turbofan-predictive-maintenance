"""Cached data loading and scoring helpers for the Streamlit dashboard."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAW_COLS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]
DROPPED = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
SENSORS = [f"s_{i}" for i in range(1, 22) if f"s_{i}" not in DROPPED]
FEATURES = ["setting_1", "setting_2", "setting_3"] + SENSORS
CLASSES = ["Healthy", "Warning", "Critical"]
KEY_SENSORS = ["s_14", "s_4", "s_9", "s_3", "s_17"]


def hazard_from_rul(rul: pd.Series) -> np.ndarray:
    """Map RUL values to Healthy / Warning / Critical labels."""
    r = np.asarray(rul)
    return np.select([r > 30, r > 15], ["Healthy", "Warning"], default="Critical")


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any, Any, Any]:
    """Load scaler, PCA and classifier from ``artifacts/`` (stops the app with a message on failure)."""
    paths = [os.path.join("artifacts", f) for f in ("scaler.joblib", "pca.joblib", "best_model.joblib")]
    if not all(os.path.exists(p) for p in paths):
        st.error("Pipeline artifacts not found. Run the serialization step first (see README).")
        st.stop()
    try:
        return tuple(joblib.load(p) for p in paths)  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.error(f"Failed to load artifacts: {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_presets() -> pd.DataFrame:
    """Load the bundled Normal / Degrading / Critical sample cycles."""
    return pd.read_csv("data/sample_test_cycles.csv")


@st.cache_data(show_spinner="Building sensor-space baseline…")
def load_background(_scaler, _pca, n_healthy: int = 2500) -> Dict[str, Any]:
    """Project the training set into PCA space and compute per-class baselines.

    Keeps every Warning/Critical row but only a sample of Healthy rows so the
    scatter plot stays readable and fast.
    """
    df = pd.read_csv("data/raw/train_FD001.txt", sep=r"\s+", header=None, names=RAW_COLS)
    df["RUL"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    df["Status"] = hazard_from_rul(df["RUL"])

    scaled = _scaler.transform(df[FEATURES].to_numpy(float))
    pcs = _pca.transform(scaled)

    masks = {c: (df["Status"] == c).to_numpy() for c in CLASSES}
    raw_mean = {c: df.loc[masks[c], FEATURES].mean().to_numpy() for c in CLASSES}
    scaled_mean = {c: scaled[masks[c]].mean(axis=0) for c in CLASSES}

    rng = np.random.default_rng(42)
    healthy_idx = rng.choice(np.where(masks["Healthy"])[0], size=n_healthy, replace=False)
    keep = np.sort(np.concatenate([healthy_idx, np.where(~masks["Healthy"])[0]]))
    points = pd.DataFrame({"PC1": pcs[keep, 0], "PC2": pcs[keep, 1], "Status": df["Status"].to_numpy()[keep]})

    ranges = pd.DataFrame({"min": df[FEATURES].min(), "max": df[FEATURES].max()})
    return {"points": points, "raw_mean": raw_mean, "scaled_mean": scaled_mean, "ranges": ranges}


@st.cache_data(show_spinner=False)
def load_test_engines() -> pd.DataFrame:
    """Load the FD001 test set with ground-truth RUL (clipped at 125) for every cycle."""
    te = pd.read_csv("data/raw/test_FD001.txt", sep=r"\s+", header=None, names=RAW_COLS)
    rul_last = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0].to_numpy()
    max_c = te.groupby("engine_id")["cycle"].transform("max")
    last = te["engine_id"].map(dict(enumerate(rul_last, start=1)))
    te["RUL"] = (last + max_c - te["cycle"]).clip(upper=125)
    te["Status"] = hazard_from_rul(te["RUL"])
    return te


def feature_weights(model, pca) -> np.ndarray:
    """Random-Forest importances projected back to the original features through |PCA loadings|."""
    return model.feature_importances_ @ np.abs(pca.components_)


def risk_index(proba: pd.DataFrame | Dict[str, float]) -> Any:
    """Single 0-100 number: P(Critical) counts fully, P(Warning) counts half."""
    return 100.0 * (0.5 * proba["Warning"] + proba["Critical"])


def _proba_frame(model, pcs: np.ndarray) -> pd.DataFrame:
    p = pd.DataFrame(model.predict_proba(pcs), columns=model.classes_)
    return p[CLASSES]


@st.cache_data(show_spinner="Scoring every cycle of this engine…")
def score_engine(engine_id: int, _scaler, _pca, _model) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every cycle of one test engine.

    Returns
    -------
    traj : DataFrame
        cycle, true RUL/status, class probabilities, prediction, risk, PC1/PC2 and raw features.
    scaled : DataFrame
        MinMax-scaled feature matrix (same row order).
    """
    te = load_test_engines()
    sub = te[te["engine_id"] == engine_id].reset_index(drop=True)
    scaled = _scaler.transform(sub[FEATURES].to_numpy(float))
    pcs = _pca.transform(scaled)
    proba = _proba_frame(_model, pcs)
    traj = sub[["cycle", "RUL", "Status"] + FEATURES].copy()
    traj[CLASSES] = proba.to_numpy()
    traj["Pred"] = proba.idxmax(axis=1).to_numpy()
    traj["Risk"] = risk_index(proba).to_numpy()
    traj["PC1"], traj["PC2"] = pcs[:, 0], pcs[:, 1]
    return traj, pd.DataFrame(scaled, columns=FEATURES)


def assess(vector: np.ndarray, art: Tuple[Any, Any, Any], bg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one raw feature vector through scaler → PCA → classifier and build all UI quantities."""
    scaler, pca, model = art
    scaled = scaler.transform(vector.reshape(1, -1))
    pcs = pca.transform(scaled)
    proba = _proba_frame(model, pcs).iloc[0]
    status = str(proba.idxmax())

    weights = feature_weights(model, pca)
    dev = scaled[0] - bg["scaled_mean"]["Healthy"]
    contrib = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Score": np.abs(dev) * weights,
            "Direction": np.sign(dev),
            "Raw": vector,
            "HealthyRaw": bg["raw_mean"]["Healthy"],
        }
    )
    contrib = contrib[contrib["Feature"].str.startswith("s_")]
    contrib["DeltaPct"] = 100.0 * (contrib["Raw"] - contrib["HealthyRaw"]) / contrib["HealthyRaw"].abs()
    return {
        "status": status,
        "confidence": float(proba.max()),
        "proba": proba,
        "risk": float(risk_index(proba)),
        "scaled": scaled[0],
        "pcs": pcs[0],
        "contrib": contrib.sort_values("Score", ascending=False).reset_index(drop=True),
        "vector": vector,
    }
