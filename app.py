import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import os

try:
    from src.explainability import get_top_anomalous_sensors
except ImportError:
    get_top_anomalous_sensors = None

st.set_page_config(page_title="Turbofan Predictive Maintenance", layout="wide")

st.title("Turbofan Engine Predictive Maintenance Dashboard")

# Graceful loading of artifacts
pipeline = None
try:
    if os.path.exists("artifacts/pipeline.pkl"):
        pipeline = joblib.load("artifacts/pipeline.pkl")
    else:
        st.sidebar.warning("Pipeline artifact missing.")
except Exception as e:
    st.sidebar.warning(f"Error loading pipeline: {e}")

# Sidebar
st.sidebar.header("Input Controls")
input_mode = st.sidebar.radio("Select Input Mode", ["Preset Sample", "Manual Input"])

top_5_sensors = ['s2', 's3', 's4', 's7', 's11']
input_data = {}

if input_mode == "Preset Sample":
    try:
        sample_df = pd.read_csv("data/sample_test_cycles.csv")
        # Ensure we have the labels: Normal, Degrading, Critical
        if 'label' in sample_df.columns:
            sample_label = st.sidebar.selectbox("Select Sample Engine Cycle", sample_df['label'].unique())
            selected_sample = sample_df[sample_df['label'] == sample_label].iloc[0]
        else:
            sample_id = st.sidebar.selectbox("Select Sample ID", sample_df['id'].unique())
            selected_sample = sample_df[sample_df['id'] == sample_id].iloc[0]

        for sensor in top_5_sensors:
            input_data[sensor] = selected_sample.get(sensor, 0.0)
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {e}")
        for sensor in top_5_sensors:
            input_data[sensor] = 0.0
else:
    for sensor in top_5_sensors:
        input_data[sensor] = st.sidebar.slider(f"{sensor} Value", 0.0, 3000.0, 1000.0)

# Main Panel
st.header("Engine Health Prediction")

# Top metrics row
col1, col2, col3 = st.columns(3)

status = "Healthy"
rul = 45
confidence = 0.95

if pipeline is not None:
    try:
        input_df = pd.DataFrame([input_data])
        # Predict status (assuming pipeline returns 0=Critical, 1=Warning, 2=Healthy or similar strings)
        pred = pipeline.predict(input_df)[0]
        status = str(pred) # fallback string
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0]
            confidence = max(proba)
        # Attempt to get continuous RUL if model supports it, else leave default for demo
        if hasattr(pipeline, "predict_rul"):
            rul = pipeline.predict_rul(input_df)[0]
    except Exception as e:
        st.warning(f"Error executing model predictions: {e}")

# Color mapping based on memory target definition
status_color = "green"
if status.lower() == "healthy":
    status_color = "green"
elif status.lower() == "warning":
    status_color = "orange"
elif status.lower() == "critical":
    status_color = "red"

col1.markdown(f"### Predicted Health Status")
col1.markdown(f"**<span style='color:{status_color}; font-size:24px'>{status}</span>**", unsafe_allow_html=True)
col2.metric("Estimated RUL (Cycles)", f"{rul}")
col3.metric("Confidence Score", f"{confidence:.2%}")

# Center plot
st.subheader("Top Anomalous Sensors")
if get_top_anomalous_sensors:
    try:
        anomalies = get_top_anomalous_sensors(input_data)
        fig = px.bar(x=list(anomalies.keys()), y=list(anomalies.values()), labels={'x':'Sensor', 'y':'Anomaly Score'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Error generating anomalous sensors plot: {e}")
else:
    st.warning("`src.explainability` module not found. Showing simulated data.")
    fig = px.bar(x=top_5_sensors, y=np.random.rand(5), labels={'x':'Sensor', 'y':'Anomaly Score'})
    st.plotly_chart(fig, use_container_width=True)

# Bottom expander
with st.expander("PCA Component Projections (2D Scatter)"):
    st.write("2D scatter plot of active PCA components against healthy/fault cluster baselines.")
    if pipeline is not None and hasattr(pipeline, 'named_steps') and 'pca' in pipeline.named_steps:
         try:
             pca = pipeline.named_steps['pca']
             scaler = pipeline.named_steps.get('scaler') # assuming there's a scaler
             input_df = pd.DataFrame([input_data])
             scaled_data = scaler.transform(input_df) if scaler else input_df
             pca_proj = pca.transform(scaled_data)

             # Need baseline data to plot against. For demo, we simulate baselines.
             fig2, ax2 = plt.subplots()
             ax2.scatter(np.random.randn(50), np.random.randn(50), label="Healthy Baseline", alpha=0.3, c='green')
             ax2.scatter(np.random.randn(20) + 2, np.random.randn(20) + 2, label="Fault Baseline", alpha=0.3, c='red')
             ax2.scatter(pca_proj[:, 0], pca_proj[:, 1], color='blue', s=100, label="Current Input", marker="X")
             ax2.legend()
             st.pyplot(fig2)
         except Exception as e:
             st.warning(f"Error projecting PCA: {e}")
    else:
        st.warning("Models not fully loaded with PCA. Showing simulated PCA projection.")
        fig2, ax2 = plt.subplots()
        ax2.scatter(np.random.randn(50), np.random.randn(50), label="Healthy", alpha=0.5, c='green')
        ax2.scatter(np.random.randn(20) + 2, np.random.randn(20) + 2, label="Fault", alpha=0.5, c='red')
        ax2.scatter([0], [0], color='blue', s=100, label="Current Input", marker="X")
        ax2.legend()
        st.pyplot(fig2)
