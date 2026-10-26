import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
from src.explainability import get_top_fault_sensors
from src.ui_components import render_sidebar, render_metrics, render_plots

st.set_page_config(page_title="Turbofan Predictive Maintenance", layout="wide")

@st.cache_resource
def load_artifacts():
    scaler_path = os.path.join('artifacts', 'scaler.joblib')
    pca_path = os.path.join('artifacts', 'pca.joblib')
    model_path = os.path.join('artifacts', 'best_model.joblib')

    if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and os.path.exists(model_path)):
        st.error("Pipeline artifacts not found. Please run the serialization script first.")
        st.stop()

    try:
        scaler = joblib.load(scaler_path)
        pca_model = joblib.load(pca_path)
        rf_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

    return scaler, pca_model, rf_model

def main():
    st.title("✈️ Turbofan Predictive Maintenance Dashboard")

    # Inject custom CSS for metric cards
    st.markdown("""
    <style>
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin: 10px 0;
    }
    .metric-title {
        font-size: 1.1rem;
        color: #e0e0e0;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    scaler, pca_model, rf_model = load_artifacts()

    # The pipeline drops: 's_1', 's_5', 's_10', 's_16', 's_18', 's_19'
    all_sensors = [f's_{i}' for i in range(1, 22)]
    dropped_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    retained_sensors = [s for s in all_sensors if s not in dropped_sensors]
    feature_names = ['setting_1', 'setting_2', 'setting_3'] + retained_sensors

    # Load baseline presets
    try:
        presets_df = pd.read_csv('data/sample_test_cycles.csv')
    except Exception as e:
        st.error(f"Error loading presets data: {e}")
        st.stop()

    normal_preset = presets_df[presets_df['Preset'] == 'Normal'].iloc[0]

    sample_vector = render_sidebar(presets_df, normal_preset, feature_names)

    # Main panel logic
    if sample_vector is not None:
        tab1, tab2 = st.tabs(['Real-time Monitoring', 'Algorithm Explainability'])

        with st.spinner('Running prediction model...'):
            # Prediction
            scaled_sample = scaler.transform(sample_vector.reshape(1, -1))
            pca_sample = pca_model.transform(scaled_sample)

            pred_class = rf_model.predict(pca_sample)[0]
            pred_proba = rf_model.predict_proba(pca_sample)[0]
            confidence = np.max(pred_proba)

            if pred_class == "Healthy":
                status_color = "#2ecc71" # green
                rul_est = "> 30"
            elif pred_class == "Warning":
                status_color = "#f39c12" # orange
                rul_est = "16 - 30"
            else:
                status_color = "#e74c3c" # red
                rul_est = "≤ 15"

            with tab1:
                render_metrics(pred_class, rul_est, confidence, status_color)

            with tab2:
                # Explainability
                top_sensors = get_top_fault_sensors(sample_vector, top_n=5)
                render_plots(sample_vector, scaled_sample, pca_sample, scaler, pca_model, rf_model, feature_names, top_sensors)

        st.toast("Simulation complete!", icon="✅")

if __name__ == "__main__":
    main()
