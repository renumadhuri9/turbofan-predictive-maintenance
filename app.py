import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.explainability import get_top_fault_sensors

st.set_page_config(page_title="Turbofan Predictive Maintenance", layout="wide")

@st.cache_resource
def load_artifacts():
    scaler_path = os.path.join('artifacts', 'scaler.joblib')
    pca_path = os.path.join('artifacts', 'pca.joblib')
    model_path = os.path.join('artifacts', 'best_model.joblib')

    if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and os.path.exists(model_path)):
        raise FileNotFoundError("Pipeline artifacts not found. Please run the serialization script first.")

    scaler = joblib.load(scaler_path)
    pca_model = joblib.load(pca_path)
    rf_model = joblib.load(model_path)

    return scaler, pca_model, rf_model

def main():
    st.title("✈️ Turbofan Predictive Maintenance Dashboard")

    try:
        scaler, pca_model, rf_model = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading artifacts: {e}")
        st.stop()

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

    st.sidebar.header("Data Source Configuration")
    mode = st.sidebar.radio("Input Mode", ["Preset Sample", "Manual Mode"])

    sample_vector = None

    if mode == "Preset Sample":
        preset_choice = st.sidebar.selectbox("Select Engine Status", ["Normal", "Degrading", "Critical"])
        selected_row = presets_df[presets_df['Preset'] == preset_choice].iloc[0]
        sample_vector = selected_row[feature_names].values.astype(np.float64)
    else:
        st.sidebar.subheader("Adjust High-Variance Sensors")
        # Base on the mean of the "Normal" preset for defaults
        normal_preset = presets_df[presets_df['Preset'] == 'Normal'].iloc[0]

        s_4_val = st.sidebar.slider("Sensor 4", min_value=1380.0, max_value=1445.0, value=float(normal_preset['s_4']))
        s_9_val = st.sidebar.slider("Sensor 9", min_value=9020.0, max_value=9245.0, value=float(normal_preset['s_9']))
        s_14_val = st.sidebar.slider("Sensor 14", min_value=8099.0, max_value=8295.0, value=float(normal_preset['s_14']))
        s_3_val = st.sidebar.slider("Sensor 3", min_value=1570.0, max_value=1620.0, value=float(normal_preset['s_3']))
        s_17_val = st.sidebar.slider("Sensor 17", min_value=388.0, max_value=400.0, value=float(normal_preset['s_17']))

        # Construct vector
        manual_vector = []
        for feat in feature_names:
            if feat == 's_4':
                manual_vector.append(s_4_val)
            elif feat == 's_9':
                manual_vector.append(s_9_val)
            elif feat == 's_14':
                manual_vector.append(s_14_val)
            elif feat == 's_3':
                manual_vector.append(s_3_val)
            elif feat == 's_17':
                manual_vector.append(s_17_val)
            else:
                manual_vector.append(float(normal_preset[feat]))

        sample_vector = np.array(manual_vector, dtype=np.float64)

    # Main panel logic
    if sample_vector is not None:
        # Prediction
        scaled_sample = scaler.transform(sample_vector.reshape(1, -1))
        pca_sample = pca_model.transform(scaled_sample)

        pred_class = rf_model.predict(pca_sample)[0]
        pred_proba = rf_model.predict_proba(pca_sample)[0]
        confidence = np.max(pred_proba)

        if pred_class == "Healthy":
            status_color = "green"
            rul_est = "> 30"
        elif pred_class == "Warning":
            status_color = "orange"
            rul_est = "16 - 30"
        else:
            status_color = "red"
            rul_est = "≤ 15"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"### Health Status")
            st.markdown(f"<h2 style='color: {status_color};'>{pred_class}</h2>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"### Estimated RUL (Cycles)")
            st.markdown(f"<h2>{rul_est}</h2>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"### Confidence Score")
            st.markdown(f"<h2>{confidence:.2%}</h2>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Top Anomalous Sensors")

        # Explainability
        # get_top_fault_sensors returns a list of string features
        top_sensors = get_top_fault_sensors(sample_vector, top_n=5)

        # We want to display the anomaly score for the plot, so we re-compute them for the bar chart
        rf_importances = rf_model.feature_importances_
        pca_components = pca_model.components_
        original_importances = np.dot(rf_importances, np.abs(pca_components))
        scaled_sample_flat = scaled_sample.flatten()
        anomaly_scores = np.abs(scaled_sample_flat) * original_importances

        # Map back scores for the top sensors
        scores = []
        for sensor in top_sensors:
            if sensor in feature_names:
                idx = feature_names.index(sensor)
                scores.append(anomaly_scores[idx])
            else:
                scores.append(0.0)

        fig_bar = px.bar(
            x=top_sensors,
            y=scores,
            labels={'x': 'Sensor', 'y': 'Anomaly Score'},
            title="Sensor Anomaly Scores"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        with st.expander("View PCA Component Projections"):
            # Sample data for scatter plot baselines
            try:
                # Get a slightly larger sample from training data for context
                train_data = pd.read_csv('data/raw/train_FD001.txt', sep=r'\s+', header=None, nrows=1000)
                train_data.columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]

                # Get max cycle per engine to calculate RUL
                max_cycles = train_data.groupby('engine_id')['cycle'].max()
                train_data['RUL'] = train_data.apply(lambda row: max_cycles[row['engine_id']] - row['cycle'], axis=1)

                # Add hazard class
                conditions = [
                    (train_data['RUL'] > 30),
                    (train_data['RUL'] > 15) & (train_data['RUL'] <= 30),
                    (train_data['RUL'] <= 15)
                ]
                train_data['hazard_class'] = np.select(conditions, ['Healthy', 'Warning', 'Critical'], default='Unknown')

                # Transform baseline
                X_base = train_data[feature_names]
                X_base_scaled = scaler.transform(X_base)
                X_base_pca = pca_model.transform(X_base_scaled)

                df_pca = pd.DataFrame(X_base_pca[:, :2], columns=['PCA Component 1', 'PCA Component 2'])
                df_pca['Status'] = train_data['hazard_class']

                fig_scatter = px.scatter(
                    df_pca,
                    x='PCA Component 1',
                    y='PCA Component 2',
                    color='Status',
                    color_discrete_map={'Healthy': 'green', 'Warning': 'orange', 'Critical': 'red'},
                    title="PCA Projection (Baseline vs Current Sample)",
                    opacity=0.6
                )

                # Overlay current sample
                fig_scatter.add_trace(go.Scatter(
                    x=[pca_sample[0, 0]],
                    y=[pca_sample[0, 1]],
                    mode='markers',
                    marker=dict(size=15, color='black', symbol='star'),
                    name='Current Sample'
                ))

                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load baseline PCA projection: {e}")

if __name__ == "__main__":
    main()
