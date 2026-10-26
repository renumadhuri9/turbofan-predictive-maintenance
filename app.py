import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.explainability import get_top_fault_sensors
from src.clustering import CustomKMeans

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
        tab1, tab2 = st.tabs(['Real-time Monitoring', 'Algorithm Explainability'])

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
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Health Status</div>
                    <div class="metric-value" style="color: {status_color};">{pred_class}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Estimated RUL (Cycles)</div>
                    <div class="metric-value">{rul_est}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Confidence Score</div>
                    <div class="metric-value">{confidence:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
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
            gini_importances = []
            raw_values = []
            for sensor in top_sensors:
                if sensor in feature_names:
                    idx = feature_names.index(sensor)
                    scores.append(anomaly_scores[idx])
                    gini_importances.append(original_importances[idx])
                    raw_values.append(sample_vector[idx])
                else:
                    scores.append(0.0)
                    gini_importances.append(0.0)
                    raw_values.append(0.0)

            bar_df = pd.DataFrame({
                'Sensor': top_sensors,
                'Anomaly Score': scores,
                'Gini Importance': gini_importances,
                'Raw Value': raw_values
            })

            fig_bar = px.bar(
                bar_df,
                x='Sensor',
                y='Anomaly Score',
                hover_data=['Gini Importance', 'Raw Value'],
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

                    # Cluster baseline data
                    kmeans = CustomKMeans(n_clusters=3, random_state=42)
                    kmeans.fit(X_base_pca[:, :2])
                    cluster_labels = kmeans.predict(X_base_pca[:, :2])

                    df_pca = pd.DataFrame(X_base_pca[:, :2], columns=['PCA Component 1', 'PCA Component 2'])
                    df_pca['Cluster'] = cluster_labels
                    df_pca['Status'] = train_data['hazard_class']

                    # Map clusters to approximate health states based on the dominant class (optional context) or just color them uniquely
                    # To keep it simple, we'll assign distinct colors to clusters and name them Cluster 0, 1, 2
                    cluster_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
                    cluster_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}

                    fig_scatter = go.Figure()

                    # Add trace for each cluster
                    for cluster_id in sorted(df_pca['Cluster'].unique()):
                        cluster_data = df_pca[df_pca['Cluster'] == cluster_id]
                        fig_scatter.add_trace(go.Scatter(
                            x=cluster_data['PCA Component 1'],
                            y=cluster_data['PCA Component 2'],
                            mode='markers',
                            marker=dict(color=cluster_colors.get(cluster_id, 'grey'), size=6, opacity=0.6),
                            name=cluster_names.get(cluster_id, f'Cluster {cluster_id}')
                        ))

                    # Overlay current sample
                    fig_scatter.add_trace(go.Scatter(
                        x=[pca_sample[0, 0]],
                        y=[pca_sample[0, 1]],
                        mode='markers',
                        marker=dict(size=15, color='white', line=dict(color='black', width=2), symbol='star'),
                        name='Current Sample'
                    ))

                    fig_scatter.update_layout(
                        title="PCA Projection (Clustered Baseline vs Current Sample)",
                        xaxis_title="PCA Component 1",
                        yaxis_title="PCA Component 2",
                        dragmode="zoom",
                        showlegend=True
                    )

                    st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load baseline PCA projection: {e}")

if __name__ == "__main__":
    main()
