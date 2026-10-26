import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.clustering import CustomKMeans

def render_sidebar(presets_df, normal_preset, feature_names):
    # Initialize session state for high-variance sensors
    if 's_14' not in st.session_state:
        st.session_state['s_14'] = float(normal_preset['s_14'])
    if 's_4' not in st.session_state:
        st.session_state['s_4'] = float(normal_preset['s_4'])
    if 's_9' not in st.session_state:
        st.session_state['s_9'] = float(normal_preset['s_9'])
    if 's_3' not in st.session_state:
        st.session_state['s_3'] = float(normal_preset['s_3'])
    if 's_17' not in st.session_state:
        st.session_state['s_17'] = float(normal_preset['s_17'])

    st.sidebar.header("Data Source Configuration")
    what_if_simulation = st.sidebar.toggle("What-If Simulation", value=False)

    sample_vector = None

    if not what_if_simulation:
        # Existing Selectbox behavior for normal presets
        preset_choice = st.sidebar.selectbox("Select Engine Status", ["Normal", "Degrading", "Critical"])
        selected_row = presets_df[presets_df['Preset'] == preset_choice].iloc[0]
        sample_vector = selected_row[feature_names].values.astype(np.float64)
    else:
        st.sidebar.subheader("What-If Analysis")

        # Reset Button functionality
        def reset_sliders():
            st.session_state['s_14'] = float(normal_preset['s_14'])
            st.session_state['s_4'] = float(normal_preset['s_4'])
            st.session_state['s_9'] = float(normal_preset['s_9'])
            st.session_state['s_3'] = float(normal_preset['s_3'])
            st.session_state['s_17'] = float(normal_preset['s_17'])

        st.sidebar.button("Reset", on_click=reset_sliders)

        # Sliders bound directly to session state
        st.sidebar.slider("Sensor 14", min_value=8099.0, max_value=8295.0, key="s_14")
        st.sidebar.slider("Sensor 4", min_value=1380.0, max_value=1445.0, key="s_4")
        st.sidebar.slider("Sensor 9", min_value=9020.0, max_value=9245.0, key="s_9")
        st.sidebar.slider("Sensor 3", min_value=1570.0, max_value=1620.0, key="s_3")
        st.sidebar.slider("Sensor 17", min_value=388.0, max_value=400.0, key="s_17")

        # Construct vector using state values for modified sensors
        manual_vector = []
        for feat in feature_names:
            if feat in ['s_14', 's_4', 's_9', 's_3', 's_17']:
                manual_vector.append(st.session_state[feat])
            else:
                manual_vector.append(float(normal_preset[feat]))

        sample_vector = np.array(manual_vector, dtype=np.float64)

    return sample_vector


def render_metrics(pred_class, rul_est, confidence, status_color):
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


def render_plots(sample_vector, scaled_sample, pca_sample, scaler, pca_model, rf_model, feature_names, top_sensors):
    st.subheader("Top Anomalous Sensors")

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
