"""
Module for model explainability and physical sensor fault isolation.
"""

import os
import joblib
import numpy as np
from typing import List, Optional

def load_artifacts():
    """
    Load the saved pipeline artifacts (Scaler, PCA, Best Model).

    Returns
    -------
    tuple
        (scaler, pca_model, rf_model)
    """
    scaler_path = os.path.join('artifacts', 'scaler.joblib')
    pca_path = os.path.join('artifacts', 'pca.joblib')
    model_path = os.path.join('artifacts', 'best_model.joblib')

    if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and os.path.exists(model_path)):
        raise FileNotFoundError("Pipeline artifacts not found. Run serialize_pipeline.py first.")

    scaler = joblib.load(scaler_path)
    pca_model = joblib.load(pca_path)
    rf_model = joblib.load(model_path)

    return scaler, pca_model, rf_model

def get_top_fault_sensors(sample_vector: np.ndarray, top_n: int = 3) -> List[str]:
    """
    Projects Gini feature importances from the Random Forest model back through
    the PCA component loadings matrix to assign an anomaly score to original
    physical sensor channels. Returns the top N sensor names contributing most
    heavily to an abnormal health status.

    Parameters
    ----------
    sample_vector : np.ndarray
        A 1D numpy array representing a single sample. Must match the feature
        dimension the MinMaxScaler was fitted on (e.g., 3 settings + 15 sensors = 18 features).
    top_n : int, optional
        The number of top fault sensors to return, by default 3.

    Returns
    -------
    List[str]
        List of the top `top_n` sensor names contributing to the fault.
    """
    scaler, pca_model, rf_model = load_artifacts()

    # 1. Project RF feature importances through PCA components to original features
    # RF importances shape: (n_components,)
    # PCA components shape: (n_components, n_original_features)
    rf_importances = rf_model.feature_importances_
    pca_components = pca_model.components_

    # original_importances shape: (n_original_features,)
    original_importances = np.dot(rf_importances, np.abs(pca_components))

    # 2. Scale the sample vector to match training preprocessing
    # sample_vector needs to be reshaped for the scaler: (1, n_features)
    scaled_sample = scaler.transform(sample_vector.reshape(1, -1)).flatten()

    # 3. Calculate anomaly score for each feature
    anomaly_scores = np.abs(scaled_sample) * original_importances

    # The pipeline drops: 's_1', 's_5', 's_10', 's_16', 's_18', 's_19'
    # Original sensors (1 to 21)
    # The features the scaler sees: 'setting_1', 'setting_2', 'setting_3' followed by the remaining sensors
    all_sensors = [f's_{i}' for i in range(1, 22)]
    dropped_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    retained_sensors = [s for s in all_sensors if s not in dropped_sensors]
    feature_names = ['setting_1', 'setting_2', 'setting_3'] + retained_sensors

    # Get indices of top N highest anomaly scores
    # argsort sorts ascending, so we take the last `top_n` and reverse
    top_indices = np.argsort(anomaly_scores)[-top_n:][::-1]

    # Map indices back to feature names (sensors only, or just feature names)
    # The instructions say "Returns the top N sensor names"
    # Filter to ensure we only return sensor names if a setting happens to be highly anomalous
    top_sensors = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        if feat_name.startswith('s_'):
            top_sensors.append(feat_name)
            if len(top_sensors) == top_n:
                break

    # If we didn't find enough sensors, keep looking
    if len(top_sensors) < top_n:
        for idx in np.argsort(anomaly_scores)[::-1]:
            feat_name = feature_names[idx]
            if feat_name.startswith('s_') and feat_name not in top_sensors:
                top_sensors.append(feat_name)
                if len(top_sensors) == top_n:
                    break

    return top_sensors
