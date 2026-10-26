"""
Integration tests for the inference pipeline and explainability module.
"""

import numpy as np
import joblib
import pytest
from src.explainability import get_top_fault_sensors
import os

def test_inference():
    """
    End-to-end integration test verifying that a raw feature vector
    can be passed through the saved .joblib files to produce a
    valid prediction and fault ranking.
    """
    # Verify artifacts exist
    assert os.path.exists('artifacts/scaler.joblib')
    assert os.path.exists('artifacts/pca.joblib')
    assert os.path.exists('artifacts/best_model.joblib')

    # Load artifacts
    scaler = joblib.load('artifacts/scaler.joblib')
    pca_model = joblib.load('artifacts/pca.joblib')
    rf_model = joblib.load('artifacts/best_model.joblib')

    # In preprocessing, the pipeline drops 6 sensors and keeps 3 settings + 15 sensors = 18 features
    # The test requires verifying a raw 21-sensor array vector.
    raw_21_sensors = np.random.rand(21)

    # Map raw 21 sensors to the 18 features expected by the pipeline
    # (Settings 1-3, followed by the retained sensors)
    dropped_indices = [0, 4, 9, 15, 17, 18] # corresponds to s_1, s_5, s_10, s_16, s_18, s_19
    retained_sensors = [val for i, val in enumerate(raw_21_sensors) if i not in dropped_indices]

    settings = [0.0, 0.0, 0.0]
    sample_vector = np.array(settings + retained_sensors)

    # 1. Preprocess using scaler and PCA
    scaled_sample = scaler.transform(sample_vector.reshape(1, -1))
    pca_sample = pca_model.transform(scaled_sample)

    # 2. Prediction
    prediction = rf_model.predict(pca_sample)

    # Valid prediction classes are ['Healthy', 'Warning', 'Critical'] (or 'Unknown')
    assert prediction[0] in ['Healthy', 'Warning', 'Critical', 'Unknown']

    # 3. Fault ranking
    top_sensors = get_top_fault_sensors(sample_vector, top_n=3)

    # Assertions
    assert len(top_sensors) == 3
    for sensor in top_sensors:
        assert sensor.startswith('s_')
