import joblib
import numpy as np
import pytest

from src.explainability import get_top_fault_sensors
from src.models import load_data, preprocess_data

def test_inference_pipeline():
    """
    End-to-end integration test verifying that a raw 21-sensor array vector
    can be passed through the saved .joblib files to produce a valid prediction
    and fault ranking.
    """
    # 1. Get a realistic raw 21-sensor array vector from the dataset
    # We can load a small portion or just take one sample from the loaded data
    train_df, test_df, true_rul = load_data()
    X_train_full, X_test_full, y_train, y_test = preprocess_data(train_df, test_df, true_rul)

    # Extract 21 sensor columns (drop setting_1, setting_2, setting_3)
    X_test_sensors = X_test_full[:, 3:]
    assert X_test_sensors.shape[1] == 21

    # Take the first sample
    sample_vector = X_test_sensors[0]

    # 2. Load the artifacts
    scaler = joblib.load('artifacts/scaler.joblib')
    pca = joblib.load('artifacts/pca.joblib')
    rf_model = joblib.load('artifacts/best_model.joblib')

    # 3. Transform the vector
    sample_2d = sample_vector.reshape(1, -1)
    sample_scaled = scaler.transform(sample_2d)
    sample_pca = pca.transform(sample_scaled)

    # 4. Predict health status
    prediction = rf_model.predict(sample_pca)

    # Assert prediction is valid (0=Healthy, 1=Warning, 2=Critical)
    assert prediction[0] in [0, 1, 2], f"Invalid prediction: {prediction[0]}"

    # 5. Call explainability module
    top_sensors = get_top_fault_sensors(sample_vector, top_n=3)

    # 6. Assertions for explainability
    assert isinstance(top_sensors, list), "Expected a list of sensor names"
    assert len(top_sensors) == 3, f"Expected 3 sensors, got {len(top_sensors)}"
    for sensor in top_sensors:
        assert isinstance(sensor, str), f"Expected sensor name to be string, got {type(sensor)}"
        assert sensor.startswith("sensor_"), f"Expected sensor name to start with 'sensor_', got {sensor}"
