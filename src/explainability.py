import joblib
import numpy as np
from typing import List

def get_top_fault_sensors(sample_vector: np.ndarray, top_n: int = 3) -> List[str]:
    """
    Extracts Random Forest feature importances, maps them back to the original
    21 sensors using PCA loadings, and returns the names of the most anomalous sensors.

    Parameters
    ----------
    sample_vector : np.ndarray
        A NumPy array representing a single sample's 21 sensor values.
        Shape should be (1, 21) or (21,).
    top_n : int, optional
        The number of top anomalous sensors to return (default is 3).

    Returns
    -------
    List[str]
        A list of the top N sensor names (e.g., ["sensor_1", "sensor_5", "sensor_3"]).
    """
    # Ensure sample_vector is 2D
    if sample_vector.ndim == 1:
        sample_vector = sample_vector.reshape(1, -1)

    if sample_vector.shape[1] != 21:
        raise ValueError(f"Expected 21 sensors, got {sample_vector.shape[1]}")

    # Load artifacts
    try:
        scaler = joblib.load('artifacts/scaler.joblib')
        pca = joblib.load('artifacts/pca.joblib')
        rf_model = joblib.load('artifacts/best_model.joblib')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing artifact: {e}. Run serialize_pipeline.py first.")

    # 1. Scale the input sample
    sample_scaled = scaler.transform(sample_vector)

    # 2. Get RF feature importances (shape: (n_components,))
    rf_importances = rf_model.feature_importances_

    # 3. Project feature importances back to original 21 sensors
    # PCA components_ shape is (n_components, 21)
    # Projecting: np.dot(rf_importances, np.abs(pca.components_)) -> shape (21,)
    global_importances = np.dot(rf_importances, np.abs(pca.components_))

    # 4. Calculate anomalousness for this specific sample
    # Here, we simply weight the absolute scaled values by the global feature importance
    # Alternative ways could be deviation from mean, but since MinMax scales to [0,1],
    # we can use the raw scaled values or deviation from the expected 'healthy' state.
    # For now, multiplying scaled values by importance gives a good proxy.
    sample_anomalousness = global_importances * np.abs(sample_scaled[0])

    # 5. Rank sensors
    sensor_names = [f"sensor_{i}" for i in range(1, 22)]

    # Get indices of top_n largest anomalousness scores
    top_indices = np.argsort(sample_anomalousness)[::-1][:top_n]

    top_sensors = [sensor_names[i] for i in top_indices]

    return top_sensors
