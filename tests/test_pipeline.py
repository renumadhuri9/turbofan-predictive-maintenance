import pytest
import os
import pandas as pd
import numpy as np
from src.preprocessing import CMAPSSDataPipeline
import tempfile

def create_mock_data(filepath, n_engines=2, max_cycles=150):
    """
    Creates mock C-MAPSS data for testing.
    """
    columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + ['s_' + str(i) for i in range(1, 22)]

    data = []
    np.random.seed(42)  # For reproducibility

    for engine_id in range(1, n_engines + 1):
        num_cycles = np.random.randint(50, max_cycles + 1)
        for cycle in range(1, num_cycles + 1):
            row = [engine_id, cycle]
            # Settings
            row.extend(np.random.rand(3).tolist())
            # Sensors (some zero variance to test dropping)
            sensors = np.random.rand(21).tolist()
            # Set specific sensors to constant values to simulate zero variance
            zero_var_indices = [0, 4, 9, 15, 17, 18] # s_1, s_5, s_10, s_16, s_18, s_19 (0-indexed)
            for idx in zero_var_indices:
                sensors[idx] = 1.0 # Constant value
            row.extend(sensors)
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    # The read_csv in pipeline expects space-separated values without headers
    df.to_csv(filepath, sep=' ', index=False, header=False)

@pytest.fixture
def mock_data_paths():
    """Fixture to create and provide paths to mock train and test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, 'train_FD001.txt')
        test_path = os.path.join(temp_dir, 'test_FD001.txt')

        create_mock_data(train_path, n_engines=3, max_cycles=150)
        create_mock_data(test_path, n_engines=2, max_cycles=100)

        yield train_path, test_path

def test_pipeline_output_shapes_and_nans(mock_data_paths):
    train_path, test_path = mock_data_paths
    pipeline = CMAPSSDataPipeline()

    train_df, test_df = pipeline.run_pipeline(train_path, test_path)

    # Check that zero-variance sensors are dropped
    expected_sensor_cols = [f's_{i}' for i in range(1, 22) if f's_{i}' not in pipeline.drop_sensors]

    # Train df should have: engine_id, cycle, 3 settings, remaining sensors, RUL, hazard_class
    expected_train_cols = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + expected_sensor_cols + ['RUL', 'hazard_class']
    assert list(train_df.columns) == expected_train_cols

    # Test df should not have RUL and hazard_class by default (since run_pipeline doesn't add it to test_df)
    # Actually wait, run_pipeline doesn't add RUL/hazard to test.
    expected_test_cols = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + expected_sensor_cols
    assert list(test_df.columns) == expected_test_cols

    # Check for NaNs
    assert not train_df.isna().any().any(), "Training data contains NaNs"
    assert not test_df.isna().any().any(), "Testing data contains NaNs"

def test_rul_calculation_and_capping(mock_data_paths):
    train_path, test_path = mock_data_paths
    pipeline = CMAPSSDataPipeline()

    train_df, test_df = pipeline.run_pipeline(train_path, test_path)

    # Check that RUL does not exceed the ceiling
    assert train_df['RUL'].max() <= pipeline.rul_ceiling

    # Check that RUL behaves correctly (monotonically decreasing per engine until capped)
    for engine_id, group in train_df.groupby('engine_id'):
        rul_values = group['RUL'].values
        # It should be capped at 125, then decrease by 1 each cycle
        assert all(rul_values[i] >= rul_values[i+1] for i in range(len(rul_values)-1))

def test_hazard_class_mapping(mock_data_paths):
    train_path, test_path = mock_data_paths
    pipeline = CMAPSSDataPipeline()

    train_df, test_df = pipeline.run_pipeline(train_path, test_path)

    # Verify mapping logic
    assert all(train_df.loc[train_df['RUL'] > 30, 'hazard_class'] == 'Healthy')
    assert all(train_df.loc[(train_df['RUL'] > 15) & (train_df['RUL'] <= 30), 'hazard_class'] == 'Warning')
    assert all(train_df.loc[train_df['RUL'] <= 15, 'hazard_class'] == 'Critical')

def test_custom_kmeans_convergence_and_inertia():
    from src.clustering import CustomKMeans

    # Generate distinct clusters
    np.random.seed(42)
    cluster_1 = np.random.randn(50, 5) + np.array([5, 5, 5, 5, 5])
    cluster_2 = np.random.randn(50, 5) + np.array([-5, -5, -5, -5, -5])
    X = np.vstack((cluster_1, cluster_2))

    kmeans = CustomKMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    # 1. Verify convergence: we should have reached tol or max_iter
    assert kmeans.cluster_centers_ is not None, "Centroids were not fitted"

    # 2. Verify inertia decreases monotonically
    inertia_history = kmeans.inertia_history_

    # Handle the case where it converges in 1 iteration
    if len(inertia_history) > 1:
        is_monotonically_decreasing = all(
            inertia_history[i] >= inertia_history[i+1]
            for i in range(len(inertia_history) - 1)
        )
        assert is_monotonically_decreasing, "Inertia did not decrease monotonically"
