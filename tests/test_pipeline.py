import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing import CMAPSSDataPipeline

@pytest.fixture
def synthetic_data_paths(tmp_path):
    # Train data: 2 engines, various cycles
    # 26 columns total
    train_data = []
    # Engine 1: 3 cycles
    train_data.extend([
        [1, 1] + [0.1, 0.2, 0.3] + [0.0]*21,
        [1, 2] + [0.11, 0.21, 0.31] + [0.1]*21,
        [1, 3] + [0.12, 0.22, 0.32] + [0.2]*21,
    ])
    # Engine 2: 2 cycles
    train_data.extend([
        [2, 1] + [0.5, 0.6, 0.7] + [1.0]*21,
        [2, 2] + [0.51, 0.61, 0.71] + [1.1]*21,
    ])

    # Test data: 2 engines, subset of their cycles
    test_data = []
    # Engine 1: 2 cycles
    test_data.extend([
        [1, 1] + [0.1, 0.2, 0.3] + [0.0]*21,
        [1, 2] + [0.11, 0.21, 0.31] + [0.1]*21,
    ])
    # Engine 2: 1 cycle
    test_data.extend([
        [2, 1] + [0.5, 0.6, 0.7] + [1.0]*21,
    ])

    # RUL data: Ground truth RUL for each engine in test dataset
    rul_data = [
        [100], # Engine 1 has 100 RUL remaining after cycle 2
        [15]   # Engine 2 has 15 RUL remaining after cycle 1
    ]

    train_path = tmp_path / "train_FD001.txt"
    test_path = tmp_path / "test_FD001.txt"
    rul_path = tmp_path / "RUL_FD001.txt"

    # Write space-separated files
    pd.DataFrame(train_data).to_csv(train_path, sep=' ', header=False, index=False)
    pd.DataFrame(test_data).to_csv(test_path, sep=' ', header=False, index=False)
    pd.DataFrame(rul_data).to_csv(rul_path, sep=' ', header=False, index=False)

    return str(train_path), str(test_path), str(rul_path)

def test_missing_files():
    pipeline = CMAPSSDataPipeline("fake_train.txt", "fake_test.txt", "fake_rul.txt")
    with pytest.raises(FileNotFoundError):
        pipeline.load_all()

def test_file_parsing_and_columns(synthetic_data_paths):
    train_path, test_path, rul_path = synthetic_data_paths
    pipeline = CMAPSSDataPipeline(train_path, test_path, rul_path)
    pipeline.load_all()

    assert list(pipeline._train_df.columns) == CMAPSSDataPipeline.COLUMNS
    assert list(pipeline._test_df.columns) == CMAPSSDataPipeline.COLUMNS
    assert list(pipeline._rul_df.columns) == ["rul"]

    assert len(pipeline._train_df) == 5
    assert len(pipeline._test_df) == 3
    assert len(pipeline._rul_df) == 2

def test_rul_calculation_and_cap(synthetic_data_paths):
    train_path, test_path, rul_path = synthetic_data_paths
    pipeline = CMAPSSDataPipeline(train_path, test_path, rul_path)
    pipeline.load_all()

    train_df = pipeline.prepare_train_data()

    # Engine 1 has max_cycle=3. RUL for cycle 1 = 2, cycle 2 = 1, cycle 3 = 0.
    eng1 = train_df[train_df['engine_id'] == 1]
    assert eng1['RUL'].tolist() == [2, 1, 0]
    assert eng1['RUL_capped'].tolist() == [2, 1, 0]

    test_df = pipeline.prepare_test_data()
    # Engine 1 has ground truth 100 at cycle 2. RUL at cycle 1 = 101, cycle 2 = 100.
    eng1_test = test_df[test_df['engine_id'] == 1]
    assert eng1_test['RUL'].tolist() == [101, 100]
    assert eng1_test['RUL_capped'].tolist() == [101, 100]

def test_health_class_boundaries(synthetic_data_paths):
    train_path, test_path, rul_path = synthetic_data_paths
    pipeline = CMAPSSDataPipeline(train_path, test_path, rul_path)

    # Mocking data to test boundaries
    mock_df = pd.DataFrame({
        'RUL_capped': [31, 30, 16, 15, 0]
    })
    result_df = pipeline._assign_health_class(mock_df)
    expected_classes = ["Healthy", "Warning", "Warning", "Critical", "Critical"]
    assert result_df['health_class'].tolist() == expected_classes

def test_rul_length_validation(tmp_path, synthetic_data_paths):
    train_path, test_path, _ = synthetic_data_paths
    # Create an RUL file with wrong number of engines (3 instead of 2)
    bad_rul_path = tmp_path / "bad_rul.txt"
    pd.DataFrame([[100], [15], [50]]).to_csv(bad_rul_path, sep=' ', header=False, index=False)

    pipeline = CMAPSSDataPipeline(train_path, test_path, str(bad_rul_path))
    pipeline.load_all()
    with pytest.raises(ValueError, match="Length of RUL file"):
        pipeline.prepare_test_data()

def test_feature_preprocessing(synthetic_data_paths):
    train_path, test_path, rul_path = synthetic_data_paths
    pipeline = CMAPSSDataPipeline(train_path, test_path, rul_path)
    train_proc, test_proc = pipeline.run_pipeline()

    # Check dropped columns
    for col in CMAPSSDataPipeline.DROP_COLUMNS:
        assert col not in train_proc.columns
        assert col not in test_proc.columns

    # Check identical columns in train and test
    assert list(train_proc.columns) == list(test_proc.columns)

    # Check for NaNs or infs
    assert not train_proc.isna().any().any()
    assert not test_proc.isna().any().any()
    assert not np.isinf(train_proc.select_dtypes(include=[np.number])).any().any()
    assert not np.isinf(test_proc.select_dtypes(include=[np.number])).any().any()

    # Check scaling was applied (values should be roughly between 0 and 1)
    feature_cols = [c for c in train_proc.columns if c not in ['engine_id', 'cycle', 'RUL', 'RUL_capped', 'health_class']]
    for col in feature_cols:
        assert train_proc[col].min() >= -0.001
        assert train_proc[col].max() <= 1.001
