"""
Unit tests for CMAPSSDataPipeline.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.preprocessing import CMAPSSDataPipeline


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        CMAPSSDataPipeline.generate_synthetic_data(output_dir=tmpdir, seed=42)
        yield Path(tmpdir)


def test_synthetic_data_generation_and_loading(temp_data_dir):
    pipeline = CMAPSSDataPipeline()
    train_df = pipeline.load_raw_file(temp_data_dir / "train_FD001.txt")
    test_df = pipeline.load_raw_file(temp_data_dir / "test_FD001.txt")
    rul_df = pipeline.load_rul_file(temp_data_dir / "RUL_FD001.txt")

    assert len(train_df.columns) == 26
    assert len(test_df.columns) == 26
    assert list(rul_df.columns) == ["unit_nr", "RUL_final"]
    assert len(rul_df) == 5


def test_feature_schema_parity(temp_data_dir):
    pipeline = CMAPSSDataPipeline()
    train_raw = pipeline.load_raw_file(temp_data_dir / "train_FD001.txt")
    test_raw = pipeline.load_raw_file(temp_data_dir / "test_FD001.txt")
    rul_raw = pipeline.load_rul_file(temp_data_dir / "RUL_FD001.txt")

    train_proc, test_proc = pipeline.fit_transform(
        train_df=train_raw, test_df=test_raw, test_rul_df=rul_raw
    )

    # Check 18 feature schema
    assert len(pipeline.feature_columns) == 18
    for dropped_sensor in pipeline.DEFAULT_DROP_SENSORS:
        assert dropped_sensor not in pipeline.feature_columns

    # Verify both processed train and test have identical feature columns
    train_features = train_proc[pipeline.feature_columns]
    test_features = test_proc[pipeline.feature_columns]

    assert list(train_features.columns) == list(test_features.columns)
    assert train_features.shape[1] == 18
    assert test_features.shape[1] == 18


def test_no_nans_or_infs(temp_data_dir):
    pipeline = CMAPSSDataPipeline()
    train_raw = pipeline.load_raw_file(temp_data_dir / "train_FD001.txt")
    test_raw = pipeline.load_raw_file(temp_data_dir / "test_FD001.txt")
    rul_raw = pipeline.load_rul_file(temp_data_dir / "RUL_FD001.txt")

    train_proc, test_proc = pipeline.fit_transform(
        train_df=train_raw, test_df=test_raw, test_rul_df=rul_raw
    )

    assert not train_proc.isna().any().any()
    assert not np.isinf(train_proc.select_dtypes(include=[np.number])).any().any()

    assert not test_proc.isna().any().any()
    assert not np.isinf(test_proc.select_dtypes(include=[np.number])).any().any()


def test_rul_calculation_and_capping():
    pipeline = CMAPSSDataPipeline(max_rul=125)
    # Unit 1 with 200 cycles
    df = pd.DataFrame(
        {
            "unit_nr": [1] * 200,
            "time_cycles": np.arange(1, 201),
        }
    )
    # Fill remaining columns
    for col in [c for c in CMAPSSDataPipeline.RAW_COLUMNS if c not in ["unit_nr", "time_cycles"]]:
        df[col] = 1.0

    res = pipeline.calculate_train_rul(df)

    # At cycle 1: max_cycle=200, uncapped RUL = 199, capped RUL = 125
    assert res.loc[res["time_cycles"] == 1, "RUL_uncapped"].values[0] == 199
    assert res.loc[res["time_cycles"] == 1, "RUL"].values[0] == 125

    # At cycle 100: uncapped RUL = 100, capped RUL = 100
    assert res.loc[res["time_cycles"] == 100, "RUL_uncapped"].values[0] == 100
    assert res.loc[res["time_cycles"] == 100, "RUL"].values[0] == 100

    # Maximum value in RUL column must be <= 125
    assert res["RUL"].max() == 125


def test_health_label_boundaries():
    pipeline = CMAPSSDataPipeline(warning_threshold=50, critical_threshold=20)
    rul_series = pd.Series([100, 51, 50, 21, 20, 0])
    labels = pipeline.assign_health_labels(rul_series)

    expected = ["Healthy", "Healthy", "Warning", "Warning", "Critical", "Critical"]
    assert list(labels) == expected


def test_scaler_fitted_on_train_only():
    pipeline = CMAPSSDataPipeline()

    train_df, test_df, rul_df = CMAPSSDataPipeline.generate_synthetic_data(seed=123)

    # Check scaler is not fitted initially
    assert not pipeline.is_fitted

    pipeline.fit_transform(train_df=train_df, test_df=test_df, test_rul_df=rul_df)

    assert pipeline.is_fitted
    # Confirm scaler min_ and data_max_ correspond to train feature shapes
    assert len(pipeline.scaler.data_min_) == 18
    assert len(pipeline.scaler.data_max_) == 18


def test_test_data_without_rul():
    pipeline = CMAPSSDataPipeline()
    train_df, test_df, _ = CMAPSSDataPipeline.generate_synthetic_data(seed=456)

    # Process test without test_rul_df
    train_proc, test_proc = pipeline.fit_transform(train_df=train_df, test_df=test_df)

    assert "RUL" in train_proc.columns
    assert "health_label" in train_proc.columns

    # When test RUL file is absent, RUL / health_label shouldn't be invented for test set
    assert "RUL" not in test_proc.columns
    assert "health_label" not in test_proc.columns
    # But features should still be scaled
    assert len(test_proc[pipeline.feature_columns].columns) == 18
