import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional

class CMAPSSDataPipeline:
    """
    Pipeline for data ingestion and preprocessing of the NASA C-MAPSS FD001 dataset.
    """

    # 26 column names as specified
    COLUMNS = [
        "engine_id", "cycle", "setting_1", "setting_2", "setting_3",
        "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9",
        "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16", "s_17",
        "s_18", "s_19", "s_20", "s_21"
    ]

    # Sensors that are constant in FD001 and should be dropped
    DROP_COLUMNS = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]

    def __init__(self, train_path: str, test_path: str, rul_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.rul_path = rul_path
        self.scaler = MinMaxScaler()
        self._train_df = None
        self._test_df = None
        self._rul_df = None

    def _load_data(self, filepath: str, is_rul: bool = False) -> pd.DataFrame:
        """Loads whitespace-separated data."""
        try:
            if is_rul:
                return pd.read_csv(filepath, sep=r'\s+', header=None, names=["rul"])
            else:
                df = pd.read_csv(filepath, sep=r'\s+', header=None)
                # Ensure the number of columns loaded isn't more than our names (sometimes there's a trailing space)
                df = df.iloc[:, :26]
                df.columns = self.COLUMNS
                return df
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required data file not found: {filepath}") from e

    def load_all(self):
        """Load train, test, and RUL files."""
        self._train_df = self._load_data(self.train_path)
        self._test_df = self._load_data(self.test_path)
        self._rul_df = self._load_data(self.rul_path, is_rul=True)

    def _assign_health_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign health classes based on RUL_capped."""
        def get_health_class(rul):
            if rul > 30:
                return "Healthy"
            elif 15 < rul <= 30:
                return "Warning"
            else:
                return "Critical"

        df['health_class'] = df['RUL_capped'].apply(get_health_class)
        return df

    def prepare_train_data(self) -> pd.DataFrame:
        """Calculate RUL, cap it, and assign health classes for train data."""
        if self._train_df is None:
            self.load_all()

        df = self._train_df.copy()

        # Calculate max cycle per engine
        max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycle.columns = ['engine_id', 'max_cycle']

        # Merge back to calculate RUL
        df = df.merge(max_cycle, on='engine_id', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df['RUL_capped'] = df['RUL'].clip(upper=125)

        # Drop max_cycle column
        df = df.drop(columns=['max_cycle'])

        # Assign health classes
        df = self._assign_health_class(df)

        return df

    def preprocess_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Drop constant columns, fit MinMaxScaler on train features,
        and transform both train and test features.
        """
        train_processed = train_df.copy()
        test_processed = test_df.copy()

        # Drop constant columns
        train_processed = train_processed.drop(columns=self.DROP_COLUMNS, errors='ignore')
        test_processed = test_processed.drop(columns=self.DROP_COLUMNS, errors='ignore')

        # Identify feature columns (exclude engine_id, cycle, RUL, RUL_capped, health_class)
        exclude_cols = ['engine_id', 'cycle', 'RUL', 'RUL_capped', 'health_class']
        feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

        # Fit scaler on train features and transform train
        train_processed[feature_cols] = self.scaler.fit_transform(train_processed[feature_cols])

        # Transform test features using fitted scaler
        test_processed[feature_cols] = self.scaler.transform(test_processed[feature_cols])

        return train_processed, test_processed

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full preprocessing pipeline."""
        self.load_all()
        train_df = self.prepare_train_data()
        test_df = self.prepare_test_data()
        return self.preprocess_features(train_df, test_df)

    def prepare_test_data(self) -> pd.DataFrame:
        """Calculate RUL, cap it, and assign health classes for test data."""
        if self._test_df is None or self._rul_df is None:
            self.load_all()

        df = self._test_df.copy()
        rul_df = self._rul_df.copy()

        # Validate that number of values matches distinct test engines
        distinct_engines = df['engine_id'].nunique()
        if len(rul_df) != distinct_engines:
            raise ValueError(
                f"Length of RUL file ({len(rul_df)}) does not match "
                f"number of distinct engines in test dataset ({distinct_engines})."
            )

        # Ground truth RUL is provided per engine for the last recorded cycle
        rul_df['engine_id'] = rul_df.index + 1

        # Calculate max cycle per engine in test data
        max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycle.columns = ['engine_id', 'max_cycle']

        # Merge ground truth RUL and max_cycle back into test df
        df = df.merge(max_cycle, on='engine_id', how='left')
        df = df.merge(rul_df, on='engine_id', how='left')

        # Test RUL = ground_truth_RUL + (max_cycle - current_cycle)
        df['RUL'] = df['rul'] + (df['max_cycle'] - df['cycle'])
        df['RUL_capped'] = df['RUL'].clip(upper=125)

        # Drop intermediate columns
        df = df.drop(columns=['max_cycle', 'rul'])

        # Assign health classes
        df = self._assign_health_class(df)

        return df
