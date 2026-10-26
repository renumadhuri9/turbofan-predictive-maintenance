"""
Preprocessing pipeline for NASA C-MAPSS FD001 dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CMAPSSDataPipeline:
    """
    Data ingestion and preprocessing pipeline for NASA C-MAPSS FD001 dataset.
    """

    DEFAULT_RAW_COLUMNS = [
        "unit_nr",
        "time_cycles",
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"s{i}" for i in range(1, 22)]

    DEFAULT_DROP_SENSORS = ["s1", "s5", "s10", "s16", "s18", "s19"]

    def __init__(
        self,
        max_rul: int = 125,
        warning_threshold: int = 50,
        critical_threshold: int = 20,
        drop_sensors: list[str] | None = None,
        raw_columns: list[str] | None = None,
        unit_col: str | None = None,
        cycle_col: str | None = None,
    ):
        self.max_rul = max_rul
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.raw_columns = raw_columns if raw_columns is not None else list(self.DEFAULT_RAW_COLUMNS)

        # Detect unit_col and cycle_col from raw_columns if not explicitly provided
        if unit_col is None:
            if "unit_nr" in self.raw_columns:
                self.unit_col = "unit_nr"
            elif "engine_id" in self.raw_columns:
                self.unit_col = "engine_id"
            else:
                self.unit_col = self.raw_columns[0]
        else:
            self.unit_col = unit_col

        if cycle_col is None:
            if "time_cycles" in self.raw_columns:
                self.cycle_col = "time_cycles"
            elif "cycle" in self.raw_columns:
                self.cycle_col = "cycle"
            else:
                self.cycle_col = self.raw_columns[1]
        else:
            self.cycle_col = cycle_col

        if drop_sensors is not None:
            self.drop_sensors = drop_sensors
        else:
            # Check if raw columns use 's_' prefix or 's' prefix
            if "s_1" in self.raw_columns:
                self.drop_sensors = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
            else:
                self.drop_sensors = list(self.DEFAULT_DROP_SENSORS)

        self.feature_columns = [
            col
            for col in self.raw_columns
            if col not in [self.unit_col, self.cycle_col] + self.drop_sensors
        ]

        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def load_raw_file(self, filepath: str | Path) -> pd.DataFrame:
        """
        Load a C-MAPSS space-separated sensor data file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=self.raw_columns,
            engine="python",
        )
        return df

    def load_rul_file(self, filepath: str | Path) -> pd.DataFrame:
        """
        Load C-MAPSS ground truth RUL file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=["RUL_final"],
            engine="python",
        )
        df[self.unit_col] = np.arange(1, len(df) + 1)
        return df[[self.unit_col, "RUL_final"]]

    def calculate_train_rul(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RUL for training data and cap at max_rul.
        """
        df = train_df.copy()
        max_cycles = df.groupby(self.unit_col)[self.cycle_col].transform("max")
        df["RUL_uncapped"] = max_cycles - df[self.cycle_col]
        df["RUL"] = df["RUL_uncapped"].clip(upper=self.max_rul)
        return df

    def calculate_test_rul(
        self, test_df: pd.DataFrame, rul_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Calculate test set RUL if ground truth RUL dataframe is provided.
        """
        df = test_df.copy()
        if rul_df is None:
            return df

        max_cycles = df.groupby(self.unit_col)[self.cycle_col].transform("max")
        df = df.merge(rul_df, on=self.unit_col, how="left")

        # RUL at cycle t = (max_cycle_test - t) + RUL_final
        df["RUL_uncapped"] = (max_cycles - df[self.cycle_col]) + df["RUL_final"]
        df["RUL"] = df["RUL_uncapped"].clip(upper=self.max_rul)
        df = df.drop(columns=["RUL_final"])
        return df

    def assign_health_labels(self, rul_series: pd.Series) -> pd.Series:
        """
        Generate Healthy, Warning, and Critical health labels based on RUL.
        """
        conditions = [
            rul_series > self.warning_threshold,
            (rul_series > self.critical_threshold) & (rul_series <= self.warning_threshold),
            rul_series <= self.critical_threshold,
        ]
        choices = ["Healthy", "Warning", "Critical"]
        labels = pd.Series(
            np.select(conditions, choices, default="Unknown"),
            index=rul_series.index,
            name="health_label",
        )
        return labels

    def fit_scaler(self, train_df: pd.DataFrame) -> "CMAPSSDataPipeline":
        """
        Fit MinMaxScaler strictly on training features.
        """
        self.scaler.fit(train_df[self.feature_columns])
        self.is_fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale feature columns using fitted MinMaxScaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_scaler or fit_transform first.")

        scaled_df = df.copy()
        scaled_features = self.scaler.transform(df[self.feature_columns])
        scaled_df[self.feature_columns] = scaled_features
        return scaled_df

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        test_rul_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Fit pipeline on train data and transform both train and test data.
        """
        # Process train
        train_processed = self.calculate_train_rul(train_df)
        train_processed["health_label"] = self.assign_health_labels(train_processed["RUL"])

        self.fit_scaler(train_processed)
        train_processed = self.transform_features(train_processed)

        # Process test if provided
        test_processed = None
        if test_df is not None:
            test_processed = self.calculate_test_rul(test_df, test_rul_df)
            if "RUL" in test_processed.columns:
                test_processed["health_label"] = self.assign_health_labels(test_processed["RUL"])
            test_processed = self.transform_features(test_processed)

        return train_processed, test_processed

    @staticmethod
    def generate_synthetic_data(
        output_dir: str | Path | None = None,
        num_train_units: int = 5,
        num_test_units: int = 5,
        seed: int = 42,
        raw_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate deterministic synthetic C-MAPSS dataset for unit testing without downloading.
        """
        rng = np.random.default_rng(seed)

        cols = raw_columns if raw_columns is not None else CMAPSSDataPipeline.DEFAULT_RAW_COLUMNS

        # Generate train
        train_rows = []
        for unit in range(1, num_train_units + 1):
            max_cycles = rng.integers(150, 250)
            for cycle in range(1, max_cycles + 1):
                # 3 settings + 21 sensors
                settings = rng.normal(0, 1, 3)
                sensors = rng.normal(100, 15, 21)
                # Introduce constant sensor values for constant sensors
                sensors[0] = 518.67  # s1
                sensors[4] = 14.62   # s5
                sensors[9] = 1.30    # s10
                sensors[15] = 0.03   # s16
                sensors[17] = 2388.0 # s18
                sensors[18] = 100.0  # s19

                row = [unit, cycle] + list(settings) + list(sensors)
                train_rows.append(row)

        train_df = pd.DataFrame(train_rows, columns=cols)

        # Generate test
        test_rows = []
        test_ruls = []
        for unit in range(1, num_test_units + 1):
            max_cycles = rng.integers(50, 100)
            final_rul = rng.integers(10, 100)
            test_ruls.append(final_rul)
            for cycle in range(1, max_cycles + 1):
                settings = rng.normal(0, 1, 3)
                sensors = rng.normal(100, 15, 21)
                sensors[0] = 518.67
                sensors[4] = 14.62
                sensors[9] = 1.30
                sensors[15] = 0.03
                sensors[17] = 2388.0
                sensors[18] = 100.0

                row = [unit, cycle] + list(settings) + list(sensors)
                test_rows.append(row)

        test_df = pd.DataFrame(test_rows, columns=cols)
        unit_col_name = cols[0]
        rul_df = pd.DataFrame({
            unit_col_name: np.arange(1, num_test_units + 1),
            "RUL_final": test_ruls,
        })

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save space-delimited text files without header/index
            train_df.to_csv(output_dir / "train_FD001.txt", sep=" ", index=False, header=False)
            test_df.to_csv(output_dir / "test_FD001.txt", sep=" ", index=False, header=False)
            rul_df[["RUL_final"]].to_csv(output_dir / "RUL_FD001.txt", sep=" ", index=False, header=False)

        return train_df, test_df, rul_df
