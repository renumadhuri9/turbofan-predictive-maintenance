import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

class CMAPSSDataPipeline:
    """
    Pipeline for data ingestion and preprocessing of NASA C-MAPSS FD001 dataset.
    """

    def __init__(self):
        """
        Initializes the CMAPSS data pipeline.
        """
        self.columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + ['s_' + str(i) for i in range(1, 22)]
        self.scaler = MinMaxScaler()
        self.drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
        self.rul_ceiling = 125

    def calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds Remaining Useful Life (RUL) column to the DataFrame.
        RUL is calculated as max_cycle - current_cycle for each engine.
        It is also clipped at the specified ceiling.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing 'engine_id' and 'cycle' columns.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the 'RUL' column added.
        """
        # Find max cycle for each engine
        max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles = max_cycles.rename(columns={'cycle': 'max_cycle'})

        # Merge max_cycle back to the original dataframe
        df = df.merge(max_cycles, on='engine_id', how='left')

        # Calculate RUL
        df['RUL'] = df['max_cycle'] - df['cycle']

        # Clip RUL
        df['RUL'] = df['RUL'].clip(upper=self.rul_ceiling)

        # Drop max_cycle column as it's no longer needed
        df = df.drop(columns=['max_cycle'])

        return df

    def drop_zero_variance_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the specified zero-variance sensors from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame without the zero-variance sensors.
        """
        return df.drop(columns=self.drop_sensors, errors='ignore')

    def fit_transform_scaler(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits MinMaxScaler on the training feature columns and transforms both train and test sets.
        Only scales sensor and setting columns.

        Parameters
        ----------
        train_df : pd.DataFrame
            The training DataFrame.
        test_df : pd.DataFrame
            The testing DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The scaled training and testing DataFrames.
        """
        feature_cols = [col for col in train_df.columns if col not in ['engine_id', 'cycle', 'RUL', 'hazard_class']]

        train_df[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])

        return train_df, test_df

    def add_hazard_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a discrete hazard class column based on RUL.
        'Healthy' (RUL > 30), 'Warning' (15 < RUL <= 30), and 'Critical' (RUL <= 15).

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the 'RUL' column.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the 'hazard_class' column added.
        """
        conditions = [
            (df['RUL'] > 30),
            (df['RUL'] > 15) & (df['RUL'] <= 30),
            (df['RUL'] <= 15)
        ]
        choices = ['Healthy', 'Warning', 'Critical']
        df['hazard_class'] = np.select(conditions, choices, default='Unknown')
        return df

    def run_pipeline(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the full preprocessing pipeline.

        Parameters
        ----------
        train_path : str
            Path to the training data text file.
        test_path : str
            Path to the testing data text file.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The processed training and testing DataFrames.
        """
        # 1. Load data
        train_df, test_df = self.load_data(train_path, test_path)

        # 2. Drop zero-variance sensors
        train_df = self.drop_zero_variance_sensors(train_df)
        test_df = self.drop_zero_variance_sensors(test_df)

        # 3. Calculate RUL for training data
        train_df = self.calculate_rul(train_df)

        # 4. Fit transform scaler
        train_df, test_df = self.fit_transform_scaler(train_df, test_df)

        # 5. Add hazard class to training data
        train_df = self.add_hazard_class(train_df)

        return train_df, test_df

    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the training and testing data from the given paths.

        Parameters
        ----------
        train_path : str
            Path to the training data text file.
        test_path : str
            Path to the testing data text file.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the training and testing pandas DataFrames.
        """
        train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=self.columns)
        test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=self.columns)
        return train_df, test_df
