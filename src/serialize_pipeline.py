"""
Script to train and serialize the CMAPSS preprocessing and inference pipeline.
Saves MinMaxScaler, PCA model, and the best-performing Random Forest model.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import CMAPSSDataPipeline
from src.decomposition import PCAWrapper

def serialize_pipeline() -> None:
    """
    Trains the CMAPSS dataset pipeline (Scaler, PCA, Random Forest)
    and serializes the models to the artifacts/ directory.
    """
    os.makedirs('artifacts', exist_ok=True)

    # 1. Load and preprocess data
    pipeline = CMAPSSDataPipeline()
    train_df, test_df = pipeline.run_pipeline('data/raw/train_FD001.txt', 'data/raw/test_FD001.txt')

    feature_cols = [col for col in train_df.columns if col not in ['engine_id', 'cycle', 'RUL', 'hazard_class']]

    X_train = train_df[feature_cols]
    y_train = train_df['hazard_class']

    # The pipeline's run_pipeline method already fits and transforms the data using MinMaxScaler.
    # The scaled features are stored in train_df[feature_cols].
    # We can save the fitted scaler from the pipeline object.

    scaler_path = os.path.join('artifacts', 'scaler.joblib')
    joblib.dump(pipeline.scaler, scaler_path)
    print(f"Saved MinMaxScaler to {scaler_path}")

    # 2. Apply PCA
    pca_wrapper = PCAWrapper(target_variance=0.90, random_state=42)
    X_train_pca = pca_wrapper.fit_transform(X_train)

    pca_path = os.path.join('artifacts', 'pca.joblib')
    joblib.dump(pca_wrapper.pca_, pca_path)
    print(f"Saved PCA to {pca_path}")

    # 3. Train best model (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_pca, y_train)

    model_path = os.path.join('artifacts', 'best_model.joblib')
    joblib.dump(rf_model, model_path)
    print(f"Saved Random Forest classifier to {model_path}")

if __name__ == "__main__":
    serialize_pipeline()
