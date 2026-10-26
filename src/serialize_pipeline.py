import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from src.models import load_data, preprocess_data

def serialize_pipeline():
    """
    Trains and serializes the preprocessing and inference pipeline.
    Saves MinMaxScaler, PCA, and RandomForestClassifier to artifacts/.
    """
    print("Loading data...")
    train_df, test_df, true_rul = load_data()

    print("Preprocessing data...")
    X_train_full, _, y_train, _ = preprocess_data(train_df, test_df, true_rul)

    # In preprocess_data, the columns are dropped in this order:
    # ["unit_number", "time_in_cycles", "RUL", "target"]
    # The remaining columns are: setting_1, setting_2, setting_3, sensor_1 ... sensor_21
    # We want to extract only the 21 sensor columns (index 3 to 24)
    # The requirement specifically says: Extract only the 21 sensor columns (dropping setting_1, setting_2, setting_3).

    # Drop setting_1, setting_2, setting_3 (first 3 columns)
    X_train = X_train_full[:, 3:]

    print(f"Features shape: {X_train.shape}")
    assert X_train.shape[1] == 21, f"Expected 21 sensor columns, got {X_train.shape[1]}"

    # Fit MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit PCA (reusing logic from benchmark if needed, or default PCA)
    # Requirement: Fit PCA on scaled sensors
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    print(f"PCA reduced to {pca.n_components_} components")

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_pca, y_train)

    # Save artifacts
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    joblib.dump(pca, 'artifacts/pca.joblib')
    joblib.dump(rf_model, 'artifacts/best_model.joblib')
    print("Saved pipeline artifacts to artifacts/")

if __name__ == "__main__":
    serialize_pipeline()
