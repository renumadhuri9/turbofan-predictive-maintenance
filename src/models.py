import pandas as pd
import numpy as np
import time
import joblib
import os
from typing import Tuple, List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def load_data(
    train_path: str = "data/raw/train_FD001.txt",
    test_path: str = "data/raw/test_FD001.txt",
    rul_path: str = "data/raw/RUL_FD001.txt"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the NASA C-MAPSS dataset from text files.

    Parameters
    ----------
    train_path : str
        Path to the training data file.
    test_path : str
        Path to the test data file.
    rul_path : str
        Path to the true RUL values file for the test data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        train_df, test_df, and true_rul series.
    """
    col_names = ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3"]
    col_names += [f"sensor_{i}" for i in range(1, 22)]

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])["RUL"]

    return train_df, test_df, true_rul


def map_rul_to_class(rul: float) -> int:
    """
    Map Remaining Useful Life (RUL) to 3 operational classes.

    Healthy: RUL > 30 (mapped to 0)
    Warning: 15 < RUL <= 30 (mapped to 1)
    Critical: RUL <= 15 (mapped to 2)

    Parameters
    ----------
    rul : float
        Remaining Useful Life.

    Returns
    -------
    int
        Class index (0, 1, or 2).
    """
    if rul > 30:
        return 0
    elif rul > 15:
        return 1
    else:
        return 2

def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    true_rul: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate RUL, map to classes, and prepare feature matrices.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Test data.
    true_rul : pd.Series
        True RUL values for test engines.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    # Calculate RUL for training data
    rul_train = pd.DataFrame(train_df.groupby("unit_number")["time_in_cycles"].max()).reset_index()
    rul_train.columns = ["unit_number", "max_cycles"]
    train_df = train_df.merge(rul_train, on=["unit_number"], how="left")
    train_df["RUL"] = train_df["max_cycles"] - train_df["time_in_cycles"]
    train_df.drop("max_cycles", axis=1, inplace=True)

    # Calculate RUL for test data (last cycle only for evaluation)
    # The true_rul corresponds to the RUL at the LAST cycle of each engine in test_df
    rul_test = pd.DataFrame(test_df.groupby("unit_number")["time_in_cycles"].max()).reset_index()
    rul_test.columns = ["unit_number", "max_cycles"]
    rul_test["true_rul"] = true_rul.values
    test_df = test_df.merge(rul_test, on=["unit_number"], how="left")

    # The RUL at any point in the test set is true_rul + (max_cycles - current_cycle)
    test_df["RUL"] = test_df["true_rul"] + (test_df["max_cycles"] - test_df["time_in_cycles"])
    test_df.drop(["max_cycles", "true_rul"], axis=1, inplace=True)

    # Map to 3 classes
    train_df["target"] = train_df["RUL"].apply(map_rul_to_class)
    test_df["target"] = test_df["RUL"].apply(map_rul_to_class)

    # Prepare feature matrices (drop engine ID, cycle, RUL, target)
    cols_to_drop = ["unit_number", "time_in_cycles", "RUL", "target"]

    X_train = train_df.drop(columns=cols_to_drop).values
    y_train = train_df["target"].values

    X_test = test_df.drop(columns=cols_to_drop).values
    y_test = test_df["target"].values

    return X_train, X_test, y_train, y_test


def get_models() -> Dict[str, Any]:
    """
    Initialize and return the required machine learning models.

    Returns
    -------
    Dict[str, Any]
        Dictionary of model names and their corresponding initialized objects.
    """
    return {
        "Logistic Regression": LogisticRegression(penalty="l2", random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            early_stopping=True,
            random_state=42,
            max_iter=1000
        )
    }

def train_and_evaluate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Train and evaluate models, recording metrics and confusion matrices.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of initialized models.
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Test features.
    y_train : np.ndarray
        Training targets.
    y_test : np.ndarray
        Test targets.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, np.ndarray]]
        Metrics DataFrame and dictionary of confusion matrices.
    """
    results = []
    confusion_matrices = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Evaluate on training data
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Inference time on test data
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        end_time = time.time()

        # Calculate inference latency per 1000 samples in milliseconds
        latency_ms_per_1000 = ((end_time - start_time) / len(X_test)) * 1000 * 1000

        # Metrics on test data
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        # Confusion matrix for all models to have them returned
        cm = confusion_matrix(y_test, y_test_pred, normalize="true")
        confusion_matrices[name] = cm

        results.append({
            "Model Name": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Precision": precision,
            "Recall": recall,
            "Macro F1-Score": f1,
            "Inference Latency per 1,000 samples (ms)": latency_ms_per_1000
        })

    metrics_df = pd.DataFrame(results)
    return metrics_df, confusion_matrices

def save_metrics(metrics_df: pd.DataFrame, filepath: str = "artifacts/benchmark_metrics.csv") -> None:
    """
    Save the metrics DataFrame to a CSV file.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics DataFrame.
    filepath : str
        Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    metrics_df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def save_confusion_matrices(confusion_matrices: Dict[str, np.ndarray], output_dir: str = "artifacts/") -> None:
    """
    Save confusion matrices for Random Forest and MLP.

    Parameters
    ----------
    confusion_matrices : Dict[str, np.ndarray]
        Dictionary of confusion matrices.
    output_dir : str
        Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    models_to_plot = ["Random Forest", "MLP Classifier"]
    classes = ["Healthy (0)", "Warning (1)", "Critical (2)"]

    for name in models_to_plot:
        if name in confusion_matrices:
            cm = confusion_matrices[name]
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)

            # Add labels and annotations
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Normalized Confusion Matrix - {name}')

            # Add text annotations
            for (i, j), z in np.ndenumerate(cm):
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

            filename = name.lower().replace(" ", "_") + "_cm.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            print(f"Saved confusion matrix for {name} to {filepath}")

def main() -> None:
    """
    Main function to run the benchmark pipeline.
    """
    print("Loading data...")
    train_df, test_df, true_rul = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df, true_rul)

    print("Initializing models...")
    models = get_models()

    print("Training and evaluating models...")
    metrics_df, confusion_matrices = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)

    print("Saving results...")
    save_metrics(metrics_df)
    save_confusion_matrices(confusion_matrices)

    print("Benchmarking complete.")


if __name__ == "__main__":
    main()
