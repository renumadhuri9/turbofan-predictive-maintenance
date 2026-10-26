"""
Benchmarking script for evaluating multiple classification models.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from src.preprocessing import CMAPSSDataPipeline


def get_test_data(pipeline: CMAPSSDataPipeline) -> pd.DataFrame:
    """
    Process the test set to include Remaining Useful Life (RUL) and hazard class labels.

    Parameters
    ----------
    pipeline : CMAPSSDataPipeline
        Initialized data pipeline.

    Returns
    -------
    pd.DataFrame
        Test DataFrame enriched with RUL and hazard_class.
    """
    _, test_df = pipeline.run_pipeline('data/raw/train_FD001.txt', 'data/raw/test_FD001.txt')

    rul_df = pd.read_csv('data/raw/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    rul_df['engine_id'] = rul_df.index + 1

    max_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles = max_cycles.rename(columns={'cycle': 'max_cycle'})
    max_cycles = max_cycles.merge(rul_df, on='engine_id')
    max_cycles['true_max_cycle'] = max_cycles['max_cycle'] + max_cycles['RUL']

    test_df = test_df.merge(max_cycles[['engine_id', 'true_max_cycle']], on='engine_id', how='left')
    test_df['RUL'] = test_df['true_max_cycle'] - test_df['cycle']
    test_df['RUL'] = test_df['RUL'].clip(upper=pipeline.rul_ceiling)

    test_df = pipeline.add_hazard_class(test_df)
    test_df = test_df.drop(columns=['true_max_cycle'])

    return test_df


def main() -> None:
    """
    Execute model benchmarking pipeline.

    Trains multiple models, evaluates their performance, logs metrics,
    and generates normalized confusion matrices for selected models.
    """
    pipeline = CMAPSSDataPipeline()
    train_df, _ = pipeline.run_pipeline('data/raw/train_FD001.txt', 'data/raw/test_FD001.txt')
    test_df = get_test_data(pipeline)

    feature_cols = [col for col in train_df.columns if col not in ['engine_id', 'cycle', 'RUL', 'hazard_class']]

    X_train = train_df[feature_cols]
    y_train = train_df['hazard_class']

    X_test = test_df[feature_cols]
    y_test = test_df['hazard_class']

    models = {
        'Logistic Regression': LogisticRegression(penalty='l2', random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', early_stopping=True, random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        start_time = time.time()
        y_test_pred = model.predict(X_test)
        end_time = time.time()

        latency_ms_per_1000 = ((end_time - start_time) * 1000000) / len(X_test)

        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

        results.append({
            'Model Name': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision': precision,
            'Recall': recall,
            'Macro F1-Score': f1,
            'Inference Latency per 1,000 samples (milliseconds)': latency_ms_per_1000
        })

        if name in ['Random Forest', 'MLP']:
            cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap='Blues')
            plt.title(f'Normalized Confusion Matrix - {name}')
            filename = f"artifacts/{name.lower().replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(filename)
            plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv('artifacts/benchmark_metrics.csv', index=False)


if __name__ == "__main__":
    main()
