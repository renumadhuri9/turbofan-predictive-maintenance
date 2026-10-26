import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple


def apply_pca_with_variance_threshold(X: np.ndarray, variance_threshold: float = 0.90) -> Tuple[np.ndarray, PCA]:
    """
    Applies PCA to the given data to find the minimum number of principal
    components needed to explain a cumulative variance above a given threshold.

    Saves an annotated scree plot image to `artifacts/scree_plot.png`.

    Parameters
    ----------
    X : np.ndarray
        The input data to perform PCA on. Expected shape is (n_samples, n_features).
    variance_threshold : float, optional
        The desired cumulative explained variance. Defaults to 0.90.

    Returns
    -------
    X_pca : np.ndarray
        The data projected onto the retained principal components.
    pca : sklearn.decomposition.PCA
        The fitted PCA object.
    """
    # Fit PCA with all components to compute variance ratio
    full_pca = PCA(random_state=42)
    full_pca.fit(X)

    explained_variance_ratio = full_pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find minimum components needed
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Fit PCA with required number of components
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Save scree plot
    os.makedirs('artifacts', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=variance_threshold, color='r', linestyle='-')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.title('Scree Plot: Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.annotate(
        f'{n_components} components explain >= {variance_threshold*100:.0f}% variance',
        xy=(n_components, variance_threshold),
        xytext=(n_components + 1, variance_threshold - 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05)
    )
    plt.savefig('artifacts/scree_plot.png')
    plt.close()

    return X_pca, pca
