import numpy as np
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple, Dict

class CustomKMeans:
    """
    Vectorized NumPy implementation of KMeans clustering from scratch.
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray) -> 'CustomKMeans':
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : np.ndarray
            Training instances to cluster. Shape (n_samples, n_features).

        Returns
        -------
        self
            Fitted estimator.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Random centroid initialization
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()

        for _ in range(self.max_iter):
            # Compute distances and assign labels
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([X[new_labels == k].mean(axis=0) if np.sum(new_labels == k) > 0 else self.centroids[k] for k in range(self.n_clusters)])

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids
            self.labels_ = new_labels

        # Calculate inertia
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray
            New data to predict. Shape (n_samples, n_features).

        Returns
        -------
        labels : np.ndarray
            Index of the cluster each sample belongs to.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

class SklearnClusterBench:
    """
    Benchmarks sklearn KMeans and Birch clustering on PCA-transformed data.
    """
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.birch = Birch(n_clusters=self.n_clusters)

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits models and returns predictions.

        Parameters
        ----------
        X : np.ndarray
            Data to cluster. Shape (n_samples, n_features).

        Returns
        -------
        kmeans_labels : np.ndarray
            Labels predicted by KMeans.
        birch_labels : np.ndarray
            Labels predicted by Birch.
        """
        kmeans_labels = self.kmeans.fit_predict(X)
        birch_labels = self.birch.fit_predict(X)
        return kmeans_labels, birch_labels

def evaluate_clusters(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluates clustering using Silhouette Score and Davies-Bouldin Index.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    labels : np.ndarray
        The predicted cluster labels.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing 'silhouette_score' and 'davies_bouldin_score'.
    """
    if len(np.unique(labels)) < 2:
        return {"silhouette_score": 0.0, "davies_bouldin_score": float('inf')}

    return {
        "silhouette_score": silhouette_score(X, labels, random_state=42),
        "davies_bouldin_score": davies_bouldin_score(X, labels)
    }

if __name__ == "__main__":
    from src.decomposition import apply_pca_with_variance_threshold

    # Fabricate synthetic data representing real data
    np.random.seed(42)
    X_synthetic = np.random.rand(1000, 50)

    # Apply PCA
    X_pca, _ = apply_pca_with_variance_threshold(X_synthetic, variance_threshold=0.90)

    # Run Sklearn Benchmark
    bench = SklearnClusterBench(n_clusters=3, random_state=42)
    kmeans_labels, birch_labels = bench.fit_predict(X_pca)

    # Evaluate
    kmeans_metrics = evaluate_clusters(X_pca, kmeans_labels)
    birch_metrics = evaluate_clusters(X_pca, birch_labels)

    # Print results to console
    print("Sklearn KMeans Benchmarks:")
    print(f"  Silhouette Score: {kmeans_metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin Score: {kmeans_metrics['davies_bouldin_score']:.4f}")

    print("\nSklearn Birch Benchmarks:")
    print(f"  Silhouette Score: {birch_metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin Score: {birch_metrics['davies_bouldin_score']:.4f}")
