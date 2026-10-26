import numpy as np
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, List, Optional, Tuple, Union

class CustomKMeans:
    """
    Custom KMeans clustering implementation from scratch using NumPy.
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        """
        Initializes the CustomKMeans model.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run.
        tol : float
            Relative tolerance with regards to Frobenius norm of the difference in the
            cluster centers of two consecutive iterations to declare convergence.
        random_state : int
            Determines random number generation for centroid initialization.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.inertia_history_: List[float] = []

    def fit(self, X: np.ndarray) -> 'CustomKMeans':
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : np.ndarray
            Training instances to cluster.

        Returns
        -------
        CustomKMeans
            Fitted estimator.
        """
        np.random.seed(self.random_state)
        # Randomly initialize centroids from data points
        random_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[random_indices].copy()

        self.inertia_history_ = []

        for i in range(self.max_iter):
            # Calculate distances from each point to each centroid
            # Shape of X: (n_samples, n_features)
            # Shape of cluster_centers_: (n_clusters, n_features)
            # Distance matrix shape: (n_samples, n_clusters)
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)

            # Assign labels based on closest centroid
            self.labels_ = np.argmin(distances, axis=1)

            # Calculate inertia (sum of squared distances to closest centroid)
            min_distances = np.min(distances, axis=1)
            inertia = np.sum(min_distances ** 2)
            self.inertia_history_.append(inertia)

            # Update centroids
            new_centers = np.array([X[self.labels_ == j].mean(axis=0) if np.any(self.labels_ == j) else self.cluster_centers_[j] for j in range(self.n_clusters)])

            # Check for convergence
            center_shift = np.linalg.norm(new_centers - self.cluster_centers_)
            self.cluster_centers_ = new_centers

            if center_shift < self.tol:
                break

        self.inertia_ = self.inertia_history_[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray
            New data to predict.

        Returns
        -------
        np.ndarray
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)


class SklearnClusterBench:
    """
    Benchmarking suite for scikit-learn clustering algorithms.
    """
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initializes the SklearnClusterBench.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to use for benchmarking.
        random_state : int
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.models = {
            'KMeans': KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10),
            'Birch': Birch(n_clusters=self.n_clusters)
        }

    def evaluate_clusters(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluates the clustering algorithms using Silhouette Score and Davies-Bouldin Index.

        Parameters
        ----------
        X : np.ndarray
            Data to cluster and evaluate.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing evaluation metrics for each model.
        """
        results = {}
        for name, model in self.models.items():
            model.fit(X)
            labels = model.predict(X)

            if len(np.unique(labels)) > 1: # Silhouette score requires at least 2 clusters
                silhouette = silhouette_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)
            else:
                silhouette = -1.0
                davies_bouldin = float('inf')

            results[name] = {
                'Silhouette Score': silhouette,
                'Davies-Bouldin Index': davies_bouldin
            }
        return results

if __name__ == "__main__":
    from src.decomposition import PCAWrapper

    # Generate some dummy data for demonstration if actual data isn't loaded
    print("Generating dummy data for clustering demonstration...")
    np.random.seed(42)

    # Simulate 3 clusters with 20 features
    cluster_1 = np.random.randn(100, 20) + np.array([2] * 20)
    cluster_2 = np.random.randn(100, 20) + np.array([-2] * 20)
    cluster_3 = np.random.randn(100, 20) + np.array([0] * 20)
    X_dummy = np.vstack((cluster_1, cluster_2, cluster_3))

    print(f"Data shape: {X_dummy.shape}")

    # Apply PCA
    print("Applying PCA to reduce dimensionality...")
    pca_wrapper = PCAWrapper(target_variance=0.90)
    X_pca = pca_wrapper.fit_transform(X_dummy)
    print(f"PCA reduced data shape: {X_pca.shape} (Components: {pca_wrapper.n_components_})")

    # Evaluate clusters
    print("\nBenchmarking clustering algorithms...")
    bench = SklearnClusterBench(n_clusters=3)
    metrics = bench.evaluate_clusters(X_pca)

    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Silhouette Score: {model_metrics['Silhouette Score']:.4f}")
        print(f"  Davies-Bouldin Index: {model_metrics['Davies-Bouldin Index']:.4f}")

    print("\nEvaluating CustomKMeans...")
    custom_kmeans = CustomKMeans(n_clusters=3)
    custom_kmeans.fit(X_pca)
    custom_labels = custom_kmeans.predict(X_pca)

    custom_silhouette = silhouette_score(X_pca, custom_labels)
    custom_db = davies_bouldin_score(X_pca, custom_labels)

    print(f"\nCustomKMeans:")
    print(f"  Silhouette Score: {custom_silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {custom_db:.4f}")
    print(f"  Final Inertia: {custom_kmeans.inertia_:.4f}")
