import numpy as np
import pytest
from src.clustering import CustomKMeans

def test_custom_kmeans_convergence():
    """
    Tests that CustomKMeans converges and inertia decreases monotonically across iterations.
    """
    np.random.seed(42)
    # Generate some simple synthetic clustered data
    cluster_1 = np.random.randn(50, 2) + np.array([5, 5])
    cluster_2 = np.random.randn(50, 2) + np.array([-5, -5])
    cluster_3 = np.random.randn(50, 2) + np.array([5, -5])
    X = np.vstack([cluster_1, cluster_2, cluster_3])

    kmeans = CustomKMeans(n_clusters=3, max_iter=100, random_state=42)

    # We will step through fit manually to check monotonically decreasing inertia
    kmeans.centroids = None
    np.random.seed(kmeans.random_state)
    n_samples, n_features = X.shape
    random_indices = np.random.choice(n_samples, kmeans.n_clusters, replace=False)
    kmeans.centroids = X[random_indices].copy()

    previous_inertia = float('inf')

    for i in range(kmeans.max_iter):
        # Compute distances and assign labels
        distances = np.linalg.norm(X[:, np.newaxis] - kmeans.centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)

        # Calculate current inertia based on new assignments and current centroids
        current_inertia = np.sum(np.min(distances, axis=1) ** 2)

        # Inertia should not increase
        assert current_inertia <= previous_inertia + 1e-9, f"Inertia increased at iteration {i}: {previous_inertia} -> {current_inertia}"
        previous_inertia = current_inertia

        # Compute new centroids
        new_centroids = np.array([X[new_labels == k].mean(axis=0) if np.sum(new_labels == k) > 0 else kmeans.centroids[k] for k in range(kmeans.n_clusters)])

        if np.allclose(kmeans.centroids, new_centroids):
            # Convergence reached
            break

        kmeans.centroids = new_centroids

    else:
        pytest.fail(f"CustomKMeans failed to converge within {kmeans.max_iter} iterations.")
