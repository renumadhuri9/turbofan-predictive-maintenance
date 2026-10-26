import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from typing import Optional, Union

class PCAWrapper:
    """
    Wrapper for Principal Component Analysis (PCA) to achieve a desired
    explained variance ratio.
    """

    def __init__(self, target_variance: float = 0.90, random_state: int = 42):
        """
        Initializes the PCAWrapper.

        Parameters
        ----------
        target_variance : float
            The target cumulative explained variance ratio (default 0.90).
        random_state : int
            Random seed for reproducibility.
        """
        self.target_variance = target_variance
        self.random_state = random_state
        self.pca_: Optional[PCA] = None
        self.n_components_: Optional[int] = None
        self.explained_variance_ratio_full_: Optional[np.ndarray] = None

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fits PCA on the data and transforms it, finding the minimum components
        to reach the target variance.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The input data array or DataFrame.

        Returns
        -------
        np.ndarray
            The PCA-transformed data.
        """
        # First fit PCA with all components to find the explained variance
        initial_pca = PCA(random_state=self.random_state)
        initial_pca.fit(X)

        self.explained_variance_ratio_full_ = initial_pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(self.explained_variance_ratio_full_)

        # Find minimum number of components explaining >= target_variance
        self.n_components_ = int(np.argmax(cumulative_variance >= self.target_variance) + 1)

        self.pca_ = PCA(n_components=self.n_components_, random_state=self.random_state)
        return self.pca_.fit_transform(X)

    def save_scree_plot(self, save_path: str = "artifacts/scree_plot.png") -> None:
        """
        Generates and saves an annotated scree plot.

        Parameters
        ----------
        save_path : str
            The path where the scree plot will be saved.
        """
        if self.explained_variance_ratio_full_ is None or self.n_components_ is None:
            raise ValueError("PCA has not been fitted yet.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(10, 6))
        components = np.arange(1, len(self.explained_variance_ratio_full_) + 1)
        cumulative_variance = np.cumsum(self.explained_variance_ratio_full_)

        plt.bar(components, self.explained_variance_ratio_full_, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(components, cumulative_variance, where='mid', label='Cumulative explained variance')

        # Annotate the target variance cutoff
        plt.axhline(y=self.target_variance, color='r', linestyle='--', label=f'{self.target_variance*100:.0f}% Target Variance')
        plt.axvline(x=self.n_components_, color='g', linestyle='--', label=f'{self.n_components_} Components Chosen')

        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.title('PCA Scree Plot')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
