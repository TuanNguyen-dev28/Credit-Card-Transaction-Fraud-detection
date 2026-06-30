"""
Local Outlier Factor (LOF) Model
=================================
Anomaly detection using Local Outlier Factor algorithm.
"""

import numpy as np
import logging
from typing import Tuple
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


class LOFModel:
    """
    Local Outlier Factor for anomaly detection.
    
    LOF compares the local density of a point with the densities of its neighbors.
    Points with much lower density than their neighbors are considered outliers.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        """
        Initialize LOF model.
        
        Args:
            n_neighbors: Number of neighbors to use for density estimation
            contamination: Expected proportion of anomalies
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True  # Enable predict on new data
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Train the model.
        
        Args:
            X: Training data (n_samples, n_features)
        """
        logger.info(f"Training LOF on {X.shape[0]} samples...")
        logger.info(f"Parameters: n_neighbors={self.n_neighbors}, contamination={self.contamination}")
        
        self.model.fit(X)
        self.is_fitted = True
        logger.info("LOF training complete")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: -1 for anomaly, 1 for normal
            anomaly_scores: LOF scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        # LOF returns negative scores for outliers, normalize to [0, 1]
        # Higher score = more anomalous
        scores_normalized = 1 - (scores + 1) / 2
        
        return predictions, scores_normalized

    def save(self, path: str):
        """Save model to file."""
        import joblib
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LOFModel":
        """Load model from file."""
        import joblib
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
