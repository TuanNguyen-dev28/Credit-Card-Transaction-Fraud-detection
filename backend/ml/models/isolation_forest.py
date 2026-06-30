"""
Isolation Forest Model
======================
Anomaly detection using Isolation Forest algorithm.
"""

import numpy as np
import logging
from typing import Tuple
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Isolation Forest for anomaly detection.
    
    Isolation Forest isolates observations by randomly selecting a feature
    and then randomly selecting a split value. Anomalies are easier to isolate,
    so they tend to have shorter path lengths in the trees.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize Isolation Forest model.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples="auto",
            max_features=1.0
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Train the model.
        
        Args:
            X: Training data (n_samples, n_features)
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples...")
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Isolation Forest training complete")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: -1 for anomaly, 1 for normal
            anomaly_scores: Decision function values (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        # Normalize scores to [0, 1] range (higher = more anomalous)
        scores_normalized = 1 - (scores + 1) / 2
        
        return predictions, scores_normalized

    def save(self, path: str):
        """Save model to file."""
        import joblib
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IsolationForestModel":
        """Load model from file."""
        import joblib
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
