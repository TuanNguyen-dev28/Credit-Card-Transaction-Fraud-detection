"""
One-Class SVM Model
===================
Anomaly detection using One-Class Support Vector Machine.
"""

import numpy as np
import logging
from typing import Tuple
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


class OneClassSVMModel:
    """
    One-Class SVM for anomaly detection.
    
    Learns a decision boundary that encompasses the normal data points.
    Points outside the boundary are considered anomalies.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "auto"):
        """
        Initialize One-Class SVM.
        
        Args:
            nu: Upper bound on fraction of training errors (anomalies)
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Train the model on normal data only.
        
        Args:
            X: Training data (should be mostly normal samples)
        """
        logger.info(f"Training One-Class SVM on {X.shape[0]} samples...")
        logger.info(f"Parameters: nu={self.nu}, kernel={self.kernel}")
        
        self.model.fit(X)
        self.is_fitted = True
        logger.info("One-Class SVM training complete")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: -1 for anomaly, 1 for normal
            anomaly_scores: Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        # Normalize scores to [0, 1] range
        # decision_function returns signed distance to boundary
        scores_normalized = 1 / (1 + np.exp(-scores))  # Sigmoid normalization
        
        return predictions, scores_normalized

    def save(self, path: str):
        """Save model to file."""
        import joblib
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "OneClassSVMModel":
        """Load model from file."""
        import joblib
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
