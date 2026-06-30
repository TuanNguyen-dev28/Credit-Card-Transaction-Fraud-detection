"""
Statistical Anomaly Detection
==============================
Z-score and IQR methods for anomaly detection.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class StatisticalDetector:
    """
    Statistical methods for anomaly detection.
    
    Methods:
    1. Z-score: Flag if |z| > 3
    2. IQR: Flag if value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
    """

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize statistical detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
            iqr_multiplier: IQR multiplier for outlier boundaries
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.statistics: Dict[str, Dict] = {}
        self.fitted = False

    def fit(self, X: np.ndarray, feature_names: Optional[list] = None):
        """
        Compute statistics from training data.
        
        Args:
            X: Training data (n_samples, n_features)
            feature_names: Optional feature names
        """
        logger.info(f"Fitting statistical detector on {X.shape[0]} samples...")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        for i, name in enumerate(feature_names):
            feature_data = X[:, i]
            
            # Z-score statistics
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            
            # IQR statistics
            q1 = np.percentile(feature_data, 25)
            q3 = np.percentile(feature_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            self.statistics[name] = {
                "mean": mean,
                "std": std,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
        
        self.fitted = True
        logger.info(f"Statistical detector fitted for {len(self.statistics)} features")
        return self

    def detect(self, X: np.ndarray, feature_names: Optional[list] = None) -> Dict:
        """
        Detect anomalies using Z-score and IQR methods.
        
        Args:
            X: Input data (n_samples, n_features)
            feature_names: Optional feature names
            
        Returns:
            Dictionary with prediction, score, and details
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        if feature_names is None:
            feature_names = list(self.statistics.keys())
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        anomaly_scores = np.zeros(n_samples)
        anomaly_flags = np.zeros(n_samples)
        
        for i, name in enumerate(feature_names):
            if name not in self.statistics:
                continue
            
            stats = self.statistics[name]
            feature_values = X[:, i]
            
            # Z-score method
            z_scores = np.abs((feature_values - stats["mean"]) / (stats["std"] + 1e-10))
            z_anomalies = z_scores > self.z_threshold
            
            # IQR method
            iqr_anomalies = (feature_values < stats["lower_bound"]) | \
                           (feature_values > stats["upper_bound"])
            
            # Combine methods
            feature_anomalies = z_anomalies | iqr_anomalies
            
            # Calculate anomaly score for this feature
            feature_score = np.where(
                feature_anomalies,
                np.minimum(z_scores / self.z_threshold, 1.0),
                0.0
            )
            
            anomaly_scores += feature_score
            anomaly_flags += feature_anomalies.astype(int)
        
        # Average score across features
        avg_scores = anomaly_scores / max(len(feature_names), 1)
        
        # Final prediction: anomaly if flagged in any feature
        predictions = np.where(anomaly_flags > 0, -1, 1)
        
        # For single sample
        if n_samples == 1:
            pred = "anomaly" if predictions[0] == -1 else "normal"
            score = float(avg_scores[0])
        else:
            pred = "anomaly" if predictions[0] == -1 else "normal"
            score = float(avg_scores[0])
        
        return {
            "prediction": pred,
            "score": score,
            "anomaly_count": int(anomaly_flags[0]),
            "total_features": len(feature_names)
        }

    def detect_zscore(self, X: np.ndarray, feature_names: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using only Z-score method.
        
        Returns:
            Tuple of (predictions, scores)
            predictions: 1 for anomaly, 0 for normal
            scores: Anomaly scores (0-1)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        if feature_names is None:
            feature_names = list(self.statistics.keys())
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        max_z_scores = np.zeros(X.shape[0])
        
        for i, name in enumerate(feature_names):
            if name not in self.statistics:
                continue
            
            stats = self.statistics[name]
            z_scores = np.abs((X[:, i] - stats["mean"]) / (stats["std"] + 1e-10))
            max_z_scores = np.maximum(max_z_scores, z_scores)
        
        predictions = (max_z_scores > self.z_threshold).astype(int)
        scores = np.minimum(max_z_scores / self.z_threshold, 1.0)
        
        return predictions, scores

    def detect_iqr(self, X: np.ndarray, feature_names: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using only IQR method.
        
        Returns:
            Tuple of (predictions, scores)
            predictions: 1 for anomaly, 0 for normal
            scores: Anomaly scores (0-1)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        if feature_names is None:
            feature_names = list(self.statistics.keys())
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        anomaly_counts = np.zeros(n_samples)
        
        for i, name in enumerate(feature_names):
            if name not in self.statistics:
                continue
            
            stats = self.statistics[name]
            feature_values = X[:, i]
            
            is_anomaly = (feature_values < stats["lower_bound"]) | \
                        (feature_values > stats["upper_bound"])
            anomaly_counts += is_anomaly.astype(int)
        
        predictions = (anomaly_counts > 0).astype(int)
        scores = anomaly_counts / max(len(feature_names), 1)
        
        return predictions, scores

    def save(self, path: str):
        """Save detector to file."""
        import joblib
        joblib.dump({
            "statistics": self.statistics,
            "z_threshold": self.z_threshold,
            "iqr_multiplier": self.iqr_multiplier,
            "fitted": self.fitted
        }, path)
        logger.info(f"Statistical detector saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StatisticalDetector":
        """Load detector from file."""
        import joblib
        data = joblib.load(path)
        detector = cls(
            z_threshold=data["z_threshold"],
            iqr_multiplier=data["iqr_multiplier"]
        )
        detector.statistics = data["statistics"]
        detector.fitted = data["fitted"]
        logger.info(f"Statistical detector loaded from {path}")
        return detector
