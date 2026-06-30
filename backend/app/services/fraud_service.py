"""
Fraud Detection Service
=======================
Orchestrates multiple anomaly detection models.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.ml.models.isolation_forest import IsolationForestModel
from app.ml.models.one_class_svm import OneClassSVMModel
from app.ml.models.lof import LOFModel
from app.ml.models.statistical import StatisticalDetector

logger = logging.getLogger(__name__)


class FraudDetectionService:
    """Service for fraud detection using multiple models."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.statistical_detector = StatisticalDetector()
        self.model_dir = Path(__file__).parent.parent.parent.parent.parent / "saved_models"
        self.model_dir.mkdir(exist_ok=True)
        
        # Load or initialize models
        self._load_models()

    def _load_models(self):
        """Load saved models or initialize new ones."""
        try:
            # Try to load saved models
            isolation_model = joblib.load(self.model_dir / "isolation_forest.pkl")
            self.models["isolation_forest"] = isolation_model
            logger.info("Loaded Isolation Forest model")
        except FileNotFoundError:
            logger.warning("Isolation Forest model not found, will train on first request")
        
        try:
            svm_model = joblib.load(self.model_dir / "one_class_svm.pkl")
            self.models["one_class_svm"] = svm_model
            logger.info("Loaded One-Class SVM model")
        except FileNotFoundError:
            logger.warning("One-Class SVM model not found, will train on first request")
        
        try:
            lof_model = joblib.load(self.model_dir / "lof.pkl")
            self.models["lof"] = lof_model
            logger.info("Loaded LOF model")
        except FileNotFoundError:
            logger.warning("LOF model not found, will train on first request")

    def predict_single(self, transaction: Dict) -> Dict:
        """
        Predict fraud for a single transaction.
        
        Uses ensemble of all models for better accuracy.
        """
        # Extract features
        features = self._extract_features(transaction)
        
        # Get predictions from all models
        predictions = {}
        scores = {}
        
        # Statistical detection (Z-score and IQR)
        stat_result = self.statistical_detector.detect(features)
        predictions["statistical"] = stat_result["prediction"]
        scores["statistical"] = stat_result["score"]
        
        # ML models
        for model_name, model in self.models.items():
            try:
                pred, score = model.predict(features)
                predictions[model_name] = pred
                scores[model_name] = score
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
        
        # Ensemble voting
        fraud_votes = sum(1 for p in predictions.values() if p == "anomaly")
        total_votes = len(predictions)
        
        # Calculate average anomaly score
        avg_score = np.mean(list(scores.values())) if scores else 0.5
        
        # Determine final prediction
        is_fraud = fraud_votes >= (total_votes / 2)
        
        # Select best model (highest anomaly score for fraud)
        best_model = max(scores, key=scores.get) if scores else "unknown"
        
        return {
            "prediction": "fraud" if is_fraud else "normal",
            "model": best_model,
            "anomaly_score": round(float(avg_score), 4),
            "confidence": round(float(max(avg_score, 1 - avg_score)), 4),
            "details": {
                "votes": predictions,
                "scores": {k: round(float(v), 4) for k, v in scores.items()},
                "fraud_votes": fraud_votes,
                "total_models": total_votes
            }
        }

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for multiple transactions."""
        results = []
        for i, trans in enumerate(transactions):
            try:
                result = self.predict_single(trans)
                result["index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting transaction {i}: {e}")
                results.append({
                    "index": i,
                    "prediction": "error",
                    "model": "none",
                    "anomaly_score": 0.0,
                    "confidence": 0.0,
                    "details": {"error": str(e)}
                })
        return results

    def _extract_features(self, transaction: Dict) -> np.ndarray:
        """Extract features from transaction dict."""
        features = [
            transaction.get("amt", 0),
            transaction.get("lat", 0),
            transaction.get("long", 0),
            transaction.get("merch_lat", 0),
            transaction.get("merch_long", 0),
            transaction.get("city_pop", 0),
        ]
        return np.array(features).reshape(1, -1)

    def get_models_info(self) -> Dict:
        """Get information about all models."""
        models_info = []
        
        for name, model in self.models.items():
            models_info.append({
                "name": name,
                "type": type(model).__name__,
                "status": "loaded"
            })
        
        models_info.append({
            "name": "statistical",
            "type": "StatisticalDetector",
            "methods": ["z-score", "iqr"],
            "status": "active"
        })
        
        return {
            "models": models_info,
            "best_model": "ensemble"
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        try:
            train_path = Path(__file__).parent.parent.parent.parent.parent / "dataset" / "fraudTrain.csv"
            df = pd.read_csv(train_path, nrows=10000)
            
            return {
                "total_transactions": len(df),
                "fraud_count": int(df["is_fraud"].sum()),
                "normal_count": int((df["is_fraud"] == 0).sum()),
                "fraud_rate": round(df["is_fraud"].mean() * 100, 2),
                "categories": df["category"].nunique(),
                "states": df["state"].nunique(),
                "avg_amount": round(df["amt"].mean(), 2),
                "max_amount": round(df["amt"].max(), 2),
                "min_amount": round(df["amt"].min(), 2)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
