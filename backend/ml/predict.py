"""
Prediction Module
=================
Make predictions using trained models.
"""

import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from app.ml.feature_engineering import FeatureEngineer
from app.ml.preprocessing import DataPreprocessor
from app.ml.models.isolation_forest import IsolationForestModel
from app.ml.models.one_class_svm import OneClassSVMModel
from app.ml.models.lof import LOFModel
from app.ml.models.statistical import StatisticalDetector

logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Make predictions using ensemble of anomaly detection models.
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent.parent.parent / "saved_models"
        else:
            self.model_dir = Path(model_dir)
        
        self.feature_engineer = FeatureEngineer()
        self.preprocessor: Optional[DataPreprocessor] = None
        self.models: Dict[str, Any] = {}
        self.statistical_detector: Optional[StatisticalDetector] = None
        
        self._load_components()

    def _load_components(self):
        """Load all components."""
        logger.info("Loading prediction components...")
        
        # Load preprocessor
        try:
            self.preprocessor = DataPreprocessor.load(self.model_dir / "preprocessor.pkl")
            logger.info("Loaded preprocessor")
        except FileNotFoundError:
            logger.error("Preprocessor not found! Run training first.")
        
        # Load models
        model_files = {
            "isolation_forest": "isolation_forest.pkl",
            "one_class_svm": "one_class_svm.pkl",
            "lof": "lof.pkl"
        }
        
        for name, filename in model_files.items():
            try:
                self.models[name] = joblib.load(self.model_dir / filename)
                logger.info(f"Loaded {name}")
            except FileNotFoundError:
                logger.warning(f"Model {name} not found")
        
        # Load statistical detector
        try:
            self.statistical_detector = StatisticalDetector.load(self.model_dir / "statistical.pkl")
            logger.info("Loaded statistical detector")
        except FileNotFoundError:
            logger.warning("Statistical detector not found")

    def predict(self, transaction: Dict) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Prediction result dictionary
        """
        try:
            # 1. Feature engineering
            df = self.feature_engineer.transform(
                pd.DataFrame([transaction]),
                is_training=False
            )
            
            # 2. Preprocessing
            X = self.preprocessor.transform(df)
            
            # 3. Get predictions from all models
            predictions = {}
            scores = {}
            
            for model_name, model in self.models.items():
                try:
                    pred, score = model.predict(X)
                    predictions[model_name] = "fraud" if pred[0] == -1 else "normal"
                    scores[model_name] = float(score[0])
                except Exception as e:
                    logger.error(f"Error with {model_name}: {e}")
            
            # 4. Statistical detection
            if self.statistical_detector:
                try:
                    stat_result = self.statistical_detector.detect(X[0])
                    predictions["statistical"] = stat_result["prediction"]
                    scores["statistical"] = stat_result["score"]
                except Exception as e:
                    logger.error(f"Error with statistical detector: {e}")
            
            # 5. Ensemble voting
            fraud_votes = sum(1 for p in predictions.values() if p == "fraud")
            total_votes = len(predictions)
            
            # 6. Calculate final score (average of all scores)
            avg_score = np.mean(list(scores.values())) if scores else 0.5
            
            # 7. Final prediction
            is_fraud = fraud_votes >= (total_votes / 2)
            
            # 8. Determine best model (highest score for fraud)
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
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "prediction": "error",
                "model": "none",
                "anomaly_score": 0.0,
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def predict_batch(self, transactions: list) -> list:
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for i, trans in enumerate(transactions):
            result = self.predict(trans)
            result["index"] = i
            results.append(result)
        return results


def main():
    """CLI for predictions."""
    import argparse
    import pandas as pd
    import json
    
    parser = argparse.ArgumentParser(description="Fraud Detection Prediction")
    parser.add_argument("--input", type=str, required=True, help="Input CSV or JSON file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FraudPredictor(model_dir=args.model_dir)
    
    # Load input
    input_path = Path(args.input)
    if input_path.suffix == ".json":
        with open(input_path, "r") as f:
            transactions = json.load(f)
    else:
        df = pd.read_csv(input_path)
        transactions = df.to_dict("records")
    
    logger.info(f"Loaded {len(transactions)} transactions")
    
    # Predict
    results = predictor.predict_batch(transactions)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    fraud_count = sum(1 for r in results if r["prediction"] == "fraud")
    print(f"\nTotal: {len(results)}")
    print(f"Fraud: {fraud_count} ({fraud_count/len(results)*100:.1f}%)")
    print(f"Normal: {len(results)-fraud_count} ({(len(results)-fraud_count)/len(results)*100:.1f}%)")
