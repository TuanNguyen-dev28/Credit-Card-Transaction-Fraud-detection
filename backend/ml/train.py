"""
Model Training Module
=====================
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime

from app.ml.models.isolation_forest import IsolationForestModel
from app.ml.models.one_class_svm import OneClassSVMModel
from app.ml.models.lof import LOFModel
from app.ml.preprocessing import DataPreprocessor, prepare_train_test_split
from app.ml.feature_engineering import FeatureEngineer
from app.ml.load_data import DataLoader

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train multiple anomaly detection models."""

    def __init__(self, data_dir: str = None, model_dir: str = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent.parent / "dataset"
        else:
            self.data_dir = Path(data_dir)
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent.parent.parent / "saved_models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(exist_ok=True)
        
        self.data_loader = DataLoader(self.data_dir)
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.models: Dict[str, Any] = {}

    def train_all(self, sample_size: int = None) -> Dict[str, Any]:
        """
        Train all anomaly detection models.
        
        Args:
            sample_size: Optional sample size for training
            
        Returns:
            Dictionary with training results
        """
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*60)
        
        # 1. Load data
        logger.info("Step 1: Loading data...")
        train_df, test_df = self.data_loader.load_data(sample_size=sample_size)
        
        # 2. Feature engineering
        logger.info("Step 2: Feature engineering...")
        train_df = self.feature_engineer.transform(train_df, is_training=True)
        test_df = self.feature_engineer.transform(test_df, is_training=False)
        
        # 3. Preprocessing
        logger.info("Step 3: Preprocessing...")
        X_train, y_train = self.preprocessor.fit_transform(train_df, target_col="is_fraud")
        X_test, y_test = self.preprocessor.transform(test_df.drop(columns=["is_fraud"])), test_df["is_fraud"].values
        
        # 4. Train models
        results = {}
        
        # Isolation Forest
        logger.info("Step 4: Training Isolation Forest...")
        iso_model = IsolationForestModel()
        iso_model.fit(X_train)
        iso_metrics = self._evaluate_model(iso_model, X_test, y_test)
        results["isolation_forest"] = iso_metrics
        joblib.dump(iso_model, self.model_dir / "isolation_forest.pkl")
        logger.info(f"Isolation Forest - F1: {iso_metrics['f1']:.4f}, ROC-AUC: {iso_metrics['roc_auc']:.4f}")
        
        # One-Class SVM (train on normal data only)
        logger.info("Step 5: Training One-Class SVM...")
        svm_model = OneClassSVMModel()
        X_train_normal = X_train[y_train == 0]  # Train only on normal data
        svm_model.fit(X_train_normal)
        svm_metrics = self._evaluate_model(svm_model, X_test, y_test)
        results["one_class_svm"] = svm_metrics
        joblib.dump(svm_model, self.model_dir / "one_class_svm.pkl")
        logger.info(f"One-Class SVM - F1: {svm_metrics['f1']:.4f}, ROC-AUC: {svm_metrics['roc_auc']:.4f}")
        
        # LOF (Local Outlier Factor)
        logger.info("Step 6: Training LOF...")
        lof_model = LOFModel()
        lof_model.fit(X_train)
        lof_metrics = self._evaluate_model(lof_model, X_test, y_test)
        results["lof"] = lof_metrics
        joblib.dump(lof_model, self.model_dir / "lof.pkl")
        logger.info(f"LOF - F1: {lof_metrics['f1']:.4f}, ROC-AUC: {lof_metrics['roc_auc']:.4f}")
        
        # Save preprocessor
        self.preprocessor.save(self.model_dir / "preprocessor.pkl")
        
        # Save results
        self._save_results(results)
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        
        return results

    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a single model."""
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        y_pred, y_scores = model.predict(X_test)
        
        # Convert -1/1 to 0/1 for compatibility
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        # Ensure scores are in [0, 1] range
        y_scores_normalized = (y_scores + 1) / 2
        
        metrics = {
            "precision": precision_score(y_test, y_pred_binary, zero_division=0),
            "recall": recall_score(y_test, y_pred_binary, zero_division=0),
            "f1": f1_score(y_test, y_pred_binary, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_scores_normalized),
        }
        
        cm = confusion_matrix(y_test, y_pred_binary)
        metrics["confusion_matrix"] = cm.tolist()
        
        return metrics

    def _save_results(self, results: Dict[str, Any]):
        """Save training results to CSV."""
        import csv
        
        results_path = self.model_dir / "model_results.csv"
        
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Precision", "Recall", "F1", "ROC-AUC"])
            
            for model_name, metrics in results.items():
                writer.writerow([
                    model_name,
                    round(metrics["precision"], 4),
                    round(metrics["recall"], 4),
                    round(metrics["f1"], 4),
                    round(metrics["roc_auc"], 4)
                ])
        
        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    trainer = ModelTrainer()
    results = trainer.train_all(sample_size=100000)
    
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"  {metric}: {value:.4f}")
