"""
Credit Card Fraud Detection - Model Training Pipeline
==================================================
"""

import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from src.models import (
    LogisticRegressionModel, RandomForestModel,
    XGBoostModel, LightGBMModel, ModelFactory
)
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer, Preprocessor, prepare_features

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and comparison."""

    def __init__(
        self,
        train_path: Path,
        test_path: Path,
        model_dir: Path,
        random_state: int = 42
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = Path(model_dir)
        self.random_state = random_state

        # Components
        self.data_loader = DataLoader(train_path, test_path)
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = Preprocessor()

        # Data placeholders
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Results storage
        self.training_results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None

    def load_and_prepare_data(self, sample_size: Optional[int] = None) -> None:
        """Load and prepare data for training."""
        logger.info("=" * 60)
        logger.info("[STEP 1] LOADING AND PREPARING DATA")
        logger.info("=" * 60)

        # Load data
        train_df, test_df = self.data_loader.load_data()

        # Sample if needed (for faster iteration)
        if sample_size and sample_size < len(train_df):
            train_df = train_df.sample(n=sample_size, random_state=self.random_state)
            logger.info(f"Sampled {sample_size:,} training records")

        # Apply feature engineering
        logger.info("Applying feature engineering...")
        train_df = self.feature_engineer.transform(train_df)
        test_df = self.feature_engineer.transform(test_df)

        # Prepare features
        X, y = prepare_features(
            train_df,
            feature_engineer=self.feature_engineer,
            preprocessor=self.preprocessor,
            is_training=True
        )

        _, y_test_raw = prepare_features(
            test_df,
            feature_engineer=self.feature_engineer,
            preprocessor=self.preprocessor,
            is_training=False
        )

        # Store test data (for final evaluation)
        X_test_full, _ = prepare_features(
            test_df,
            feature_engineer=self.feature_engineer,
            preprocessor=self.preprocessor,
            is_training=False
        )
        self.X_test = X_test_full
        self.y_test = y_test_raw

        # Train-validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )

        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Validation set: {self.X_val.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Training fraud ratio: {self.y_train.mean()*100:.4f}%")

    def train_single_model(
        self,
        model_name: str,
        use_smote: bool = False,
        class_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """Train a single model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}" + (" with SMOTE" if use_smote else ""))
        logger.info(f"{'='*60}")

        start_time = time.time()

        try:
            # Create model
            model = ModelFactory.create(model_name)

            # Apply class weight for imbalanced data
            if class_weight and model_name in ["xgboost", "lgbm"]:
                model.config["scale_pos_weight"] = class_weight

            # Prepare data (apply SMOTE if requested)
            X_train = self.X_train.copy()
            y_train = self.y_train.copy()

            if use_smote:
                smote = SMOTE(random_state=self.random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE: {X_train.shape}, fraud ratio: {y_train.mean()*100:.2f}%")

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            metrics = self._evaluate_model(model, self.X_val, self.y_val, self.X_test, self.y_test)

            training_time = time.time() - start_time
            metrics["training_time"] = training_time

            model_key = f"{model_name}" + ("_SMOTE" if use_smote else "")
            self.training_results[model_key] = metrics

            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"  Val F1: {metrics['val_f1']:.4f}, ROC-AUC: {metrics['val_roc_auc']:.4f}")
            logger.info(f"  Test F1: {metrics['test_f1']:.4f}, ROC-AUC: {metrics['test_roc_auc']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return {"error": str(e)}

    def _evaluate_model(
        self,
        model,
        X_val,
        y_val,
        X_test,
        y_test
    ) -> Dict[str, Any]:
        """Evaluate a trained model."""
        # Predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Validation metrics
        val_metrics = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred),
            "recall": recall_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred),
            "roc_auc": roc_auc_score(y_val, y_val_proba),
            "pr_auc": average_precision_score(y_val, y_val_proba)
        }

        # Test metrics
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
            "roc_auc": roc_auc_score(y_test, y_test_proba),
            "pr_auc": average_precision_score(y_test, y_test_proba)
        }

        return {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"]
        }

    def train_all_models(
        self,
        models: List[str] = None,
        use_smote: bool = True
    ) -> Dict[str, Dict]:
        """Train all configured models."""
        if models is None:
            models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

        # Calculate class weight
        fraud_weight = (self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)

        # Train base models
        for model_name in models:
            self.train_single_model(model_name, use_smote=False, class_weight=fraud_weight)

        # Train with SMOTE
        if use_smote:
            for model_name in models:
                if model_name in ["xgboost", "lightgbm"]:
                    self.train_single_model(model_name, use_smote=True)

        return self.training_results

    def select_best_model(self, metric: str = "test_f1") -> Tuple[str, Any]:
        """Select the best performing model."""
        logger.info("\n" + "=" * 60)
        logger.info("MODEL SELECTION")
        logger.info("=" * 60)

        # Filter out failed models
        valid_results = {
            name: results for name, results in self.training_results.items()
            if "error" not in results
        }

        if not valid_results:
            raise ValueError("No valid models to select from!")

        # Sort by metric
        sorted_models = sorted(
            valid_results.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )

        logger.info(f"\nModel Rankings by {metric}:")
        for i, (name, results) in enumerate(sorted_models, 1):
            logger.info(f"  {i}. {name}: {metric}={results[metric]:.4f}")

        best_name = sorted_models[0][0]
        self.best_model_name = best_name

        logger.info(f"\nBest Model: {best_name}")
        return best_name, sorted_models[0][1]

    def save_model(self, model, name: str = None) -> Path:
        """Save the trained model and related artifacts."""
        if name is None:
            name = self.best_model_name or "model"

        # Save model as model.pkl (required by predict.py)
        model_path = self.model_dir / "model.pkl"
        model.save(str(model_path))
        
        # Also save as named version
        named_model_path = self.model_dir / f"{name}.pkl"
        if str(model_path) != str(named_model_path):
            model.save(str(named_model_path))

        # Save preprocessor and feature engineer
        import pickle
        with open(self.model_dir / "feature_engineer.pkl", "wb") as f:
            pickle.dump(self.feature_engineer, f)
        with open(self.model_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)

        logger.info(f"Model and artifacts saved to {self.model_dir}")
        return model_path

    def save_results(self, filepath: Path = None) -> None:
        """Save training results to CSV and JSON."""
        if filepath is None:
            filepath = self.model_dir / "training_results"

        # Convert to DataFrame
        records = []
        for name, results in self.training_results.items():
            record = {"model": name}
            if "val_metrics" in results:
                record.update({f"val_{k}": v for k, v in results["val_metrics"].items()})
            if "test_metrics" in results:
                record.update({f"test_{k}": v for k, v in results["test_metrics"].items()})
            record["training_time"] = results.get("training_time", 0)
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(f"{filepath}.csv", index=False)

        with open(f"{filepath}.json", "w") as f:
            json.dump(self.training_results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}.csv and {filepath}.json")


def main():
    """Main training pipeline."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )

    from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_DIR

    # Initialize trainer
    trainer = ModelTrainer(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        model_dir=MODEL_DIR,
        random_state=42
    )

    # Load data
    trainer.load_and_prepare_data()

    # Train all models
    trainer.train_all_models(
        models=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        use_smote=True
    )

    # Select best model
    best_name, best_metrics = trainer.select_best_model(metric="test_f1")

    # Save results
    trainer.save_results()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best Model: {best_name}")
    print(f"Test F1: {best_metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC: {best_metrics['test_roc_auc']:.4f}")
    print(f"Test Recall: {best_metrics['test_recall']:.4f}")
    print(f"Test Precision: {best_metrics['test_precision']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
