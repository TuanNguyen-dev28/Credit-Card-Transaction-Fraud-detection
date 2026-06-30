"""
Model Evaluation Module
=======================
Compare all anomaly detection models.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare multiple anomaly detection models."""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent.parent.parent / "saved_models"
        else:
            self.model_dir = Path(model_dir)
        
        self.results: Dict[str, Dict] = {}
        self.models: Dict[str, any] = {}

    def load_models(self) -> Dict[str, any]:
        """Load all saved models."""
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
                logger.warning(f"Model {name} not found at {self.model_dir / filename}")
        
        return self.models

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with results for all models
        """
        logger.info("="*60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*60)
        
        self.load_models()
        
        results = []
        
        # Evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            metrics = self._evaluate_single(model, X_test, y_test, model_name)
            self.results[model_name] = metrics
            results.append({
                "Model": model_name,
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1": round(metrics["f1"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
                "PR-AUC": round(metrics["pr_auc"], 4)
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("F1", ascending=False)
        
        # Save results
        results_path = self.model_dir / "model_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("MODEL RANKINGS BY F1-SCORE")
        logger.info("="*60)
        for i, row in results_df.iterrows():
            logger.info(f"{i+1}. {row['Model']}: F1={row['F1']:.4f}, ROC-AUC={row['ROC-AUC']:.4f}")
        
        best_model = results_df.iloc[0]["Model"]
        logger.info(f"\nBest Model: {best_model}")
        
        return results_df

    def _evaluate_single(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict:
        """Evaluate a single model."""
        try:
            y_pred, y_scores = model.predict(X_test)
            
            # Convert -1/1 to 0/1
            y_pred_binary = np.where(y_pred == -1, 1, 0)
            
            # Normalize scores to [0, 1]
            y_scores_norm = np.clip(y_scores, 0, 1)
            
            metrics = {
                "precision": precision_score(y_test, y_pred_binary, zero_division=0),
                "recall": recall_score(y_test, y_pred_binary, zero_division=0),
                "f1": f1_score(y_test, y_pred_binary, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_scores_norm),
                "pr_auc": average_precision_score(y_test, y_scores_norm),
                "confusion_matrix": confusion_matrix(y_test, y_pred_binary).tolist(),
                "classification_report": classification_report(y_test, y_pred_binary, output_dict=True)
            }
            
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {"error": str(e)}

    def plot_comparison(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot model comparison charts.
        
        Args:
            results_df: Results DataFrame
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. F1 Score Comparison
        axes[0, 0].barh(results_df["Model"], results_df["F1"], color='steelblue')
        axes[0, 0].set_xlabel("F1 Score")
        axes[0, 0].set_title("F1 Score Comparison")
        axes[0, 0].set_xlim(0, 1)
        
        # 2. ROC-AUC Comparison
        axes[0, 1].barh(results_df["Model"], results_df["ROC-AUC"], color='coral')
        axes[0, 1].set_xlabel("ROC-AUC")
        axes[0, 1].set_title("ROC-AUC Comparison")
        axes[0, 1].set_xlim(0, 1)
        
        # 3. Precision vs Recall
        axes[1, 0].scatter(results_df["Recall"], results_df["Precision"], s=100, c='green')
        for i, model in enumerate(results_df["Model"]):
            axes[1, 0].annotate(model, (results_df["Recall"].iloc[i], results_df["Precision"].iloc[i]))
        axes[1, 0].set_xlabel("Recall")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision vs Recall")
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        
        # 4. PR-AUC
        axes[1, 1].barh(results_df["Model"], results_df["PR-AUC"], color='purple')
        axes[1, 1].set_xlabel("PR-AUC")
        axes[1, 1].set_title("PR-AUC Comparison")
        axes[1, 1].set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()

    def plot_confusion_matrices(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None):
        """
        Plot confusion matrices for all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Optional path to save figure
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            try:
                y_pred, _ = model.predict(X_test)
                y_pred_binary = np.where(y_pred == -1, 1, 0)
                cm = confusion_matrix(y_test, y_pred_binary)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{model_name}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            except Exception as e:
                logger.error(f"Error plotting confusion matrix for {model_name}: {e}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()

    def save_report(self, output_path: str = None):
        """
        Save detailed evaluation report.
        
        Args:
            output_path: Path to save report (default: model_dir/evaluation_report.json)
        """
        if output_path is None:
            output_path = self.model_dir / "evaluation_report.json"
        
        # Convert numpy types to Python types for JSON serialization
        report = {}
        for model_name, metrics in self.results.items():
            report[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    report[model_name][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    report[model_name][key] = float(value)
                else:
                    report[model_name][key] = value
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")


def main():
    """Main evaluation pipeline."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    from app.ml.load_data import DataLoader
    from app.ml.preprocessing import DataPreprocessor
    from app.ml.feature_engineering import FeatureEngineer
    
    data_loader = DataLoader()
    train_df, test_df = data_loader.load_data(sample_size=100000)
    
    # Feature engineering
    fe = FeatureEngineer()
    train_df = fe.transform(train_df, is_training=True)
    test_df = fe.transform(test_df, is_training=False)
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df, target_col="is_fraud")
    X_test, y_test = preprocessor.transform(test_df.drop(columns=["is_fraud"])), test_df["is_fraud"].values
    
    # Evaluate
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_all(X_test, y_test)
    
    # Plot
    evaluator.plot_comparison(results_df)
    evaluator.plot_confusion_matrices(X_test, y_test)
    evaluator.save_report()
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE!")
    print("="*50)
