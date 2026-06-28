"""
Credit Card Fraud Detection - Model Evaluation Module
==================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.evaluation_results: Dict[str, Dict] = {}

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info(f"Evaluating {model_name}...")

        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)

        # Store results
        self.evaluation_results[model_name] = {
            "metrics": metrics,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_true": y_test
        }

        return metrics

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "tn": int((y_true == 0) & (y_pred == 0)).sum(),
            "fp": int((y_true == 0) & (y_pred == 1)).sum(),
            "fn": int((y_true == 1) & (y_pred == 0)).sum(),
            "tp": int((y_true == 1) & (y_pred == 1)).sum()
        }

    def get_confusion_matrix(self, model_name: str = None) -> np.ndarray:
        """Get confusion matrix for a model."""
        if model_name and model_name in self.evaluation_results:
            cm = self.evaluation_results[model_name]["metrics"]["confusion_matrix"]
            return np.array(cm)
        elif self.evaluation_results:
            first_result = list(self.evaluation_results.values())[0]
            cm = first_result["metrics"]["confusion_matrix"]
            return np.array(cm)
        return None

    def plot_confusion_matrix(
        self,
        model_name: str = None,
        save_path: Path = None,
        normalize: bool = False
    ) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        cm = self.get_confusion_matrix(model_name)
        if cm is None:
            logger.warning("No evaluation results available")
            return None

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {model_name or 'Model'}" + (" (Normalized)" if normalize else ""))

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_roc_curve(
        self,
        model_name: str = None,
        save_path: Path = None
    ) -> plt.Figure:
        """Plot ROC curve."""
        if model_name and model_name not in self.evaluation_results:
            logger.warning(f"Model {model_name} not found")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        for name, result in self.evaluation_results.items():
            if model_name and name != model_name:
                continue

            y_true = result["y_true"]
            y_proba = result["y_proba"]
            roc_auc = result["metrics"]["roc_auc"]

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_precision_recall_curve(
        self,
        model_name: str = None,
        save_path: Path = None
    ) -> plt.Figure:
        """Plot Precision-Recall curve."""
        if model_name and model_name not in self.evaluation_results:
            logger.warning(f"Model {model_name} not found")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        for name, result in self.evaluation_results.items():
            if model_name and name != model_name:
                continue

            y_true = result["y_true"]
            y_proba = result["y_proba"]
            pr_auc = result["metrics"]["pr_auc"]

            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ax.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_threshold_analysis(
        self,
        model_name: str,
        save_path: Path = None
    ) -> plt.Figure:
        """Plot metrics vs threshold."""
        if model_name not in self.evaluation_results:
            logger.warning(f"Model {model_name} not found")
            return None

        result = self.evaluation_results[model_name]
        y_true = result["y_true"]
        y_proba = result["y_proba"]

        # Calculate metrics at different thresholds
        thresholds = np.arange(0.0, 1.01, 0.01)
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(thresholds, precisions)
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("Precision")
        axes[0].set_title("Precision vs Threshold")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(thresholds, recalls)
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Recall")
        axes[1].set_title("Recall vs Threshold")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(thresholds, f1_scores)
        axes[2].set_xlabel("Threshold")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("F1 Score vs Threshold")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models."""
        records = []
        for name, result in self.evaluation_results.items():
            metrics = result["metrics"]
            record = {"model": name}
            record.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values("f1", ascending=False)
        return df

    def print_evaluation_report(self, model_name: str = None) -> str:
        """Generate a comprehensive evaluation report."""
        if model_name:
            if model_name not in self.evaluation_results:
                return f"Model {model_name} not found"
            results = {model_name: self.evaluation_results[model_name]}
        else:
            results = self.evaluation_results

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 70)

        for name, result in results.items():
            metrics = result["metrics"]
            report_lines.append(f"\n{name}:")
            report_lines.append("-" * 50)
            report_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
            report_lines.append(f"  Precision: {metrics['precision']:.4f}")
            report_lines.append(f"  Recall:    {metrics['recall']:.4f}")
            report_lines.append(f"  F1 Score:  {metrics['f1']:.4f}")
            report_lines.append(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            report_lines.append(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
            report_lines.append(f"\n  Confusion Matrix:")
            report_lines.append(f"    TN: {metrics['tn']:,}  FP: {metrics['fp']:,}")
            report_lines.append(f"    FN: {metrics['fn']:,}  TP: {metrics['tp']:,}")

        return "\n".join(report_lines)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float, Dict]:
    """Find the optimal classification threshold."""
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0

    results = {"threshold": [], metric: []}

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        results["threshold"].append(threshold)
        results[metric].append(score)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score, results


def business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_cost: float = 100.0,
    false_alarm_cost: float = 5.0
) -> Dict[str, float]:
    """Calculate business-oriented metrics."""
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    # Money saved by catching fraud
    money_saved = tp * fraud_cost

    # Cost of false alarms
    false_alarm_cost_total = fp * false_alarm_cost

    # Net savings
    net_savings = money_saved - false_alarm_cost_total

    return {
        "fraud_detected": int(tp),
        "fraud_missed": int(fn),
        "false_alarms": int(fp),
        "legitimate_blocked": int(fp),
        "legitimate_approved": int(tn),
        "money_saved": money_saved,
        "false_alarm_cost": false_alarm_cost_total,
        "net_savings": net_savings
    }
