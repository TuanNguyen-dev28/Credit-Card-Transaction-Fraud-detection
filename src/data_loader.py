"""
Credit Card Fraud Detection - Data Loading Module
=================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial validation of credit card transaction data."""

    def __init__(self, train_path: Path, test_path: Path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test datasets."""
        logger.info(f"Loading training data from {self.train_path}")
        train_df = pd.read_csv(self.train_path, index_col=0)
        logger.info(f"Loaded {len(train_df):,} training records")

        logger.info(f"Loading test data from {self.test_path}")
        test_df = pd.read_csv(self.test_path, index_col=0)
        logger.info(f"Loaded {len(test_df):,} test records")

        return train_df, test_df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset."""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "fraud_ratio": df["is_fraud"].mean() * 100 if "is_fraud" in df.columns else None
        }

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Validate dataset and return issues found."""
        issues = []

        required_cols = ["amt", "is_fraud"]
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")

        if "is_fraud" in df.columns:
            unique_labels = df["is_fraud"].unique()
            if not set(unique_labels).issubset({0, 1}):
                issues.append("Target column 'is_fraud' must contain only 0 and 1")

        if df["amt"].min() < 0:
            issues.append("Negative amounts found")

        if df.isnull().any().any():
            issues.append("Missing values found in dataset")

        return len(issues) == 0, issues


class DataStatistics:
    """Compute and display statistics about the dataset."""

    @staticmethod
    def compute_class_distribution(df: pd.DataFrame, target_col: str = "is_fraud") -> pd.DataFrame:
        """Compute class distribution statistics."""
        counts = df[target_col].value_counts()
        percentages = df[target_col].value_counts(normalize=True) * 100

        return pd.DataFrame({
            "count": counts,
            "percentage": percentages.round(2)
        })

    @staticmethod
    def compute_fraud_statistics(df: pd.DataFrame) -> dict:
        """Compute fraud-specific statistics."""
        fraud_df = df[df["is_fraud"] == 1]
        non_fraud_df = df[df["is_fraud"] == 0]

        stats = {
            "total_transactions": len(df),
            "fraud_transactions": len(fraud_df),
            "legit_transactions": len(non_fraud_df),
            "fraud_ratio": f"{df['is_fraud'].mean()*100:.4f}%",
            "avg_fraud_amount": fraud_df["amt"].mean() if len(fraud_df) > 0 else 0,
            "avg_legit_amount": non_fraud_df["amt"].mean() if len(non_fraud_df) > 0 else 0,
            "max_fraud_amount": fraud_df["amt"].max() if len(fraud_df) > 0 else 0,
            "max_legit_amount": non_fraud_df["amt"].max() if len(non_fraud_df) > 0 else 0
        }

        return stats

    @staticmethod
    def compute_amount_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """Compute amount statistics by class."""
        stats = df.groupby("is_fraud")["amt"].agg([
            "count", "mean", "median", "std", "min", "max", "quantile"
        ]).round(2)
        return stats

    @staticmethod
    def compute_category_fraud_rate(df: pd.DataFrame) -> pd.DataFrame:
        """Compute fraud rate by category."""
        category_stats = df.groupby("category").agg({
            "is_fraud": ["sum", "count", "mean"]
        }).round(4)
        category_stats.columns = ["fraud_count", "total_count", "fraud_rate"]
        category_stats = category_stats.sort_values("fraud_rate", ascending=False)
        return category_stats
