"""
Data Loading Module
==================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate fraud detection dataset."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent.parent / "dataset"
        else:
            self.data_dir = Path(data_dir)
        
        self.train_path = self.data_dir / "fraudTrain.csv"
        self.test_path = self.data_dir / "fraudTest.csv"

    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets.
        
        Args:
            sample_size: If provided, sample this many rows from training data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Loading training data from {self.train_path}")
        train_df = pd.read_csv(self.train_path)
        logger.info(f"Loaded {len(train_df):,} training records")
        
        if sample_size and sample_size < len(train_df):
            train_df = train_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size:,} training records")
        
        logger.info(f"Loading test data from {self.test_path}")
        test_df = pd.read_csv(self.test_path)
        logger.info(f"Loaded {len(test_df):,} test records")
        
        return train_df, test_df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate dataset integrity.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Check required columns
        required_cols = [
            "trans_date_trans_time", "amt", "category", "merchant",
            "state", "is_fraud", "lat", "long", "merch_lat", "merch_long"
        ]
        results["missing_columns"] = [col for col in required_cols if col not in df.columns]
        results["fraud_rate"] = float(df["is_fraud"].mean()) if "is_fraud" in df.columns else None
        
        return results


def get_data_info(df: pd.DataFrame) -> dict:
    """Get comprehensive dataset information."""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }
    
    if "is_fraud" in df.columns:
        info["fraud_distribution"] = {
            "fraud": int(df["is_fraud"].sum()),
            "normal": int((df["is_fraud"] == 0).sum()),
            "fraud_rate": round(df["is_fraud"].mean() * 100, 2)
        }
    
    if "amt" in df.columns:
        info["amount_stats"] = {
            "mean": round(df["amt"].mean(), 2),
            "median": round(df["amt"].median(), 2),
            "std": round(df["amt"].std(), 2),
            "min": round(df["amt"].min(), 2),
            "max": round(df["amt"].max(), 2)
        }
    
    return info
