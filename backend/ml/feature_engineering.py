"""
Feature Engineering Module
==========================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for fraud detection.
    
    Creates:
    1. Time features (hour, day, weekday, is_night)
    2. Customer features (age)
    3. Location features (distance using Haversine)
    4. Transaction features (log_amount, zscore)
    5. Category features (encoded)
    """

    def __init__(self):
        self.amount_stats: Dict[str, float] = {}
        self.category_stats: Dict[str, float] = {}

    def transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input dataframe
            is_training: If True, compute and save statistics
            
        Returns:
            Transformed dataframe
        """
        df = df.copy()
        
        # 1. Time features
        df = self._create_time_features(df)
        
        # 2. Customer features
        df = self._create_customer_features(df)
        
        # 3. Location features
        df = self._create_location_features(df)
        
        # 4. Transaction features
        df = self._create_transaction_features(df, is_training)
        
        # 5. Category features
        df = self._create_category_features(df, is_training)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if "trans_date_trans_time" not in df.columns:
            return df
        
        # Convert to datetime
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        
        # Extract components
        df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
        df["transaction_day"] = df["trans_date_trans_time"].dt.day
        df["transaction_month"] = df["trans_date_trans_time"].dt.month
        df["transaction_weekday"] = df["trans_date_trans_time"].dt.weekday
        
        # Binary: is_night (00:00 - 05:00)
        df["is_night"] = (df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5)
        df["is_night"] = df["is_night"].astype(int)
        
        # Binary: is_weekend
        df["is_weekend"] = (df["transaction_weekday"] >= 5).astype(int)
        
        return df

    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-based features."""
        if "dob" not in df.columns:
            return df
        
        # Calculate age
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        transaction_date = df.get("trans_date_trans_time", pd.Timestamp.now())
        if not isinstance(transaction_date, pd.Series):
            transaction_date = pd.Series([transaction_date] * len(df))
        
        df["customer_age"] = (
            (transaction_date - df["dob"]).dt.days / 365.25
        ).round(1)
        
        # Fill missing ages with median
        df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())
        
        return df

    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features using Haversine formula."""
        required_cols = ["lat", "long", "merch_lat", "merch_long"]
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Haversine distance (km)
        R = 6371  # Earth's radius in km
        
        lat1 = np.radians(df["lat"])
        lon1 = np.radians(df["long"])
        lat2 = np.radians(df["merch_lat"])
        lon2 = np.radians(df["merch_long"])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        df["distance_customer_merchant"] = R * c
        
        return df

    def _create_transaction_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Create transaction-based features."""
        if "amt" not in df.columns:
            return df
        
        # Log transform
        df["log_amount"] = np.log1p(df["amt"])
        
        # Z-score (during training, compute stats)
        if is_training:
            self.amount_stats["mean"] = df["amt"].mean()
            self.amount_stats["std"] = df["amt"].std()
        
        if self.amount_stats:
            df["amount_zscore"] = (df["amt"] - self.amount_stats["mean"]) / self.amount_stats["std"]
            df["amount_zscore"] = df["amount_zscore"].fillna(0)
        
        return df

    def _create_category_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Create category-based features."""
        if "category" not in df.columns:
            return df
        
        # One-hot encoding for categories
        category_dummies = pd.get_dummies(df["category"], prefix="cat", dummy_na=False)
        df = pd.concat([df, category_dummies], axis=1)
        
        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns after transformation."""
        # Drop non-feature columns
        drop_cols = [
            "trans_date_trans_time", "cc_num", "first", "last",
            "street", "city", "trans_num", "dob"
        ]
        return [col for col in self.feature_columns if col not in drop_cols]
