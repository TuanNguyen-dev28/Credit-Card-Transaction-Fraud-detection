"""
Credit Card Fraud Detection - Feature Engineering Module
=========================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles all feature engineering for fraud detection."""

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.amt_stats: Optional[Dict] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to the dataframe."""
        df = df.copy()

        # Datetime conversion
        df = self._convert_datetime(df)

        # Time-based features
        df = self._extract_time_features(df)

        # Age features
        df = self._extract_age_features(df)

        # Distance features
        df = self._extract_distance_features(df)

        # Amount features
        df = self._extract_amount_features(df)

        # Transaction patterns
        df = self._extract_transaction_patterns(df)

        return df

    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns."""
        if "trans_date_trans_time" in df.columns:
            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        if "dob" in df.columns:
            df["dob"] = pd.to_datetime(df["dob"])
        return df

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from transaction datetime."""
        if "trans_date_trans_time" not in df.columns:
            return df

        dt = df["trans_date_trans_time"]

        df["trans_hour"] = dt.dt.hour
        df["trans_day"] = dt.dt.day
        df["trans_month"] = dt.dt.month
        df["trans_year"] = dt.dt.year
        df["trans_weekday"] = dt.dt.weekday
        df["trans_dayofyear"] = dt.dt.dayofyear

        # Binary features
        df["is_weekend"] = (df["trans_weekday"] >= 5).astype(int)
        df["is_night"] = ((df["trans_hour"] >= 22) | (df["trans_hour"] <= 5)).astype(int)
        df["is_morning"] = ((df["trans_hour"] >= 6) & (df["trans_hour"] < 12)).astype(int)
        df["is_afternoon"] = ((df["trans_hour"] >= 12) & (df["trans_hour"] < 18)).astype(int)
        df["is_evening"] = ((df["trans_hour"] >= 18) & (df["trans_hour"] < 22)).astype(int)

        # Time period buckets
        df["time_period"] = pd.cut(
            df["trans_hour"],
            bins=[-1, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"]
        )

        # Quarter of day
        df["quarter_hour"] = (df["trans_hour"] // 6).astype(int)

        return df

    def _extract_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract age-related features."""
        if "dob" not in df.columns or "trans_date_trans_time" not in df.columns:
            return df

        df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25).round(1)

        # Age buckets
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        )

        return df

    def _extract_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract distance-based features."""
        required_cols = ["lat", "long", "merch_lat", "merch_long"]
        if not all(col in df.columns for col in required_cols):
            return df

        # Euclidean distance (approximate)
        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"])**2 +
            (df["long"] - df["merch_long"])**2
        )

        # Haversine distance in km (more accurate)
        lat1 = np.radians(df["lat"])
        lon1 = np.radians(df["long"])
        lat2 = np.radians(df["merch_lat"])
        lon2 = np.radians(df["merch_long"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        df["distance_km"] = 6371 * c  # Earth's radius in km

        # Is unusual distance (potential fraud indicator)
        df["is_long_distance"] = (df["distance_km"] > 100).astype(int)

        return df

    def _extract_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract amount-based features."""
        if "amt" not in df.columns:
            return df

        # Log transform
        df["amt_log"] = np.log1p(df["amt"])

        # Square root transform
        df["amt_sqrt"] = np.sqrt(df["amt"])

        # Is high amount (top 5%)
        if self.amt_stats is None:
            self.amt_stats = {
                "high_threshold": df["amt"].quantile(0.95),
                "low_threshold": df["amt"].quantile(0.05)
            }

        df["is_high_amount"] = (df["amt"] > self.amt_stats["high_threshold"]).astype(int)
        df["is_low_amount"] = (df["amt"] < self.amt_stats["low_threshold"]).astype(int)

        # Is round amount
        df["is_round_amount"] = (df["amt"] % 1 == 0).astype(int)

        # Amount bins
        df["amt_bin"] = pd.qcut(
            df["amt"].clip(lower=0.01),
            q=10,
            labels=["very_low", "low", "below_avg", "avg", "above_avg",
                    "moderately_high", "high", "very_high", "expensive", "luxury"],
            duplicates="drop"
        )

        return df

    def _extract_transaction_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract transaction pattern features."""
        if "cc_num" not in df.columns or "trans_date_trans_time" not in df.columns:
            return df

        # Sort by card number and time
        df = df.sort_values(["cc_num", "trans_date_trans_time"])

        # Transaction count per card (will be computed later with groupby)
        # For single predictions, we use category frequency as proxy

        # Category encoding for frequency
        if "category" in df.columns:
            category_freq = df["category"].value_counts(normalize=True).to_dict()
            df["category_freq"] = df["category"].map(category_freq)

        # State risk (based on fraud rate - computed on training data)
        if "state" in df.columns and "is_fraud" in df.columns:
            state_fraud_rate = df.groupby("state")["is_fraud"].mean()
            df["state_fraud_risk"] = df["state"].map(state_fraud_rate)

        return df


class Preprocessor:
    """Handles preprocessing for model input."""

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.categorical_cols: List[str] = ["category", "state", "gender", "time_period", "age_group", "amt_bin"]

    def fit(self, df: pd.DataFrame):
        """Fit preprocessors on training data."""
        self.feature_names = [col for col in df.columns if col != "is_fraud"]

        # Fit label encoders for categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                le.fit(df[col])
                self.encoders[col] = le

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessors."""
        df = df.copy()

        # Encode categorical columns
        for col in self.categorical_cols:
            if col in df.columns and col in self.encoders:
                df[col] = df[col].astype(str)
                # Handle unseen categories
                known_classes = set(self.encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known_classes else self.encoders[col].classes_[0]
                )
                df[col] = self.encoders[col].transform(df[col])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names after preprocessing."""
        return self.feature_names


def prepare_features(
    df: pd.DataFrame,
    feature_engineer: Optional[FeatureEngineer] = None,
    preprocessor: Optional[Preprocessor] = None,
    is_training: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Complete feature preparation pipeline."""
    logger.info("Starting feature preparation...")

    # Apply feature engineering
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()
        df = feature_engineer.transform(df)
    else:
        df = feature_engineer.transform(df)

    # Drop columns not needed for modeling
    drop_cols = [
        "trans_date_trans_time", "cc_num", "merchant", "first", "last",
        "street", "city", "job", "dob", "trans_num", "unix_time", "zip",
        "lat", "long", "merch_lat", "merch_long"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Separate target
    y = None
    if "is_fraud" in df.columns:
        y = df["is_fraud"]
        df = df.drop(columns=["is_fraud"])

    # Apply preprocessing
    if preprocessor is None:
        preprocessor = Preprocessor()
        df = preprocessor.fit_transform(df)
    else:
        df = preprocessor.transform(df)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    logger.info(f"Feature preparation complete. Shape: {df.shape}")
    return df, y
