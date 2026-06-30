"""
Data Preprocessing Module
=========================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing for fraud detection."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: list = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame, target_col: str = "is_fraud") -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        df = df.copy()
        
        # Separate target
        y = df[target_col].values if target_col in df.columns else None
        X = df.drop(columns=[target_col] if target_col in df.columns else [])
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            logger.info(f"Encoded column '{col}': {len(le.classes_)} categories")
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        self.fitted = True
        logger.info(f"Preprocessor fitted. Features: {len(self.feature_columns)}")
        
        return X.values, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed X as numpy array
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
        # Encode categorical columns
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x: x if x in le.classes_ else "unknown"
                )
                # Add unknown class if needed
                if "unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "unknown")
                df[col] = le.transform(df[col])
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df[self.feature_columns].values

    def save(self, path: str):
        """Save preprocessor to file."""
        joblib.dump({
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "fitted": self.fitted
        }, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor from file."""
        data = joblib.load(path)
        preprocessor = cls()
        preprocessor.scaler = data["scaler"]
        preprocessor.label_encoders = data["label_encoders"]
        preprocessor.feature_columns = data["feature_columns"]
        preprocessor.fitted = data["fitted"]
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


def prepare_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets with stratification.
    
    Args:
        X: Features
        y: Target
        test_size: Test set size ratio
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
    logger.info(f"Test set: {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test
