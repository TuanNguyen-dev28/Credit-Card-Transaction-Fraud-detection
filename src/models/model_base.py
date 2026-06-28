"""
Credit Card Fraud Detection - Model Implementations
==================================================
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pickle
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for fraud detection models."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def _create_model(self):
        """Create the underlying model instance."""
        pass

    def fit(self, X, y, **kwargs) -> "BaseModel":
        """Train the model."""
        if self.model is None:
            self._create_model()
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Make predictions (class labels)."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet!")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet!")
        return self.model.predict_proba(X)

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model {self.name} saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """Load model from disk."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model {model.name} loaded from {filepath}")
        return model


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for fraud detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1
        }
        if config:
            default_config.update(config)
        super().__init__("Logistic Regression", default_config)

    def _create_model(self):
        self.model = LogisticRegression(**self.config)


class RandomForestModel(BaseModel):
    """Random Forest model for fraud detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "n_estimators": 150,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        if config:
            default_config.update(config)
        super().__init__("Random Forest", default_config)

    def _create_model(self):
        self.model = RandomForestClassifier(**self.config)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")

        importance = self.model.feature_importances_
        feature_names = getattr(self.model, "feature_names_in_", None)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)


class XGBoostModel(BaseModel):
    """XGBoost model for fraud detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "n_estimators": 200,
            "max_depth": 10,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "use_label_encoder": False
        }
        if config:
            default_config.update(config)
        super().__init__("XGBoost", default_config)

    def _create_model(self):
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**self.config)
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")

        importance = self.model.feature_importances_
        feature_names = getattr(self.model, "feature_names_in_", None)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)


class LightGBMModel(BaseModel):
    """LightGBM model for fraud detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "n_estimators": 200,
            "max_depth": 10,
            "learning_rate": 0.1,
            "num_leaves": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        if config:
            default_config.update(config)
        super().__init__("LightGBM", default_config)

    def _create_model(self):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(**self.config)
        except ImportError:
            logger.error("LightGBM not installed. Install with: pip install lightgbm")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")

        importance = self.model.feature_importances_
        feature_names = getattr(self.model, "feature_names_in_", None)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)


class ModelFactory:
    """Factory for creating fraud detection models."""

    _models = {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel
    }

    @classmethod
    def create(cls, model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create a model by name."""
        if model_name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        return cls._models[model_name](config)

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names."""
        return list(cls._models.keys())

    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a new model type."""
        cls._models[name] = model_class
