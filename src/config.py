"""
Credit Card Fraud Detection - Configuration
==========================================
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "saved_models"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "fraudTrain.csv"
TEST_DATA_PATH = DATA_DIR / "fraudTest.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model hyperparameters
MODELS_CONFIG = {
    "logistic_regression": {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "random_forest": {
        "n_estimators": 150,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "use_label_encoder": False
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 10,
        "learning_rate": 0.1,
        "num_leaves": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }
}

# SMOTE configuration
SMOTE_CONFIG = {
    "sampling_strategy": 0.1,  # minority:majority ratio
    "random_state": RANDOM_STATE
}

# Feature engineering
FEATURE_ENGINEERING_CONFIG = {
    "time_features": ["hour", "day", "month", "year", "weekday", "is_weekend", "is_night"],
    "distance_based": ["distance", "distance_km"],
    "amount_features": ["amt_log", "amt_sqrt", "is_high_amount", "is_round_amount"],
    "age_features": ["age"],
    "location_features": ["city_pop_log"],
    "recency_features": ["hours_since_last_transaction", "transaction_velocity"]
}

# Categorical encoding
CATEGORICAL_COLS = ["category", "state", "gender"]
DROP_COLS = [
    "trans_date_trans_time", "cc_num", "merchant", "first", "last",
    "street", "city", "job", "dob", "trans_num", "unix_time", "zip"
]

# Columns to keep after preprocessing
NUMERICAL_COLS = [
    "amt", "lat", "long", "city_pop", "merch_lat", "merch_long",
    "trans_hour", "trans_day", "trans_month", "trans_year", "trans_weekday",
    "is_weekend", "is_night", "age", "distance", "amt_log"
]

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "threshold": 0.5  # Default threshold for fraud detection
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "port": 8501,
    "title": "Credit Card Fraud Detection Dashboard"
}

# Evaluation metrics
METRICS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "confusion_matrix", "classification_report"
]

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_DIR / "app.log",
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
