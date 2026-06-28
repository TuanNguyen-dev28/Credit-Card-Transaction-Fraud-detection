"""
Credit Card Fraud Detection - Test Suite
======================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader, DataStatistics
from src.feature_engineering import FeatureEngineer, Preprocessor, prepare_features
from src.models import (
    LogisticRegressionModel,
    RandomForestModel,
    ModelFactory
)


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing."""
    data = {
        "trans_date_trans_time": pd.to_datetime(["2024-01-15 10:30:00", "2024-01-15 11:30:00"]),
        "cc_num": ["1234567890123456", "9876543210987654"],
        "merchant": ["Amazon", "Walmart"],
        "category": ["shopping_net", "grocery_pos"],
        "amt": [150.0, 50.0],
        "first": ["John", "Jane"],
        "last": ["Doe", "Smith"],
        "gender": ["M", "F"],
        "street": ["123 Main St", "456 Oak Ave"],
        "city": ["New York", "Los Angeles"],
        "state": ["NY", "CA"],
        "zip": ["10001", "90001"],
        "lat": [40.7128, 34.0522],
        "long": [-74.0060, -118.2437],
        "city_pop": [8000000, 4000000],
        "job": ["Engineer", "Teacher"],
        "dob": pd.to_datetime(["1990-01-01", "1985-05-15"]),
        "trans_num": ["abc123", "def456"],
        "unix_time": [1705315800, 1705319400],
        "merch_lat": [40.7580, 34.0522],
        "merch_long": [-73.9855, -118.2437],
        "is_fraud": [0, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


@pytest.fixture
def preprocessor():
    return Preprocessor()


class TestDataLoader:
    """Test data loading functionality."""

    def test_data_loader_initialization(self):
        """Test DataLoader can be initialized."""
        loader = DataLoader(
            train_path=Path("dataset/fraudTrain.csv"),
            test_path=Path("dataset/fraudTest.csv")
        )
        assert loader is not None

    def test_get_data_info(self, sample_data):
        """Test data info extraction."""
        loader = DataLoader(
            train_path=Path("dataset/fraudTrain.csv"),
            test_path=Path("dataset/fraudTest.csv")
        )
        info = loader.get_data_info(sample_data)
        assert "shape" in info
        assert info["shape"] == (2, 22)


class TestFeatureEngineering:
    """Test feature engineering."""

    def test_convert_datetime(self, sample_data, feature_engineer):
        """Test datetime conversion."""
        result = feature_engineer._convert_datetime(sample_data.copy())
        assert pd.api.types.is_datetime64_any_dtype(result["trans_date_trans_time"])
        assert pd.api.types.is_datetime64_any_dtype(result["dob"])

    def test_extract_time_features(self, sample_data, feature_engineer):
        """Test time feature extraction."""
        df = feature_engineer._convert_datetime(sample_data.copy())
        result = feature_engineer._extract_time_features(df)
        assert "trans_hour" in result.columns
        assert "trans_weekday" in result.columns
        assert "is_weekend" in result.columns

    def test_extract_distance_features(self, sample_data, feature_engineer):
        """Test distance feature extraction."""
        result = feature_engineer._extract_distance_features(sample_data.copy())
        assert "distance" in result.columns
        assert "distance_km" in result.columns
        assert (result["distance"] >= 0).all()

    def test_extract_amount_features(self, sample_data, feature_engineer):
        """Test amount feature extraction."""
        result = feature_engineer._extract_amount_features(sample_data.copy())
        assert "amt_log" in result.columns
        assert "amt_sqrt" in result.columns
        assert "is_round_amount" in result.columns

    def test_transform_pipeline(self, sample_data, feature_engineer):
        """Test complete transformation pipeline."""
        result = feature_engineer.transform(sample_data.copy())
        assert len(result) == len(sample_data)
        assert "trans_hour" in result.columns
        assert "distance" in result.columns


class TestPreprocessor:
    """Test preprocessing."""

    def test_fit_transform(self, sample_data, preprocessor):
        """Test fitting and transforming."""
        df = sample_data.copy()
        result = preprocessor.fit_transform(df)
        assert len(result) == len(df)
        assert preprocessor.encoders is not None


class TestModels:
    """Test model implementations."""

    def test_model_factory(self):
        """Test model factory."""
        lr = ModelFactory.create("logistic_regression")
        assert isinstance(lr, LogisticRegressionModel)

        rf = ModelFactory.create("random_forest")
        assert isinstance(rf, RandomForestModel)

    def test_logistic_regression_fit(self, sample_data, feature_engineer, preprocessor):
        """Test Logistic Regression training."""
        df = sample_data.copy()
        df = feature_engineer.transform(df)

        X, y = prepare_features(df, is_training=True)
        X = X.astype(float)

        model = LogisticRegressionModel()
        model.fit(X, y)

        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_random_forest_fit(self, sample_data, feature_engineer, preprocessor):
        """Test Random Forest training."""
        df = sample_data.copy()
        df = feature_engineer.transform(df)

        X, y = prepare_features(df, is_training=True)
        X = X.astype(float)

        model = RandomForestModel()
        model.fit(X, y)

        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_model_predict_proba(self, sample_data, feature_engineer):
        """Test probability predictions."""
        df = sample_data.copy()
        df = feature_engineer.transform(df)

        X, y = prepare_features(df, is_training=True)
        X = X.astype(float)

        model = LogisticRegressionModel()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape[0] == len(y)
        assert (proba >= 0).all()
        assert (proba <= 1).all()


class TestDataStatistics:
    """Test statistics computation."""

    def test_class_distribution(self, sample_data):
        """Test class distribution computation."""
        dist = DataStatistics.compute_class_distribution(sample_data)
        assert "count" in dist.columns
        assert "percentage" in dist.columns

    def test_fraud_statistics(self, sample_data):
        """Test fraud statistics computation."""
        stats = DataStatistics.compute_fraud_statistics(sample_data)
        assert "total_transactions" in stats
        assert "fraud_transactions" in stats
        assert "fraud_ratio" in stats

    def test_category_fraud_rate(self, sample_data):
        """Test category fraud rate computation."""
        result = DataStatistics.compute_category_fraud_rate(sample_data)
        assert "fraud_rate" in result.columns
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
