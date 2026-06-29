"""
Credit Card Fraud Detection - Flask API
======================================
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.feature_engineering import FeatureEngineer, Preprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC VALIDATION SCHEMAS
# =============================================================================

class TransactionModel(BaseModel):
    """Schema for single transaction validation."""
    trans_date_trans_time: str = Field(..., description="Transaction datetime")
    cc_num: str = Field(..., min_length=1, description="Credit card number")
    merchant: str = Field(..., min_length=1, description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., gt=0, description="Transaction amount (must be > 0)")
    first: str = Field(default="", description="First name")
    last: str = Field(default="", description="Last name")
    gender: str = Field(default="M", pattern="^[MF]$", description="Gender: M or F")
    street: str = Field(default="", description="Street address")
    city: str = Field(default="", description="City")
    state: str = Field(default="NY", min_length=2, max_length=2, description="State code (2 letters)")
    zip: str = Field(default="00000", description="ZIP code")
    lat: float = Field(default=40.7128, description="Cardholder latitude")
    long: float = Field(default=-74.0060, description="Cardholder longitude")
    city_pop: int = Field(default=0, ge=0, description="City population")
    job: str = Field(default="", description="Job title")
    dob: str = Field(default="1990-01-01", description="Date of birth")
    trans_num: str = Field(default="", description="Transaction number")
    unix_time: int = Field(default=0, description="Unix timestamp")
    merch_lat: float = Field(default=40.7128, description="Merchant latitude")
    merch_long: float = Field(default=-74.0060, description="Merchant longitude")

    @field_validator('state')
    @classmethod
    def validate_state(cls, v):
        if len(v) != 2:
            raise ValueError('State must be 2-letter code (e.g., NY, CA)')
        return v.upper()

    @field_validator('amt')
    @classmethod
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:
            raise ValueError('Amount exceeds maximum allowed (1,000,000)')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "trans_date_trans_time": "2024-01-15 10:30:00",
                "cc_num": "1234567890123456",
                "merchant": "Amazon",
                "category": "shopping_net",
                "amt": 50.0,
                "first": "John",
                "last": "Doe",
                "gender": "M",
                "state": "NY",
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 40.7580,
                "merch_long": -73.9855,
                "dob": "1990-01-01"
            }
        }


class BatchTransactionRequest(BaseModel):
    """Schema for batch transaction validation."""
    transactions: List[TransactionModel] = Field(..., min_length=1, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "trans_date_trans_time": "2024-01-15 10:30:00",
                        "cc_num": "1234567890123456",
                        "merchant": "Amazon",
                        "category": "shopping_net",
                        "amt": 50.0
                    }
                ]
            }
        }


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "saved_models"

# Global model and preprocessors
model = None
feature_engineer = None
preprocessor = None
model_name = None


def load_model():
    """Load the trained model and preprocessors."""
    global model, feature_engineer, preprocessor, model_name

    model_path = MODEL_DIR / "model.pkl"
    fe_path = MODEL_DIR / "feature_engineer.pkl"
    preproc_path = MODEL_DIR / "preprocessor.pkl"

    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        return False

    try:
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_name = getattr(model, "name", "Unknown Model")

        # Load preprocessors
        if fe_path.exists():
            with open(fe_path, "rb") as f:
                feature_engineer = pickle.load(f)

        if preproc_path.exists():
            with open(preproc_path, "rb") as f:
                preprocessor = pickle.load(f)

        logger.info(f"Model '{model_name}' loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def preprocess_transaction(data: dict) -> pd.DataFrame:
    """Preprocess a single transaction for prediction."""
    df = pd.DataFrame([data])

    # Apply feature engineering
    if feature_engineer:
        df = feature_engineer.transform(df)

    # Drop non-feature columns
    drop_cols = [
        "trans_date_trans_time", "cc_num", "merchant", "first", "last",
        "street", "city", "job", "dob", "trans_num", "unix_time", "zip",
        "lat", "long", "merch_lat", "merch_long", "is_fraud"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Apply preprocessor
    if preprocessor:
        df = preprocessor.transform(df)

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


@app.route("/")
def index():
    """Home page."""
    return render_template("index.html", model_name=model_name)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict fraud for a single transaction."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Validate input with Pydantic
        try:
            data = TransactionModel(**request.get_json())
            data_dict = data.model_dump()
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return jsonify({
                "error": "Validation failed",
                "details": str(e),
                "hint": "Check required fields: amt (>0), category, merchant, cc_num, trans_date_trans_time"
            }), 400

        # Preprocess
        X = preprocess_transaction(data_dict)

        # Predict
        fraud_probability = float(model.predict_proba(X)[:, 1][0])
        is_fraud = fraud_probability >= 0.5

        result = {
            "success": True,
            "prediction": {
                "is_fraud": bool(is_fraud),
                "fraud_probability": round(fraud_probability, 4),
                "confidence": round(max(fraud_probability, 1 - fraud_probability), 4)
            },
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Predict fraud for multiple transactions."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Validate input with Pydantic
        try:
            batch_request = BatchTransactionRequest(**data)
            transactions = [t.model_dump() for t in batch_request.transactions]
        except Exception as e:
            logger.warning(f"Batch validation error: {e}")
            return jsonify({
                "error": "Batch validation failed",
                "details": str(e),
                "hint": "Ensure 'transactions' is a non-empty list with valid transaction objects"
            }), 400

        results = []

        for i, trans in enumerate(transactions):
            try:
                X = preprocess_transaction(trans)
                fraud_probability = float(model.predict_proba(X)[:, 1][0])
                results.append({
                    "index": i,
                    "is_fraud": bool(fraud_probability >= 0.5),
                    "fraud_probability": round(fraud_probability, 4)
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e)
                })

        fraud_count = sum(1 for r in results if r.get("is_fraud", False))

        return jsonify({
            "success": True,
            "total_transactions": len(transactions),
            "fraud_detected": fraud_count,
            "legitimate": len(transactions) - fraud_count,
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    info = {
        "model_name": model_name,
        "is_fitted": getattr(model, "is_fitted", False),
        "model_type": type(model.model).__name__ if hasattr(model, "model") else "Unknown"
    }

    return jsonify(info)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/test", methods=["GET"])
def test():
    """Test endpoint with sample fraud/legitimate transactions."""
    test_cases = [
        {
            "name": "Legitimate Transaction",
            "data": {
                "trans_date_trans_time": "2024-01-15 10:30:00",
                "cc_num": "1234567890123456",
                "merchant": "Amazon",
                "category": "shopping_net",
                "amt": 50.0,
                "first": "John",
                "last": "Doe",
                "gender": "M",
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001",
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 8000000,
                "job": "Software Engineer",
                "dob": "1990-01-01",
                "trans_num": "test123",
                "unix_time": 1705315800,
                "merch_lat": 40.7580,
                "merch_long": -73.9855
            }
        },
        {
            "name": "Suspicious Transaction",
            "data": {
                "trans_date_trans_time": "2024-01-15 03:00:00",
                "cc_num": "1234567890123456",
                "merchant": "Unknown Store",
                "category": "misc_net",
                "amt": 9999.0,
                "first": "John",
                "last": "Doe",
                "gender": "M",
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001",
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 8000000,
                "job": "Software Engineer",
                "dob": "1990-01-01",
                "trans_num": "test456",
                "unix_time": 1705294800,
                "merch_lat": 34.0522,
                "merch_long": -118.2437
            }
        }
    ]

    results = []
    for tc in test_cases:
        try:
            X = preprocess_transaction(tc["data"])
            prob = float(model.predict_proba(X)[:, 1][0])
            results.append({
                "name": tc["name"],
                "fraud_probability": round(prob, 4),
                "is_fraud": prob >= 0.5
            })
        except Exception as e:
            results.append({
                "name": tc["name"],
                "error": str(e)
            })

    return jsonify({
        "success": True,
        "test_results": results
    })


if __name__ == "__main__":
    # Load model on startup
    load_model()

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
