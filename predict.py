"""
Credit Card Fraud Detection - Prediction Script
============================================
"""

import pandas as pd
import numpy as np
import pickle
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json


def load_model_and_preprocessors(model_dir: Path):
    """Load model and preprocessors."""
    model_path = model_dir / "model.pkl"
    fe_path = model_dir / "feature_engineer.pkl"
    preproc_path = model_dir / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(fe_path, "rb") as f:
        feature_engineer = pickle.load(f)

    with open(preproc_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, feature_engineer, preprocessor


def preprocess_transaction(data: dict, feature_engineer, preprocessor) -> pd.DataFrame:
    """Preprocess a single transaction for prediction."""
    df = pd.DataFrame([data])

    # Apply feature engineering
    df = feature_engineer.transform(df)

    # Drop non-feature columns
    drop_cols = [
        "trans_date_trans_time", "cc_num", "merchant", "first", "last",
        "street", "city", "job", "dob", "trans_num", "unix_time", "zip",
        "lat", "long", "merch_lat", "merch_long", "is_fraud"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Apply preprocessor
    df = preprocessor.transform(df)

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


def predict_single(model, feature_engineer, preprocessor, transaction: dict, threshold: float = 0.5) -> dict:
    """Predict fraud for a single transaction."""
    X = preprocess_transaction(transaction, feature_engineer, preprocessor)

    prob = float(model.predict_proba(X)[:, 1][0])
    is_fraud = prob >= threshold

    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": round(prob, 4),
        "confidence": round(max(prob, 1 - prob), 4),
        "threshold": threshold,
        "recommendation": "BLOCK" if is_fraud else "APPROVE"
    }


def predict_batch(model, feature_engineer, preprocessor, transactions: list, threshold: float = 0.5) -> list:
    """Predict fraud for multiple transactions."""
    results = []

    for i, trans in enumerate(transactions):
        try:
            result = predict_single(model, feature_engineer, preprocessor, trans, threshold)
            result["index"] = i
            results.append(result)
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection - Prediction")
    parser.add_argument("--model-dir", type=str, default="saved_models",
                        help="Path to model directory")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file or JSON string with transactions")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV file for results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fraud probability threshold (0-1)")
    parser.add_argument("--format", type=str, choices=["csv", "json"], default="csv",
                        help="Input file format")

    args = parser.parse_args()

    # Load model
    model_dir = Path(args.model_dir)
    print(f"Loading model from {model_dir}...")

    try:
        model, feature_engineer, preprocessor = load_model_and_preprocessors(model_dir)
        print(f"Model loaded: {getattr(model, 'name', 'Unknown')}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training first: python -m src.trainer")
        sys.exit(1)

    # Load transactions
    input_path = Path(args.input)

    if args.format == "csv" or input_path.suffix == ".csv":
        print(f"Loading transactions from {input_path}...")
        df = pd.read_csv(input_path, index_col=0)
        transactions = df.to_dict("records")
    else:
        print(f"Loading transactions from JSON...")
        with open(input_path, "r") as f:
            data = json.load(f)
            transactions = data if isinstance(data, list) else [data]

    print(f"Loaded {len(transactions)} transactions")

    # Predict
    print(f"\nPredicting with threshold={args.threshold}...")
    results = predict_batch(model, feature_engineer, preprocessor, transactions, args.threshold)

    # Summary
    fraud_count = sum(1 for r in results if r.get("is_fraud", False))
    print(f"\n{'='*50}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total transactions: {len(transactions)}")
    print(f"Fraud detected: {fraud_count} ({fraud_count/len(transactions)*100:.2f}%)")
    print(f"Legitimate: {len(transactions) - fraud_count} ({(len(transactions)-fraud_count)/len(transactions)*100:.2f}%)")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Show sample predictions
    print(f"\nSample predictions:")
    print(results_df[["index", "is_fraud", "fraud_probability", "recommendation"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
