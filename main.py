"""
Credit Card Fraud Detection - Main Entry Point
==========================================
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python main.py train

  # Run API server
  python main.py api

  # Run dashboard
  python main.py dashboard

  # Make predictions
  python main.py predict --input data/test.csv

  # Show data statistics
  python main.py stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train fraud detection models")
    train_parser.add_argument("--sample", type=int, default=None,
                              help="Sample size for training (for faster iteration)")
    train_parser.add_argument("--models", nargs="+",
                              default=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
                              help="Models to train")
    train_parser.add_argument("--no-smote", action="store_true",
                              help="Disable SMOTE oversampling")

    # API command
    api_parser = subparsers.add_parser("api", help="Start Flask API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    api_parser.add_argument("--port", type=int, default=5000, help="Port to bind")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start Streamlit dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8501, help="Port to bind")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions on data")
    predict_parser.add_argument("--input", required=True, help="Input CSV file")
    predict_parser.add_argument("--output", default="predictions.csv", help="Output file")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Fraud threshold")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--sample", type=int, default=10000, help="Sample size")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "train":
        from src.trainer import ModelTrainer
        from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_DIR

        print("=" * 60)
        print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
        print("=" * 60)

        trainer = ModelTrainer(
            train_path=TRAIN_DATA_PATH,
            test_path=TEST_DATA_PATH,
            model_dir=MODEL_DIR,
            random_state=42
        )

        trainer.load_and_prepare_data(sample_size=args.sample)
        trainer.train_all_models(
            models=args.models,
            use_smote=not args.no_smote
        )

        best_name, best_metrics = trainer.select_best_model(metric="test_f1")
        trainer.save_results()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print(f"Best Model: {best_name}")
        print(f"Test F1: {best_metrics['test_f1']:.4f}")
        print("=" * 60)

    elif args.command == "api":
        from app import app, load_model

        print(f"Starting API server on {args.host}:{args.port}")
        load_model()
        app.run(host=args.host, port=args.port, debug=True)

    elif args.command == "dashboard":
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "dashboard.py",
            "--server.port", str(args.port),
            "--server.headless", "true"
        ])

    elif args.command == "predict":
        from predict import main as predict_main
        sys.argv = ["predict.py", "--input", args.input, "--output", args.output, "--threshold", str(args.threshold)]
        predict_main()

    elif args.command == "stats":
        import pandas as pd
        from src.data_loader import DataLoader, DataStatistics
        from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH

        print("=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        loader = DataLoader(TRAIN_DATA_PATH, TEST_DATA_PATH)
        train_df, test_df = loader.load_data()

        if args.sample:
            train_df = train_df.sample(n=min(args.sample, len(train_df)), random_state=42)
            test_df = test_df.sample(n=min(args.sample, len(test_df)), random_state=42)

        print(f"\n--- Training Set ---")
        print(f"Shape: {train_df.shape}")
        stats = DataStatistics.compute_fraud_statistics(train_df)
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print(f"\n--- Test Set ---")
        print(f"Shape: {test_df.shape}")
        stats = DataStatistics.compute_fraud_statistics(test_df)
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
