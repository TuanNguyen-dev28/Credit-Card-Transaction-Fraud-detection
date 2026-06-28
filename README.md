# Credit Card Transaction Fraud Detection

A comprehensive machine learning project for detecting credit card fraud using multiple algorithms including Logistic Regression, Random Forest, XGBoost, and LightGBM.

## Features

- Multiple ML models comparison
- Advanced feature engineering
- Real-time fraud detection API
- Interactive Streamlit dashboard
- Comprehensive evaluation metrics

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Train Models
```bash
python main.py train
```

### Run API Server
```bash
python main.py api
```

### Run Dashboard
```bash
python main.py dashboard
```

### Make Predictions
```bash
python main.py predict --input dataset/fraudTest.csv
```

## Project Structure

```
├── app.py              # Flask API
├── dashboard.py        # Streamlit Dashboard
├── main.py            # Main entry point
├── predict.py         # CLI prediction
├── src/               # Source modules
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── models/
└── tests/             # Unit tests
```

## Models

- **Random Forest** - Best overall performance
- **XGBoost** - High ROC-AUC score
- **LightGBM** - Fast training
- **Logistic Regression** - Baseline model

## API Endpoints

- `POST /predict` - Single transaction prediction
- `POST /predict_batch` - Batch predictions
- `GET /model_info` - Model information
- `GET /health` - Health check

## License

MIT License
