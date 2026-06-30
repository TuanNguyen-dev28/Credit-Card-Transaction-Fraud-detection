# Vietnamese Credit Card Fraud Detection System

Hệ thống phát hiện giao dịch thẻ tín dụng bất thường sử dụng các thuật toán Anomaly Detection.

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Frontend**: React + TypeScript + TailwindCSS
- **Database**: MySQL
- **ML**: Scikit-learn, Isolation Forest, One-Class SVM, LOF, Z-score, IQR
- **Deployment**: Docker Compose

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/routes.py      # API endpoints
│   │   ├── services/          # Business logic
│   │   ├── schemas/           # Pydantic models
│   │   ├── database/          # SQLAlchemy models
│   │   └── utils/             # Utilities
│   ├── ml/
│   │   ├── models/            # Anomaly detection models
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/             # Dashboard, Fraud Detection
│   │   ├── services/          # API client
│   │   └── charts/            # Recharts components
│   ├── package.json
│   └── Dockerfile
├── dataset/
│   ├── fraudTrain.csv         # 500k training samples
│   └── fraudTest.csv          # 100k test samples
├── saved_models/              # Trained model files
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis
├── docker-compose.yml
└── README.md
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up --build

# Backend API: http://localhost:8000
# Frontend Dashboard: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Manual Setup

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend
npm install
npm start
```

#### Database

```bash
# Start PostgreSQL
docker run -d \
  --name fraud_db \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=fraud_detection \
  -p 5432:5432 \
  postgres:15-alpine
```

## API Endpoints

- `POST /api/v1/detect` - Detect fraud for single transaction
- `POST /api/v1/detect/batch` - Batch fraud detection
- `GET /api/v1/models` - Get models info
- `GET /api/v1/stats` - Get dataset statistics
- `GET /health` - Health check

## ML Models

1. **Isolation Forest** - Tree-based anomaly detection
2. **One-Class SVM** - Support vector machine for novelty detection
3. **LOF (Local Outlier Factor)** - Density-based anomaly detection
4. **Z-score** - Statistical method
5. **IQR** - Interquartile range method

## Dataset

- **Training**: 500,000 transactions (5% fraud rate)
- **Test**: 100,000 transactions
- **Features**: 22 columns including transaction details, location, customer info

## Features

### Phase 1: Data Analysis
- [x] EDA notebook with visualizations
- [x] Data validation (shape, missing values, duplicates)
- [x] Fraud distribution analysis
- [x] Correlation analysis

### Phase 2: Feature Engineering
- [x] Time features (hour, day, weekday, is_night)
- [x] Customer features (age)
- [x] Location features (Haversine distance)
- [x] Transaction features (log_amount, zscore)
- [x] Category encoding

### Phase 3: Statistical Detection
- [x] Z-score method (|z| > 3)
- [x] IQR method (outside Q1-1.5*IQR, Q3+1.5*IQR)

### Phase 4: ML Models
- [x] Isolation Forest
- [x] One-Class SVM
- [x] LOF

### Phase 5: Evaluation
- [x] Precision, Recall, F1, ROC-AUC, PR-AUC
- [x] Confusion Matrix
- [x] Model comparison plots

### Phase 6: FastAPI Backend
- [x] REST API endpoints
- [x] PostgreSQL integration
- [x] Request validation
- [x] Error handling & logging

### Phase 7: React Dashboard
- [x] Dashboard with statistics
- [x] Fraud detection form
- [x] Real-time results
- [x] Charts (Pie, Bar, Line)

### Phase 8: Docker
- [x] Docker Compose setup
- [x] Multi-service orchestration

## License

MIT
