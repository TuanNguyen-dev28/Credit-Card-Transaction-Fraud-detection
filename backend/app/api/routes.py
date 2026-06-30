"""
API Routes for Fraud Detection System
======================================
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from app.services.fraud_service import FraudDetectionService
from app.database.models import Transaction

logger = logging.getLogger(__name__)
router = APIRouter()

# Global service instance
fraud_service = FraudDetectionService()


# Request/Response Models
class TransactionRequest(BaseModel):
    """Transaction input schema."""
    amt: float = Field(..., gt=0, description="Transaction amount")
    category: str = Field(..., description="Transaction category")
    merchant: str = Field(..., description="Merchant name")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State/Province")
    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    trans_date_trans_time: Optional[str] = Field(None, description="Transaction datetime")
    gender: Optional[str] = Field("M", pattern="^[MF]$")
    city_pop: Optional[int] = Field(0, ge=0)


class BatchTransactionRequest(BaseModel):
    """Batch transaction request."""
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class DetectionResponse(BaseModel):
    """Detection result response."""
    prediction: str = Field(..., description="fraud or normal")
    model: str = Field(..., description="Model used for detection")
    anomaly_score: float = Field(..., description="Anomaly score (0-1)")
    confidence: float = Field(..., description="Confidence level")
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchDetectionResponse(BaseModel):
    """Batch detection response."""
    total: int
    fraud_count: int
    normal_count: int
    results: List[DetectionResponse]
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    """Model information."""
    models: List[Dict[str, Any]]
    best_model: Optional[str] = None


@router.post("/detect", response_model=DetectionResponse)
async def detect_fraud(transaction: TransactionRequest):
    """
    Detect fraud for a single transaction using all available models.
    
    Returns prediction from the best performing model.
    """
    try:
        logger.info(f"Received detection request for amount: {transaction.amt}")
        
        result = fraud_service.predict_single(transaction.model_dump())
        
        return DetectionResponse(
            prediction=result["prediction"],
            model=result["model"],
            anomaly_score=result["anomaly_score"],
            confidence=result["confidence"],
            details=result.get("details", {}),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_fraud_batch(request: BatchTransactionRequest):
    """
    Detect fraud for multiple transactions.
    """
    try:
        logger.info(f"Received batch detection request for {len(request.transactions)} transactions")
        
        start_time = time.time()
        results = fraud_service.predict_batch([t.model_dump() for t in request.transactions])
        processing_time = (time.time() - start_time) * 1000
        
        fraud_count = sum(1 for r in results if r["prediction"] == "fraud")
        
        return BatchDetectionResponse(
            total=len(results),
            fraud_count=fraud_count,
            normal_count=len(results) - fraud_count,
            results=[DetectionResponse(**r) for r in results],
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelInfoResponse)
async def get_models_info():
    """
    Get information about all available models and their performance.
    """
    try:
        info = fraud_service.get_models_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics():
    """
    Get dataset statistics and model performance metrics.
    """
    try:
        stats = fraud_service.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
