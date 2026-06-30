"""
Pydantic Schemas for Fraud Detection
=====================================
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Dict, Any


class TransactionSchema(BaseModel):
    """Transaction input schema."""
    amt: float = Field(..., gt=0, description="Transaction amount")
    category: str = Field(..., description="Transaction category")
    merchant: str = Field(..., description="Merchant name")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State/Province code")
    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    trans_date_trans_time: Optional[str] = Field(None, description="Transaction datetime")
    gender: Optional[str] = Field("M", pattern="^[MF]$")
    city_pop: Optional[int] = Field(0, ge=0)
    
    @field_validator('state')
    @classmethod
    def validate_state(cls, v):
        if not v or len(v) < 2:
            raise ValueError('State must be at least 2 characters')
        return v.upper()


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str
    model: str
    anomaly_score: float
    confidence: float
    timestamp: datetime


class BatchRequest(BaseModel):
    """Batch prediction request."""
    transactions: list[TransactionSchema]


class BatchResponse(BaseModel):
    """Batch prediction response."""
    total: int
    fraud_count: int
    normal_count: int
    results: list[PredictionResponse]
