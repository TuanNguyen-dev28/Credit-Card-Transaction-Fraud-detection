"""
Database Models
==============
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.sql import func
from app.database.database import Base


class Transaction(Base):
    """Transaction history table."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    category = Column(String(50), nullable=False)
    merchant = Column(String(100), nullable=False)
    city = Column(String(100), nullable=False)
    prediction = Column(String(20), nullable=False)  # fraud or normal
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Transaction(id={self.id}, amount={self.amount}, prediction={self.prediction})>"
