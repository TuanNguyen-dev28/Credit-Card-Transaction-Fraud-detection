"""
Credit Card Fraud Detection System - FastAPI Backend
====================================================
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from pathlib import Path

from app.api.routes import router as api_router
from app.database.database import engine, Base
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Fraud Detection API...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    yield
    # Shutdown
    logger.info("Shutting down Fraud Detection API...")


# Create FastAPI app
app = FastAPI(
    title="Vietnamese Credit Card Fraud Detection API",
    description="Anomaly detection system for credit card fraud using Isolation Forest, One-Class SVM, LOF, Z-score, and IQR",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} completed in {process_time:.3f}s")
    return response


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fraud-detection-api", "version": "1.0.0"}


# Include API routes
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
