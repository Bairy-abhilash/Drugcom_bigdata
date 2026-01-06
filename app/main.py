"""
Main FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.api.routes import router
from app.utils.config import settings, config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Drug Synergy Prediction API",
    description="AI-powered drug combination therapy prediction using Graph Neural Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
origins = config.get('api.cors_origins', ['*'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["api"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Drug Synergy Prediction API")
    logger.info(f"Database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    logger.info(f"Model path: {settings.MODEL_PATH}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Drug Synergy Prediction API")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Drug Synergy Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "api": "/api/v1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )