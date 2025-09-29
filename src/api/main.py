"""FastAPI Climate Prediction Service.

This module provides a REST API for climate temperature predictions
using the trained ensemble model.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheck,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from src.models.climate_model import SimpleClimatePredictor
from src.monitoring.metrics import SimpleModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "climate_predictions_total", "Total number of predictions made"
)
PREDICTION_LATENCY = Histogram(
    "climate_prediction_duration_seconds", "Time spent on predictions"
)
ERROR_COUNTER = Counter(
    "climate_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)


class ModelManager:
    """Manages model loading and caching."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.model_metadata = {}
        self.feature_columns = []

    def load_model(self, model_path: Optional[str] = None) -> SimpleClimatePredictor:
        """Load the climate prediction model.

        Args:
            model_path: Path to model file. If None, loads latest model.

        Returns:
            Loaded model
        """
        if model_path is None:
            # Find latest model file
            model_files = list(self.model_dir.glob("climate_model_*.joblib"))
            if not model_files:
                raise FileNotFoundError("No model files found")
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading model from {model_path}")

        # Load model
        self.model = SimpleClimatePredictor()
        self.model.load_model(model_path.name)

        # Load feature columns
        features_file = self.model_dir / "features.txt"
        if features_file.exists():
            with open(features_file, "r") as f:
                self.feature_columns = [line.strip() for line in f.readlines()]

        # Set metadata
        self.model_metadata = {
            "model_path": str(model_path),
            "loaded_at": datetime.now(),
            "model_version": model_path.stem,
            "feature_count": len(self.feature_columns),
        }

        logger.info(
            f"Model loaded successfully: {len(self.feature_columns)} features"
        )  # noqa: E501
        return self.model

    def get_model(self) -> SimpleClimatePredictor:
        """Get the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model


# Global model instance
model: Optional[SimpleClimatePredictor] = None
model_info: Dict = {}
monitor: Optional[SimpleModelMonitor] = None
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan."""
    global model, model_info, monitor

    # Startup
    logger.info("Starting Climate Prediction API")

    try:
        # Load model
        global model_manager
        model_manager = ModelManager()
        model = model_manager.load_model()
        model_info = model_manager.model_metadata

        # Initialize monitoring
        monitor = SimpleModelMonitor()

        logger.info("API startup completed successfully")

        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Climate Prediction API")


# Global variables
app = FastAPI(
    title="Climate Temperature Prediction API",
    description="REST API for predicting climate temperature using NASA Earth data",  # noqa: E501
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan."""
    global model, model_info, monitor

    # Startup
    logger.info("Starting Climate Prediction API")

    try:
        # Load model
        global model_manager
        model_manager = ModelManager()
        model = model_manager.load_model()
        model_info = model_manager.model_metadata

        # Initialize monitoring
        monitor = SimpleModelMonitor()

        logger.info("API startup completed successfully")

        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Climate Prediction API")


def get_model() -> SimpleClimatePredictor:
    """Dependency to get the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def create_features_from_request(
    request: PredictionRequest, feature_columns: List[str]
) -> pd.DataFrame:
    """Create feature DataFrame from prediction request.

    Args:
        request: Prediction request
        feature_columns: List of expected feature columns

    Returns:
        Feature DataFrame
    """
    # Base features from request
    features = {
        "longitude": request.longitude,
        "latitude": request.latitude,
        "date": request.date,
    }

    # Add optional features
    if request.precipitation is not None:
        features["precipitation"] = request.precipitation

    if request.ndvi is not None:
        features["ndvi"] = request.ndvi

    if request.elevation is not None:
        features["elevation_proxy"] = request.elevation

    # Create DataFrame
    df = pd.DataFrame([features])

    # Add time-based features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["week"] = df["date"].dt.isocalendar().week

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Add spatial features
    df["distance_from_center"] = 0.0  # Simplified for single point

    # Fill missing features with defaults
    for col in feature_columns:
        if col not in df.columns:
            if "lag" in col or "roll" in col:
                df[col] = 0.0  # Default for lag/roll features
            elif "interaction" in col:
                df[col] = 0.0  # Default for interaction features
            else:
                df[col] = df.get(col.replace("_C", ""), 0.0)

    # Select only required features
    df = df[feature_columns]

    return df


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now(),
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(
    current_model: SimpleClimatePredictor = Depends(get_model),
):  # noqa: E501
    """Get information about the loaded model."""
    return ModelInfo(
        model_name="Climate Temperature Ensemble",
        model_version=model_info.get("model_version", "unknown"),
        training_date=model_info.get("loaded_at", datetime.now()),
        model_metrics={"ensemble_models": len(current_model.models)},
        feature_count=model_info.get("feature_count", 0),
        supported_regions={"global": True, "coordinates": "lat/lon"},
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_temperature(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_model: SimpleClimatePredictor = Depends(get_model),
):
    """Make a single temperature prediction."""
    start_time = time.time()

    try:
        # Create features
        features_df = create_features_from_request(
            request, model_manager.feature_columns
        )

        # Make prediction
        with PREDICTION_LATENCY.time():
            prediction = current_model.predict(features_df)[0]

        # Calculate confidence interval (simplified)
        confidence_margin = 2.0  # ±2°C confidence interval

        response = PredictionResponse(
            longitude=request.longitude,
            latitude=request.latitude,
            date=request.date,
            predicted_temperature=float(prediction),
            confidence_interval_lower=float(prediction - confidence_margin),
            confidence_interval_upper=float(prediction + confidence_margin),
            model_version=model_info.get("model_version", "unknown"),
            prediction_timestamp=datetime.now(),
        )

        # Update metrics
        PREDICTION_COUNTER.inc()

        # Log prediction for monitoring
        if monitor:
            background_tasks.add_task(
                monitor.log_prediction,
                request.dict(),
                {"predicted_temperature": prediction},
                time.time() - start_time,
            )

        return response

    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )  # noqa: E501


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_temperature_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_model: SimpleClimatePredictor = Depends(get_model),
):
    """Make batch temperature predictions."""
    start_time = time.time()

    try:
        predictions = []

        # Process each prediction request
        for pred_request in request.predictions:
            # Create features
            features_df = create_features_from_request(
                pred_request, model_manager.feature_columns
            )

            # Make prediction
            prediction = current_model.predict(features_df)[0]

            # Calculate confidence interval
            confidence_margin = 2.0

            pred_response = PredictionResponse(
                longitude=pred_request.longitude,
                latitude=pred_request.latitude,
                date=pred_request.date,
                predicted_temperature=float(prediction),
                confidence_interval_lower=float(
                    prediction - confidence_margin
                ),  # noqa: E501
                confidence_interval_upper=float(
                    prediction + confidence_margin
                ),  # noqa: E501
                model_version=model_info.get("model_version", "unknown"),
                prediction_timestamp=datetime.now(),
            )

            predictions.append(pred_response)

        processing_time = time.time() - start_time

        # Update metrics
        PREDICTION_COUNTER.inc(len(predictions))

        response = BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time_seconds=processing_time,
        )

        # Log batch prediction for monitoring
        if monitor:
            background_tasks.add_task(
                monitor.log_batch_prediction, len(predictions), processing_time
            )

        return response

    except Exception as e:
        ERROR_COUNTER.labels(error_type="batch_prediction_error").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/model/reload")
async def reload_model():
    """Reload the model (admin endpoint)."""
    global model, model_info

    try:
        model = model_manager.load_model()
        model_info = model_manager.model_metadata

        return {"status": "success", "message": "Model reloaded successfully"}

    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Model reload failed: {str(e)}"
        )  # noqa: E501


def main():
    """Main function to run the API server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Climate Prediction API Server"
    )  # noqa: E501
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--model-dir", default="models", help="Model directory"
    )  # noqa: E501
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )  # noqa: E501

    args = parser.parse_args()

    # Set model directory
    model_manager.model_dir = Path(args.model_dir)

    # Run server
    uvicorn.run(
        "src.api.main:app", host=args.host, port=args.port, reload=args.reload
    )  # noqa: E501


if __name__ == "__main__":
    main()
