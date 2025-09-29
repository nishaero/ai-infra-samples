"""FastAPI Schemas for Climate Prediction API.

This module defines Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    """Request model for climate predictions."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "longitude": -118.2437,
                "latitude": 34.0522,
                "date": "2023-06-15T00:00:00",
                "precipitation": 0.5,
                "ndvi": 0.3,
                "elevation": 100,
            }
        },
        protected_namespaces=(),
    )

    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    date: datetime = Field(..., description="Date for prediction")

    # Optional environmental features
    precipitation: Optional[float] = Field(
        None, ge=0, description="Precipitation in mm"
    )
    ndvi: Optional[float] = Field(
        None, ge=-1, le=1, description="Normalized Difference Vegetation Index"
    )
    elevation: Optional[float] = Field(None, description="Elevation in meters")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    model_config = ConfigDict(protected_namespaces=())

    predictions: List[PredictionRequest] = Field(
        ..., description="List of prediction requests"
    )

    @field_validator("predictions")
    @classmethod
    def validate_predictions_length(cls, v):
        if len(v) == 0:
            raise ValueError("At least one prediction request is required")
        if len(v) > 1000:
            raise ValueError("Too many predictions requested. Maximum is 1000.")
        return v


class PredictionResponse(BaseModel):
    """Response model for climate predictions."""

    model_config = ConfigDict(protected_namespaces=())

    longitude: float
    latitude: float
    date: datetime
    predicted_temperature: float = Field(
        ..., description="Predicted temperature in Celsius"
    )
    confidence_interval_lower: Optional[float] = Field(
        None, description="Lower bound of 95% confidence interval"
    )
    confidence_interval_upper: Optional[float] = Field(
        None, description="Upper bound of 95% confidence interval"
    )
    model_version: str = Field(..., description="Version of the model used")
    prediction_timestamp: datetime = Field(
        ..., description="When the prediction was made"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    model_config = ConfigDict(protected_namespaces=())

    predictions: List[PredictionResponse]
    total_predictions: int
    processing_time_seconds: float


class ModelInfo(BaseModel):
    """Model information response."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    training_date: datetime
    model_metrics: dict
    feature_count: int
    supported_regions: dict


class HealthCheck(BaseModel):
    """Health check response."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    timestamp: datetime
    model_loaded: bool
    version: str
