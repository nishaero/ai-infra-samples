"""Unit tests for API module."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import shutil
from datetime import datetime
import numpy as np

from src.api.main import app, model_manager
from src.api.schemas import PredictionRequest


class TestClimateAPI:
    """Test Climate Prediction API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture(autouse=True)
    def setup_mock_model(self, temp_model_dir):
        """Set up mock model for testing."""
        # Mock the model manager
        mock_model = Mock()
        mock_model.predict.return_value = np.array([25.5])  # Mock temperature prediction
        mock_model.is_fitted = True
        
        # Set up model manager
        model_manager.model_dir = temp_dir
        model_manager.model = mock_model
        model_manager.feature_columns = [
            "longitude", "latitude", "month", "day",
            "month_sin", "month_cos", "day_sin", "day_cos",
            "distance_from_center", "elevation_proxy"
        ]
        model_manager.model_metadata = {
            "model_version": "test_v1.0",
            "loaded_at": datetime.now(),
            "feature_count": 10
        }
        
        # Mock the global model variable
        with patch('src.api.main.model', mock_model):
            with patch('src.api.main.model_info', model_manager.model_metadata):
                yield

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_name"] == "Climate Temperature Ensemble"
        assert data["model_version"] == "test_v1.0"
        assert data["feature_count"] == 10
        assert "training_date" in data

    def test_single_prediction(self, client):
        """Test single prediction endpoint."""
        prediction_data = {
            "longitude": -118.2437,
            "latitude": 34.0522,
            "date": "2023-06-15T00:00:00",
            "precipitation": 0.5,
            "ndvi": 0.3,
            "elevation": 100
        }
        
        response = client.post("/predict", json=prediction_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["longitude"] == prediction_data["longitude"]
        assert data["latitude"] == prediction_data["latitude"]
        assert "predicted_temperature" in data
        assert data["predicted_temperature"] == 25.5
        assert "confidence_interval_lower" in data
        assert "confidence_interval_upper" in data
        assert data["model_version"] == "test_v1.0"

    def test_single_prediction_minimal(self, client):
        """Test single prediction with minimal required fields."""
        prediction_data = {
            "longitude": -118.2437,
            "latitude": 34.0522,
            "date": "2023-06-15T00:00:00"
        }
        
        response = client.post("/predict", json=prediction_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_temperature" in data

    def test_single_prediction_invalid_coordinates(self, client):
        """Test prediction with invalid coordinates."""
        prediction_data = {
            "longitude": 200,  # Invalid longitude
            "latitude": 100,   # Invalid latitude
            "date": "2023-06-15T00:00:00"
        }
        
        response = client.post("/predict", json=prediction_data)
        
        assert response.status_code == 422  # Validation error

    def test_batch_prediction(self, client):
        """Test batch prediction endpoint."""
        batch_data = {
            "predictions": [
                {
                    "longitude": -118.2437,
                    "latitude": 34.0522,
                    "date": "2023-06-15T00:00:00"
                },
                {
                    "longitude": -118.0,
                    "latitude": 34.0,
                    "date": "2023-06-16T00:00:00"
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_predictions"] == 2
        assert len(data["predictions"]) == 2
        assert "processing_time_seconds" in data
        
        # Check individual predictions
        for pred in data["predictions"]:
            assert "predicted_temperature" in pred
            assert pred["predicted_temperature"] == 25.5

    def test_batch_prediction_empty(self, client):
        """Test batch prediction with empty list."""
        batch_data = {"predictions": []}
        
        response = client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 422  # Validation error

    def test_batch_prediction_too_many(self, client):
        """Test batch prediction with too many requests."""
        # Create more than 1000 predictions
        predictions = []
        for i in range(1001):
            predictions.append({
                "longitude": -118.0,
                "latitude": 34.0,
                "date": "2023-06-15T00:00:00"
            })
        
        batch_data = {"predictions": predictions}
        
        response = client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 422  # Validation error

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    @patch('src.api.main.model', None)
    def test_prediction_without_model(self, client):
        """Test prediction when model is not loaded."""
        prediction_data = {
            "longitude": -118.2437,
            "latitude": 34.0522,
            "date": "2023-06-15T00:00:00"
        }
        
        response = client.post("/predict", json=prediction_data)
        
        assert response.status_code == 503  # Service unavailable

    def test_model_reload(self, client):
        """Test model reload endpoint."""
        with patch('src.api.main.model_manager.load_model') as mock_load:
            mock_load.return_value = Mock()
            
            response = client.post("/model/reload")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestPredictionRequest:
    """Test PredictionRequest schema validation."""

    def test_valid_request(self):
        """Test valid prediction request."""
        request_data = {
            "longitude": -118.2437,
            "latitude": 34.0522,
            "date": datetime(2023, 6, 15),
            "precipitation": 0.5,
            "ndvi": 0.3,
            "elevation": 100
        }
        
        request = PredictionRequest(**request_data)
        
        assert request.longitude == -118.2437
        assert request.latitude == 34.0522
        assert request.precipitation == 0.5

    def test_invalid_longitude(self):
        """Test request with invalid longitude."""
        request_data = {
            "longitude": 200,  # Invalid
            "latitude": 34.0522,
            "date": datetime(2023, 6, 15)
        }
        
        with pytest.raises(ValueError):
            PredictionRequest(**request_data)

    def test_invalid_latitude(self):
        """Test request with invalid latitude."""
        request_data = {
            "longitude": -118.2437,
            "latitude": 100,  # Invalid
            "date": datetime(2023, 6, 15)
        }
        
        with pytest.raises(ValueError):
            PredictionRequest(**request_data)

    def test_invalid_ndvi(self):
        """Test request with invalid NDVI."""
        request_data = {
            "longitude": -118.2437,
            "latitude": 34.0522,
            "date": datetime(2023, 6, 15),
            "ndvi": 2.0  # Invalid (should be -1 to 1)
        }
        
        with pytest.raises(ValueError):
            PredictionRequest(**request_data)

    def test_negative_precipitation(self):
        """Test request with negative precipitation."""
        request_data = {
            "longitude": -118.2437,
            "latitude": 34.0522,
            "date": datetime(2023, 6, 15),
            "precipitation": -1.0  # Invalid
        }
        
        with pytest.raises(ValueError):
            PredictionRequest(**request_data)

    def test_minimal_request(self):
        """Test minimal valid request."""
        request_data = {
            "longitude": -118.2437,
            "latitude": 34.0522,
            "date": datetime(2023, 6, 15)
        }
        
        request = PredictionRequest(**request_data)
        
        assert request.longitude == -118.2437
        assert request.latitude == 34.0522
        assert request.precipitation is None
        assert request.ndvi is None
        assert request.elevation is None