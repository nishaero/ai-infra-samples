"""Unit tests for ML models module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import tempfile
import shutil
from pathlib import Path

from src.models.climate_model import ClimatePredictor, evaluate_model


class TestClimatePredictor:
    """Test climate prediction model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        
        # Create target with some relationship to features
        y = pd.Series(
            X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5,
            name="target"
        )
        
        return X, y

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def predictor(self, temp_model_dir):
        """Create predictor instance."""
        return ClimatePredictor(model_dir=temp_model_dir, random_state=42)

    def test_init(self, temp_model_dir):
        """Test predictor initialization."""
        predictor = ClimatePredictor(model_dir=temp_model_dir, random_state=42)
        
        assert predictor.model_dir == Path(temp_model_dir)
        assert predictor.random_state == 42
        assert not predictor.is_fitted
        assert len(predictor.models) == 6  # Expected number of base models
        assert predictor.ensemble_weights is None

    def test_model_names(self, predictor):
        """Test that all expected models are present."""
        expected_models = [
            "linear", "ridge", "random_forest", 
            "gradient_boosting", "xgboost", "lightgbm"
        ]
        
        for model_name in expected_models:
            assert model_name in predictor.models

    def test_fit(self, predictor, sample_data):
        """Test model fitting."""
        X, y = sample_data
        
        results = predictor.fit(X, y, validation_split=0.2)
        
        # Check that model is fitted
        assert predictor.is_fitted
        
        # Check results structure
        assert "model_scores" in results
        assert "ensemble_scores" in results
        assert "ensemble_weights" in results
        
        # Check that all models were trained
        for model_name in predictor.models.keys():
            assert model_name in results["model_scores"]
        
        # Check ensemble weights
        assert predictor.ensemble_weights is not None
        assert len(predictor.ensemble_weights) > 0

    def test_predict_before_fit(self, predictor, sample_data):
        """Test that prediction fails before fitting."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.predict(X)

    def test_predict_after_fit(self, predictor, sample_data):
        """Test prediction after fitting."""
        X, y = sample_data
        
        # Fit model
        predictor.fit(X, y)
        
        # Make predictions
        predictions = predictor.predict(X)
        
        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()

    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        # Fit model
        predictor.fit(X, y)
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        
        # Check that importance is returned for tree models
        tree_models = ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]
        for model_name in tree_models:
            if model_name in importance:
                assert len(importance[model_name]) == X.shape[1]

    def test_cross_validate(self, predictor, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        
        cv_results = predictor.cross_validate(X, y, cv_folds=3)
        
        # Check results structure
        assert isinstance(cv_results, dict)
        assert len(cv_results) > 0
        
        # Check that each model has results
        for model_name, results in cv_results.items():
            assert "mean_mae" in results
            assert "std_mae" in results
            assert "scores" in results
            assert len(results["scores"]) == 3  # 3 folds

    def test_save_load_model(self, predictor, sample_data, temp_model_dir):
        """Test model saving and loading."""
        X, y = sample_data
        
        # Fit and save model
        predictor.fit(X, y)
        model_filename = "test_model.joblib"
        predictor.save_model(model_filename)
        
        # Check that file was created
        model_path = Path(temp_model_dir) / model_filename
        assert model_path.exists()
        
        # Create new predictor and load model
        new_predictor = ClimatePredictor(model_dir=temp_model_dir)
        new_predictor.load_model(model_filename)
        
        # Check that model is loaded
        assert new_predictor.is_fitted
        
        # Make predictions with both models
        pred1 = predictor.predict(X)
        pred2 = new_predictor.predict(X)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_save_before_fit(self, predictor):
        """Test that saving fails before fitting."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.save_model("test.joblib")

    def test_load_nonexistent_model(self, predictor):
        """Test loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            predictor.load_model("nonexistent.joblib")


class TestEvaluateModel:
    """Test model evaluation functions."""

    def test_evaluate_model(self):
        """Test model evaluation function."""
        # Create sample data
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluate_model(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ["mae", "rmse", "r2", "mape"]
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check that metrics are reasonable
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert 0 <= metrics["r2"] <= 1
        assert metrics["mape"] >= 0

    def test_evaluate_model_perfect_prediction(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        metrics = evaluate_model(y_true, y_pred)
        
        # Perfect prediction should have zero error and R² = 1
        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1
        assert metrics["mape"] == 0

    def test_evaluate_model_with_zeros(self):
        """Test evaluation when true values contain zeros."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0.1, 1.1, 1.9, 3.1, 3.9])
        
        # Should not raise error even with zeros in denominator
        metrics = evaluate_model(y_true, y_pred)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics