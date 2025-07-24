"""Climate Prediction Models with GPU Support.

This module contains ML models for climate temperature prediction
with GPU acceleration when available.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .gpu_utils import gpu_manager

logger = logging.getLogger(__name__)


class GPUAcceleratedClimatePredictor:
    """Climate predictor with GPU acceleration support."""

    def __init__(self, model_dir: str = "models", random_state: int = 42):
        """Initialize climate predictor with GPU support.

        Args:
            model_dir: Directory to save/load models
            random_state: Random state for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.gpu_info = gpu_manager.get_gpu_info()

        # Initialize models with GPU support when available
        self.models = {}
        self._initialize_models()
        self.is_fitted = False

    def _initialize_models(self):
        """Initialize models with appropriate GPU/CPU configuration."""
        # Always include basic sklearn models
        self.models["linear"] = LinearRegression()
        self.models["random_forest"] = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=self.random_state, n_jobs=-1
        )

        # Add GPU-accelerated models if available
        if gpu_manager.is_gpu_available():
            logger.info("Initializing GPU-accelerated models")

            # XGBoost with GPU support
            try:
                import xgboost as xgb

                xgb_params = gpu_manager.get_xgboost_params()
                self.models["xgboost_gpu"] = xgb.XGBRegressor(**xgb_params)
                logger.info("XGBoost GPU model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize XGBoost GPU: {e}")

            # LightGBM with GPU support
            try:
                import lightgbm as lgb

                lgb_params = gpu_manager.get_lightgbm_params()
                self.models["lightgbm_gpu"] = lgb.LGBMRegressor(**lgb_params)
                logger.info("LightGBM GPU model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LightGBM GPU: {e}")
        else:
            logger.info("GPU not available, using CPU-only models")
            # Add CPU versions of advanced models
            try:
                import xgboost as xgb

                self.models["xgboost"] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            except ImportError:
                logger.debug("XGBoost not available")

            try:
                import lightgbm as lgb

                self.models["lightgbm"] = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                logger.debug("LightGBM not available")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the ensemble model with GPU acceleration when available.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Training results and metrics
        """
        logger.info("Training climate prediction model with GPU support")
        logger.info(f"GPU Status: {self.gpu_info['gpu_available']}")

        if self.gpu_info["gpu_available"]:
            for device in self.gpu_info["devices"]:
                logger.info(f"Using GPU: {device['name']}")

        # Train individual models
        model_scores = {}
        training_times = {}

        for name, model in self.models.items():
            logger.info(f"Training {name} model")

            import time

            start_time = time.time()

            try:
                # Fit model
                model.fit(X, y)
                training_time = time.time() - start_time
                training_times[name] = training_time

                # Make predictions
                predictions = model.predict(X)

                # Calculate metrics
                mae = mean_absolute_error(y, predictions)
                rmse = np.sqrt(mean_squared_error(y, predictions))
                r2 = r2_score(y, predictions)

                model_scores[name] = {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "training_time": training_time,
                    "gpu_accelerated": "gpu" in name.lower(),
                }

                logger.info(
                    f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                    f"R2: {r2:.4f}, Time: {training_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                # Remove failed model
                if name in self.models:
                    del self.models[name]

        self.is_fitted = True

        # Return comprehensive results
        results = {
            "model_scores": model_scores,
            "training_times": training_times,
            "n_features": X.shape[1],
            "training_samples": len(X),
            "gpu_info": self.gpu_info,
            "models_trained": list(self.models.keys()),
        }

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions using available models.

        Args:
            X: Feature matrix for prediction

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if len(self.models) == 0:
            raise ValueError("No models available for prediction")

        # Get predictions from all models
        predictions = []
        weights = []

        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)

                # Weight GPU models slightly higher if available
                weight = 1.2 if "gpu" in name.lower() else 1.0
                weights.append(weight)

            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        if not predictions:
            raise ValueError("All models failed to make predictions")

        # Weighted ensemble average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights

        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from models that support it.

        Returns:
            Dictionary of feature importances by model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance_dict = {}

        for name, model in self.models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    importance_dict[name] = model.feature_importances_
                elif hasattr(model, "coef_"):
                    # For linear models, use absolute coefficients
                    importance_dict[name] = np.abs(model.coef_)
            except Exception as e:
                logger.warning(f"Could not get feature importance for {name}: {e}")

        return importance_dict

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models.

        Returns:
            Dictionary with model information
        """
        info = {
            "is_fitted": self.is_fitted,
            "n_models": len(self.models),
            "model_names": list(self.models.keys()),
            "gpu_info": self.gpu_info,
            "gpu_accelerated_models": [
                name for name in self.models.keys() if "gpu" in name.lower()
            ],
        }
        return info

    def save_model(self, filename: str) -> None:
        """Save the trained ensemble model.

        Args:
            filename: Filename to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_path = self.model_dir / filename
        model_data = {
            "models": self.models,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "gpu_info": self.gpu_info,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str) -> None:
        """Load a trained ensemble model.

        Args:
            filename: Filename to load the model from
        """
        model_path = self.model_dir / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)
        self.models = model_data["models"]
        self.is_fitted = model_data["is_fitted"]
        self.random_state = model_data.get("random_state", 42)
        self.gpu_info = model_data.get("gpu_info", {})

        logger.info(f"Model loaded from {model_path}")


# Maintain backward compatibility
SimpleClimatePredictor = GPUAcceleratedClimatePredictor


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of evaluation metrics
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def main() -> None:
    """Main function for testing the model."""
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 6

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        X.iloc[:, 0] * 2
        + X.iloc[:, 1] * 1.5
        + np.random.randn(n_samples) * 0.5,  # noqa: E501
        name="target",
    )

    # Initialize and train model
    predictor = GPUAcceleratedClimatePredictor()
    predictor.fit(X, y)

    # Make predictions
    predictions = predictor.predict(X)

    # Evaluate
    metrics = evaluate_model(y.values, predictions)

    logger.info("Model training and evaluation complete!")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
