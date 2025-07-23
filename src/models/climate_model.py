"""Simplified Climate Prediction Models.

This module contains basic ML models for climate temperature prediction
with simple ensemble approach for learning purposes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class SimpleClimatePredictor:
    """Simple ensemble model for climate temperature prediction."""

    def __init__(self, model_dir: str = "models", random_state: int = 42):
        """Initialize climate predictor.
        
        Args:
            model_dir: Directory to save/load models
            random_state: Random state for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Initialize simple models
        self.models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        }
        
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the simple ensemble model.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Training results and metrics
        """
        logger.info("Training simple climate prediction model")
        
        # Train individual models
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            
            try:
                # Fit model
                model.fit(X, y)
                
                # Make predictions
                predictions = model.predict(X)
                
                # Calculate metrics
                mae = mean_absolute_error(y, predictions)
                rmse = np.sqrt(mean_squared_error(y, predictions))
                r2 = r2_score(y, predictions)
                
                model_scores[name] = {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2
                }
                
                logger.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                # Remove failed model
                del self.models[name]
        
        self.is_fitted = True
        
        # Return results
        results = {
            "model_scores": model_scores,
            "n_features": X.shape[1],
            "training_samples": len(X)
        }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using simple ensemble (average).
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all models
        all_predictions = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                all_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error in prediction with {name}: {e}")
        
        # Simple average ensemble
        if all_predictions:
            ensemble_pred = np.mean(all_predictions, axis=0)
        else:
            raise RuntimeError("No models available for prediction")
        
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from random forest model.
        
        Returns:
            Dictionary of feature importances
        """
        importance_dict = {}
        
        if "random_forest" in self.models:
            importance_dict["random_forest"] = self.models["random_forest"].feature_importances_
        
        return importance_dict
    
    def save_model(self, filename: str) -> None:
        """Save the trained model.
        
        Args:
            filename: Filename to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_path = self.model_dir / filename
        
        model_data = {
            "models": self.models,
            "is_fitted": self.is_fitted,
            "random_state": self.random_state
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str) -> None:
        """Load a trained model.
        
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
        
        logger.info(f"Model loaded from {model_path}")


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
        "r2": r2_score(y_true, y_pred)
    }


def main() -> None:
    """Main function for testing the model."""
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 6
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(
        X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5,
        name="target"
    )
    
    # Initialize and train model
    predictor = SimpleClimatePredictor()
    results = predictor.fit(X, y)
    
    # Make predictions
    predictions = predictor.predict(X)
    
    # Evaluate
    metrics = evaluate_model(y.values, predictions)
    
    logger.info("Model training and evaluation complete!")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()