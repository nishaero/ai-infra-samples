"""Climate Prediction Models.

This module contains ML models for climate temperature prediction,
including ensemble methods and time series forecasting.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class ClimatePredictor:
    """Ensemble model for climate temperature prediction."""

    def __init__(
        self,
        model_dir: str = "models",
        random_state: int = 42
    ):
        """Initialize climate predictor.
        
        Args:
            model_dir: Directory to save/load models
            random_state: Random state for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Initialize base models
        self.models = {
            "linear": LinearRegression(),
            "ridge": Ridge(random_state=random_state),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=random_state
            ),
            "xgboost": xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1
            ),
            "lightgbm": lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        # Ensemble weights (will be learned during training)
        self.ensemble_weights = None
        self.is_fitted = False
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the ensemble model.
        
        Args:
            X: Feature matrix
            y: Target values
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results and metrics
        """
        logger.info("Training climate prediction ensemble model")
        
        # Split data chronologically for time series
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        
        # Train individual models
        model_predictions = {}
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Store predictions for ensemble
                model_predictions[name] = val_pred
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, train_pred)
                val_mae = mean_absolute_error(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_r2 = r2_score(y_val, val_pred)
                
                model_scores[name] = {
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2
                }
                
                logger.info(f"{name} - Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                # Remove failed model
                del self.models[name]
        
        # Learn ensemble weights based on validation performance
        if model_predictions:
            self._learn_ensemble_weights(model_predictions, y_val)
        
        # Calculate ensemble predictions
        ensemble_pred = self._ensemble_predict_from_individual(model_predictions)
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        
        logger.info(f"Ensemble - Val MAE: {ensemble_mae:.4f}, Val RMSE: {ensemble_rmse:.4f}, Val R2: {ensemble_r2:.4f}")
        
        self.is_fitted = True
        
        # Prepare results
        results = {
            "model_scores": model_scores,
            "ensemble_scores": {
                "val_mae": ensemble_mae,
                "val_rmse": ensemble_rmse,
                "val_r2": ensemble_r2
            },
            "ensemble_weights": self.ensemble_weights,
            "n_features": X.shape[1],
            "training_samples": len(X_train),
            "validation_samples": len(X_val)
        }
        
        return results
    
    def _learn_ensemble_weights(
        self,
        model_predictions: Dict[str, np.ndarray],
        y_true: pd.Series
    ) -> None:
        """Learn optimal ensemble weights.
        
        Args:
            model_predictions: Dictionary of model predictions
            y_true: True target values
        """
        # Simple strategy: weight by inverse MAE
        weights = {}
        total_inverse_mae = 0
        
        for name, pred in model_predictions.items():
            mae = mean_absolute_error(y_true, pred)
            inverse_mae = 1.0 / (mae + 1e-8)  # Add small constant to avoid division by zero
            weights[name] = inverse_mae
            total_inverse_mae += inverse_mae
        
        # Normalize weights
        self.ensemble_weights = {
            name: weight / total_inverse_mae 
            for name, weight in weights.items()
        }
        
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def _ensemble_predict_from_individual(
        self,
        model_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Create ensemble predictions from individual model predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            
        Returns:
            Ensemble predictions
        """
        if not self.ensemble_weights:
            # Equal weights if no weights learned
            weights = {name: 1.0 / len(model_predictions) for name in model_predictions.keys()}
        else:
            weights = self.ensemble_weights
        
        ensemble_pred = np.zeros(len(next(iter(model_predictions.values()))))
        
        for name, pred in model_predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all models
        model_predictions = {}
        for name, model in self.models.items():
            try:
                model_predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning(f"Error in prediction with {name}: {e}")
        
        # Create ensemble prediction
        ensemble_pred = self._ensemble_predict_from_individual(model_predictions)
        
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models.
        
        Returns:
            Dictionary of feature importances
        """
        importance_dict = {}
        
        tree_models = ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]
        
        for name in tree_models:
            if name in self.models and hasattr(self.models[name], "feature_importances_"):
                importance_dict[name] = self.models[name].feature_importances_
        
        return importance_dict
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation on individual models.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation scores
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}")
            
            try:
                # Perform cross-validation
                scores = cross_val_score(
                    model, X, y, 
                    cv=tscv, 
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1
                )
                
                cv_results[name] = {
                    "mean_mae": -scores.mean(),
                    "std_mae": scores.std(),
                    "scores": -scores
                }
                
                logger.info(f"{name} CV MAE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error in cross-validation for {name}: {e}")
        
        return cv_results
    
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
            "ensemble_weights": self.ensemble_weights,
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
        self.ensemble_weights = model_data["ensemble_weights"]
        self.is_fitted = model_data["is_fitted"]
        self.random_state = model_data.get("random_state", 42)
        
        logger.info(f"Model loaded from {model_path}")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
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
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def main() -> None:
    """Main function for testing the model."""
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(
        X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5,
        name="target"
    )
    
    # Initialize and train model
    predictor = ClimatePredictor()
    results = predictor.fit(X, y)
    
    # Make predictions
    predictions = predictor.predict(X)
    
    # Evaluate
    metrics = evaluate_model(y.values, predictions)
    
    logger.info("Model training and evaluation complete!")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()