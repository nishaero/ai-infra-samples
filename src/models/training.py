"""Model Training Module with MLflow Integration.

This module handles training climate prediction models with comprehensive
experiment tracking using MLflow.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.ingestion import NASAEarthDataClient
from src.data.preprocessing import ClimateDataPreprocessor
from src.models.climate_model import ClimatePredictor, evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowExperimentTracker:
    """MLflow experiment tracking for climate prediction models."""

    def __init__(
        self,
        experiment_name: str = "climate-temperature-prediction",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """Initialize MLflow experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        else:
            # Default to local file store
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"Created new experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        self.experiment_id = experiment_id
        mlflow.set_experiment(experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            Active MLflow run
        """
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number for time series metrics
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        model_name: str,
        registered_model_name: Optional[str] = None
    ) -> None:
        """Log model to MLflow.
        
        Args:
            model: Trained model
            model_name: Name for the model artifact
            registered_model_name: Name for model registry
        """
        # Log model based on type
        if hasattr(model, "models"):
            # Custom ensemble model
            mlflow.sklearn.log_model(
                model,
                model_name,
                registered_model_name=registered_model_name
            )
        else:
            # Single model
            mlflow.sklearn.log_model(
                model,
                model_name,
                registered_model_name=registered_model_name
            )

    def log_artifacts(self, artifacts_dir: str) -> None:
        """Log artifacts directory to MLflow.
        
        Args:
            artifacts_dir: Path to artifacts directory
        """
        mlflow.log_artifacts(artifacts_dir)


class ClimateModelTrainer:
    """Comprehensive training pipeline for climate models."""

    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        experiment_name: str = "climate-temperature-prediction"
    ):
        """Initialize model trainer.
        
        Args:
            data_dir: Data directory path
            model_dir: Model directory path
            experiment_name: MLflow experiment name
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_client = NASAEarthDataClient(str(self.data_dir / "raw"))
        self.preprocessor = ClimateDataPreprocessor(str(self.data_dir))
        self.tracker = MLflowExperimentTracker(experiment_name)

    def prepare_training_data(
        self,
        start_date: str,
        end_date: str,
        bbox: Tuple[float, float, float, float],
        target_col: str = "LST_Day_C",
        fetch_new_data: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare training data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            bbox: Bounding box for geographic area
            target_col: Target column name
            fetch_new_data: Whether to fetch new data or use existing
            
        Returns:
            Tuple of (processed_df, feature_columns)
        """
        logger.info("Preparing training data")
        
        if fetch_new_data:
            # Fetch raw data
            logger.info("Fetching NASA Earth data")
            self.data_client.fetch_climate_indicators(start_date, end_date, bbox)
        
        # Preprocess data
        logger.info("Preprocessing data")
        df, features = self.preprocessor.create_training_data(
            start_date, end_date, target_col
        )
        
        return df, features

    def train_model(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: str = "LST_Day_C",
        test_size: float = 0.2,
        random_state: int = 42,
        run_name: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[ClimatePredictor, Dict[str, Any]]:
        """Train climate prediction model with MLflow tracking.
        
        Args:
            df: Training dataframe
            features: List of feature columns
            target_col: Target column name
            test_size: Test set size
            random_state: Random state
            run_name: MLflow run name
            model_params: Model hyperparameters
            
        Returns:
            Tuple of (trained_model, results)
        """
        logger.info("Starting model training with MLflow tracking")
        
        # Start MLflow run
        with self.tracker.start_run(run_name=run_name) as run:
            start_time = time.time()
            
            # Prepare data
            X = df[features]
            y = df[target_col]
            
            # Split data chronologically for time series
            split_idx = int(len(df) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Log dataset information
            self.tracker.log_params({
                "dataset_size": len(df),
                "n_features": len(features),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "target_column": target_col,
                "test_split_ratio": test_size,
                "random_state": random_state,
                "date_range": f"{df['date'].min()} to {df['date'].max()}",
                "geographic_bounds": f"lon: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}, "
                                   f"lat: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}"
            })
            
            # Initialize model
            model = ClimatePredictor(
                model_dir=str(self.model_dir),
                random_state=random_state
            )
            
            # Apply custom model parameters if provided
            if model_params:
                for model_name, params in model_params.items():
                    if model_name in model.models:
                        model.models[model_name].set_params(**params)
                
                self.tracker.log_params({f"model_{k}_{p}": v for k, v in model_params.items() for p, v in v.items()})
            
            # Train model
            logger.info("Training ensemble model")
            training_results = model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = evaluate_model(y_train.values, train_pred)
            test_metrics = evaluate_model(y_test.values, test_pred)
            
            training_time = time.time() - start_time
            
            # Log training metrics
            for metric, value in train_metrics.items():
                self.tracker.log_metrics({f"train_{metric}": value})
            
            for metric, value in test_metrics.items():
                self.tracker.log_metrics({f"test_{metric}": value})
            
            # Log individual model performance
            for model_name, scores in training_results["model_scores"].items():
                for metric, value in scores.items():
                    self.tracker.log_metrics({f"{model_name}_{metric}": value})
            
            # Log ensemble performance
            for metric, value in training_results["ensemble_scores"].items():
                self.tracker.log_metrics({f"ensemble_{metric}": value})
            
            # Log ensemble weights
            if training_results["ensemble_weights"]:
                for model_name, weight in training_results["ensemble_weights"].items():
                    self.tracker.log_metrics({f"ensemble_weight_{model_name}": weight})
            
            # Log feature importance
            feature_importance = model.get_feature_importance()
            if feature_importance:
                # Log top 10 important features for each model
                for model_name, importance in feature_importance.items():
                    top_features = sorted(
                        zip(features, importance), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                    
                    for i, (feature, imp) in enumerate(top_features):
                        self.tracker.log_metrics({f"{model_name}_feature_importance_{i+1}": imp})
                        self.tracker.log_params({f"{model_name}_top_feature_{i+1}": feature})
            
            # Log timing
            self.tracker.log_metrics({"training_time_seconds": training_time})
            
            # Save and log model
            model_filename = f"climate_model_{run.info.run_id}.joblib"
            model.save_model(model_filename)
            
            # Log model to MLflow
            self.tracker.log_model(
                model,
                "climate_ensemble_model",
                registered_model_name="climate-temperature-predictor"
            )
            
            # Create and log artifacts
            artifacts_dir = self.model_dir / f"artifacts_{run.info.run_id}"
            artifacts_dir.mkdir(exist_ok=True)
            
            # Save feature list
            features_file = artifacts_dir / "features.txt"
            with open(features_file, "w") as f:
                f.write("\n".join(features))
            
            # Save predictions for analysis
            predictions_df = pd.DataFrame({
                "actual": y_test.values,
                "predicted": test_pred,
                "date": df.iloc[split_idx:]["date"].values,
                "longitude": df.iloc[split_idx:]["longitude"].values,
                "latitude": df.iloc[split_idx:]["latitude"].values
            })
            predictions_file = artifacts_dir / "test_predictions.csv"
            predictions_df.to_csv(predictions_file, index=False)
            
            # Log artifacts
            self.tracker.log_artifacts(str(artifacts_dir))
            
            # Prepare results
            results = {
                "run_id": run.info.run_id,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "training_results": training_results,
                "model_path": str(self.model_dir / model_filename),
                "training_time": training_time,
                "predictions": predictions_df
            }
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"Test R²: {test_metrics['r2']:.4f}")
            
            return model, results

    def hyperparameter_tuning(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: str = "LST_Day_C",
        n_trials: int = 10
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning with Optuna and MLflow.
        
        Args:
            df: Training dataframe
            features: List of feature columns
            target_col: Target column name
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters and results
        """
        import optuna
        
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        def objective(trial):
            # Sample hyperparameters
            model_params = {
                "random_forest": {
                    "n_estimators": trial.suggest_int("rf_n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("rf_max_depth", 5, 15),
                    "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
                },
                "xgboost": {
                    "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
                },
                "lightgbm": {
                    "n_estimators": trial.suggest_int("lgb_n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("lgb_max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3),
                }
            }
            
            # Train model with these parameters
            run_name = f"hp_tuning_trial_{trial.number}"
            model, results = self.train_model(
                df, features, target_col,
                run_name=run_name,
                model_params=model_params
            )
            
            # Return metric to optimize (minimize MAE)
            return results["test_metrics"]["mae"]
        
        # Create study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter tuning completed")
        logger.info(f"Best MAE: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study
        }


def main() -> None:
    """Main function for command-line training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Climate Prediction Model")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-03-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bbox", nargs=4, type=float,
                       default=[-120, 35, -115, 40],
                       help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--experiment-name", default="climate-temperature-prediction", 
                       help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name")
    parser.add_argument("--hyperparameter-tuning", action="store_true",
                       help="Perform hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=10,
                       help="Number of hyperparameter tuning trials")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ClimateModelTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        experiment_name=args.experiment_name
    )
    
    # Prepare data
    df, features = trainer.prepare_training_data(
        args.start_date,
        args.end_date,
        tuple(args.bbox)
    )
    
    if args.hyperparameter_tuning:
        # Perform hyperparameter tuning
        tuning_results = trainer.hyperparameter_tuning(
            df, features, n_trials=args.n_trials
        )
        logger.info("Hyperparameter tuning completed!")
    else:
        # Train single model
        model, results = trainer.train_model(
            df, features, run_name=args.run_name
        )
        logger.info("Model training completed!")


if __name__ == "__main__":
    main()