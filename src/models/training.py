"""Simplified Model Training Module.

This module handles training climate prediction models with basic
MLflow integration for learning purposes.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.ingestion import ESAClimateDataClient
from src.data.preprocessing import SimpleClimatePreprocessor
from src.models.climate_model import SimpleClimatePredictor, evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelTrainer:
    """Simple training pipeline for climate models."""

    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        experiment_name: str = "simple-climate-prediction",
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
        self.data_client = ESAClimateDataClient(str(self.data_dir / "raw"))
        self.preprocessor = SimpleClimatePreprocessor(str(self.data_dir))

        # Setup MLflow
        self.setup_mlflow(experiment_name)

    def setup_mlflow(self, experiment_name: str) -> None:
        """Setup MLflow experiment.

        Args:
            experiment_name: Name of the experiment
        """
        # Set tracking URI to local directory
        mlflow.set_tracking_uri("file:./mlruns")

        # Create or get experiment
        try:
            mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)

    def prepare_training_data(
        self,
        start_date: str,
        end_date: str,
        bbox: Tuple[float, float, float, float],
        target_col: str = "temperature",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare training data.

        Args:
            start_date: Start date for data
            end_date: End date for data
            bbox: Bounding box for geographic area
            target_col: Target column name

        Returns:
            Tuple of (processed_df, feature_columns)
        """
        logger.info("Preparing training data")

        # Fetch data from NOAA
        logger.info("Fetching climate data from NOAA")
        self.data_client.fetch_climate_data(start_date, end_date, bbox)

        # Preprocess data
        logger.info("Preprocessing data")
        df, features = self.preprocessor.prepare_training_data(
            start_date, end_date, target_col
        )

        return df, features

    def train_model(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: str = "temperature",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[SimpleClimatePredictor, Dict[str, Any]]:
        """Train simple climate prediction model.

        Args:
            df: Training dataframe
            features: List of feature columns
            target_col: Target column name
            test_size: Test set size
            random_state: Random state

        Returns:
            Tuple of (trained_model, results)
        """
        logger.info("Starting model training")

        # Start MLflow run
        with mlflow.start_run() as run:
            start_time = time.time()

            # Prepare data
            X = df[features]
            y = df[target_col]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Log dataset information
            mlflow.log_params(
                {
                    "dataset_size": len(df),
                    "n_features": len(features),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "target_column": target_col,
                    "test_split_ratio": test_size,
                    "random_state": random_state,
                }
            )

            # Initialize and train model
            model = SimpleClimatePredictor(
                model_dir=str(self.model_dir), random_state=random_state
            )

            logger.info("Training model")
            training_results = model.fit(X_train, y_train)

            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Calculate metrics
            train_metrics = evaluate_model(y_train.values, train_pred)
            test_metrics = evaluate_model(y_test.values, test_pred)

            training_time = time.time() - start_time

            # Log metrics to MLflow
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)

            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)

            # Log individual model performance
            for model_name, scores in training_results["model_scores"].items():
                for metric, value in scores.items():
                    mlflow.log_metric(f"{model_name}_{metric}", value)

            # Log feature importance
            feature_importance = model.get_feature_importance()
            if feature_importance:
                for model_name, importance in feature_importance.items():
                    for i, (feature, imp) in enumerate(
                        zip(features, importance)
                    ):  # noqa: E501
                        mlflow.log_metric(f"{model_name}_feature_{i}", imp)

            # Log timing
            mlflow.log_metric("training_time_seconds", training_time)

            # Save and log model
            model_filename = f"simple_climate_model_{run.info.run_id}.joblib"
            model.save_model(model_filename)

            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                "climate_model",
                registered_model_name="simple-climate-predictor",  # noqa: E501
            )

            # Prepare results
            results = {
                "run_id": run.info.run_id,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "training_results": training_results,
                "model_path": str(self.model_dir / model_filename),
                "training_time": training_time,
            }

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"Test R²: {test_metrics['r2']:.4f}")

            return model, results


def main() -> None:
    """Main function for command-line training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Simple Climate Model")
    parser.add_argument(
        "--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2023-03-31", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[-120, 35, -115, 40],
        help="Bounding box: min_lon min_lat max_lon max_lat",
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--model-dir", default="models", help="Model directory"
    )  # noqa: E501
    parser.add_argument(
        "--experiment-name",
        default="simple-climate-prediction",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = SimpleModelTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        experiment_name=args.experiment_name,
    )

    # Prepare data
    df, features = trainer.prepare_training_data(
        args.start_date, args.end_date, tuple(args.bbox)
    )

    # Train model
    model, results = trainer.train_model(df, features)
    logger.info("Model training completed!")


if __name__ == "__main__":
    main()
