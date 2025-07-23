#!/usr/bin/env python3
"""Training Script for Climate Prediction Model.

This script provides a command-line interface for training the climate
prediction model with various options and configurations.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.training import ClimateModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Climate Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--start-date", 
        default="2023-01-01", 
        help="Start date for training data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        default="2023-03-31", 
        help="End date for training data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bbox", 
        nargs=4, 
        type=float,
        default=[-120, 35, -115, 40],
        help="Bounding box: min_lon min_lat max_lon max_lat"
    )
    
    # Directory arguments
    parser.add_argument(
        "--data-dir", 
        default="data", 
        help="Data directory path"
    )
    parser.add_argument(
        "--model-dir", 
        default="models", 
        help="Model directory path"
    )
    
    # MLflow arguments
    parser.add_argument(
        "--experiment-name", 
        default="climate-temperature-prediction",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name", 
        help="MLflow run name"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow tracking server URI"
    )
    
    # Training arguments
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Test set size (fraction)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random state for reproducibility"
    )
    
    # Hyperparameter tuning
    parser.add_argument(
        "--hyperparameter-tuning", 
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--n-trials", 
        type=int, 
        default=10,
        help="Number of hyperparameter tuning trials"
    )
    
    # Data fetching
    parser.add_argument(
        "--fetch-new-data", 
        action="store_true",
        help="Fetch new data instead of using existing"
    )
    parser.add_argument(
        "--skip-data-preparation", 
        action="store_true",
        help="Skip data preparation and use existing processed data"
    )
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    
    try:
        # Initialize trainer
        logger.info("Initializing Climate Model Trainer")
        trainer = ClimateModelTrainer(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            experiment_name=args.experiment_name
        )
        
        # Prepare training data
        if not args.skip_data_preparation:
            logger.info("Preparing training data")
            df, features = trainer.prepare_training_data(
                args.start_date,
                args.end_date,
                tuple(args.bbox),
                fetch_new_data=args.fetch_new_data
            )
        else:
            # Load existing processed data
            logger.info("Loading existing processed data")
            import pandas as pd
            
            processed_file = f"{args.data_dir}/processed/training_data_{args.start_date}_{args.end_date}.csv"
            df = pd.read_csv(processed_file, parse_dates=["date"])
            
            # Get feature columns
            exclude_cols = [
                "LST_Day_C", "date", "longitude", "latitude", 
                "LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"
            ]
            features = [col for col in df.columns if col not in exclude_cols]
            
            logger.info(f"Loaded data: {len(df)} samples, {len(features)} features")
        
        if args.hyperparameter_tuning:
            # Perform hyperparameter tuning
            logger.info(f"Starting hyperparameter tuning with {args.n_trials} trials")
            tuning_results = trainer.hyperparameter_tuning(
                df, features, n_trials=args.n_trials
            )
            
            logger.info("Hyperparameter tuning completed!")
            logger.info(f"Best parameters: {tuning_results['best_params']}")
            logger.info(f"Best MAE: {tuning_results['best_value']:.4f}")
            
        else:
            # Train single model
            logger.info("Training model")
            model, results = trainer.train_model(
                df, 
                features, 
                test_size=args.test_size,
                random_state=args.random_state,
                run_name=args.run_name
            )
            
            logger.info("Model training completed!")
            logger.info(f"Run ID: {results['run_id']}")
            logger.info(f"Test MAE: {results['test_metrics']['mae']:.4f}")
            logger.info(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
            logger.info(f"Test R²: {results['test_metrics']['r2']:.4f}")
            logger.info(f"Model saved to: {results['model_path']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()