#!/usr/bin/env python3
"""Deployment Script for Climate Prediction Model.

This script handles model deployment tasks including validation,
production deployment, and health checks.
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.climate_model import ClimatePredictor, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_model(model_path: str, validation_data_path: str) -> dict:
    """Validate a trained model.
    
    Args:
        model_path: Path to the model file
        validation_data_path: Path to validation data
        
    Returns:
        Validation results dictionary
    """
    import pandas as pd
    
    logger.info(f"Validating model: {model_path}")
    
    # Load model
    model = ClimatePredictor()
    model.load_model(Path(model_path).name)
    
    # Load validation data
    df = pd.read_csv(validation_data_path, parse_dates=["date"])
    
    # Prepare features
    exclude_cols = [
        "LST_Day_C", "date", "longitude", "latitude", 
        "LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"
    ]
    features = [col for col in df.columns if col not in exclude_cols]
    
    X = df[features]
    y = df["LST_Day_C"]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = evaluate_model(y.values, y_pred)
    
    # Validation thresholds
    thresholds = {
        "mae_threshold": 3.0,
        "rmse_threshold": 4.0,
        "r2_threshold": 0.5
    }
    
    # Check if validation passes
    validation_passed = (
        metrics["mae"] <= thresholds["mae_threshold"] and
        metrics["rmse"] <= thresholds["rmse_threshold"] and
        metrics["r2"] >= thresholds["r2_threshold"]
    )
    
    results = {
        "validation_passed": validation_passed,
        "metrics": metrics,
        "thresholds": thresholds,
        "model_path": model_path,
        "validation_data_path": validation_data_path,
        "validation_timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Validation results: {results}")
    
    return results


def deploy_to_production(
    model_path: str, 
    production_dir: str,
    model_metadata: dict = None
) -> dict:
    """Deploy model to production.
    
    Args:
        model_path: Path to the validated model
        production_dir: Production deployment directory
        model_metadata: Additional model metadata
        
    Returns:
        Deployment results dictionary
    """
    logger.info(f"Deploying model to production: {production_dir}")
    
    prod_path = Path(production_dir)
    prod_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model to production
    model_filename = Path(model_path).name
    production_model_path = prod_path / "climate_model_latest.joblib"
    
    shutil.copy2(model_path, production_model_path)
    logger.info(f"Model copied to: {production_model_path}")
    
    # Create model metadata
    metadata = {
        "deployed_at": datetime.now().isoformat(),
        "source_model_path": str(model_path),
        "production_model_path": str(production_model_path),
        "model_version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "deployment_status": "active"
    }
    
    if model_metadata:
        metadata.update(model_metadata)
    
    # Save metadata
    metadata_path = prod_path / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_path}")
    
    return {
        "production_model_path": str(production_model_path),
        "metadata_path": str(metadata_path),
        "model_version": metadata["model_version"],
        "deployment_timestamp": metadata["deployed_at"]
    }


def create_deployment_health_check(production_dir: str) -> dict:
    """Create a health check for the deployed model.
    
    Args:
        production_dir: Production deployment directory
        
    Returns:
        Health check results
    """
    logger.info("Running deployment health check")
    
    prod_path = Path(production_dir)
    
    # Check if model file exists
    model_path = prod_path / "climate_model_latest.joblib"
    model_exists = model_path.exists()
    
    # Check if metadata exists
    metadata_path = prod_path / "model_metadata.json"
    metadata_exists = metadata_path.exists()
    
    health_status = {
        "production_directory_exists": prod_path.exists(),
        "model_file_exists": model_exists,
        "metadata_file_exists": metadata_exists,
        "health_check_timestamp": datetime.now().isoformat()
    }
    
    if model_exists and metadata_exists:
        try:
            # Load and test model
            model = ClimatePredictor()
            model.load_model(model_path.name)
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            health_status.update({
                "model_loadable": True,
                "model_version": metadata.get("model_version", "unknown"),
                "deployment_date": metadata.get("deployed_at", "unknown")
            })
            
        except Exception as e:
            health_status.update({
                "model_loadable": False,
                "error": str(e)
            })
    
    overall_health = all([
        health_status["production_directory_exists"],
        health_status["model_file_exists"], 
        health_status["metadata_file_exists"],
        health_status.get("model_loadable", False)
    ])
    
    health_status["overall_health"] = "healthy" if overall_health else "unhealthy"  # noqa: E501
    
    logger.info(f"Health check results: {health_status}")
    
    return health_status


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy Climate Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "action",
        choices=["validate", "deploy", "health-check"],
        help="Deployment action to perform"
    )
    
    parser.add_argument(
        "--model-path",
        help="Path to the model file"
    )
    
    parser.add_argument(
        "--validation-data",
        help="Path to validation data file"
    )
    
    parser.add_argument(
        "--production-dir",
        default="models/production",
        help="Production deployment directory"
    )
    
    parser.add_argument(
        "--output-file",
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.action == "validate":
            if not args.model_path or not args.validation_data:
                raise ValueError("Model path and validation data are required for validation")  # noqa: E501
            
            results = validate_model(args.model_path, args.validation_data)
            
            if not results["validation_passed"]:
                logger.error("Model validation failed!")
                sys.exit(1)
            else:
                logger.info("Model validation passed!")
        
        elif args.action == "deploy":
            if not args.model_path:
                raise ValueError("Model path is required for deployment")
            
            results = deploy_to_production(args.model_path, args.production_dir)  # noqa: E501
            logger.info("Model deployed successfully!")
        
        elif args.action == "health-check":
            results = create_deployment_health_check(args.production_dir)
            
            if results["overall_health"] != "healthy":
                logger.warning("Deployment health check failed!")
                sys.exit(1)
            else:
                logger.info("Deployment health check passed!")
        
        # Save results to file if specified
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Deployment action failed: {e}")
        raise


if __name__ == "__main__":
    main()