"""Airflow DAG for Climate Prediction ML Pipeline.

This DAG orchestrates the complete ML pipeline including:
- Data ingestion from NASA Earth Data
- Data preprocessing and feature engineering
- Model training with MLflow tracking
- Model evaluation and validation
- Model deployment
- Performance monitoring
"""

from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

# Default arguments for the DAG
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,
}

# DAG definition
dag = DAG(
    "climate_prediction_pipeline",
    default_args=default_args,
    description="End-to-end climate prediction ML pipeline",
    schedule_interval="@daily",  # Run daily
    max_active_runs=1,
    tags=["climate", "ml", "prediction", "nasa"],
)


def get_pipeline_config():
    """Get pipeline configuration from Airflow Variables."""
    return {
        "data_dir": Variable.get("climate_data_dir", "/opt/airflow/data"),
        "model_dir": Variable.get("climate_model_dir", "/opt/airflow/models"),
        "start_date": Variable.get("climate_start_date", "2023-01-01"),
        "end_date": Variable.get("climate_end_date", "2023-12-31"),
        "bbox": Variable.get("climate_bbox", "-120,35,-115,40").split(","),
        "mlflow_tracking_uri": Variable.get(
            "mlflow_tracking_uri", "http://mlflow:5000"
        ),  # noqa: E501
        "experiment_name": Variable.get(
            "mlflow_experiment_name", "climate-prediction-airflow"
        ),  # noqa: E501
    }


def ingest_nasa_data(**context):
    """Task to ingest NASA Earth data."""
    import sys

    sys.path.append("/opt/airflow/dags")

    from src.data.ingestion import NASAEarthDataClient

    config = get_pipeline_config()

    # Initialize client
    client = NASAEarthDataClient(data_dir=f"{config['data_dir']}/raw")

    # Parse bounding box
    bbox = tuple(map(float, config["bbox"]))

    # Fetch climate indicators
    data = client.fetch_climate_indicators(
        config["start_date"], config["end_date"], bbox
    )

    # Store metadata for downstream tasks
    context["task_instance"].xcom_push(
        key="ingestion_stats",
        value={
            "datasets": list(data.keys()),
            "total_records": sum(len(df) for df in data.values()),
            "date_range": f"{config['start_date']} to {config['end_date']}",
            "bbox": bbox,
        },
    )

    return "Data ingestion completed successfully"


def preprocess_data(**context):
    """Task to preprocess climate data."""
    import sys

    sys.path.append("/opt/airflow/dags")

    from src.data.preprocessing import ClimateDataPreprocessor

    config = get_pipeline_config()

    # Initialize preprocessor
    preprocessor = ClimateDataPreprocessor(data_dir=config["data_dir"])

    # Create training data
    df, features = preprocessor.create_training_data(
        config["start_date"], config["end_date"], target_col="LST_Day_C"
    )

    # Store metadata
    context["task_instance"].xcom_push(
        key="preprocessing_stats",
        value={
            "dataset_size": len(df),
            "feature_count": len(features),
            "features": features[:10],  # First 10 features
            "target_distribution": {
                "mean": float(df["LST_Day_C"].mean()),
                "std": float(df["LST_Day_C"].std()),
                "min": float(df["LST_Day_C"].min()),
                "max": float(df["LST_Day_C"].max()),
            },
        },
    )

    return "Data preprocessing completed successfully"


def train_model(**context):
    """Task to train the climate prediction model."""
    import sys

    sys.path.append("/opt/airflow/dags")

    import pandas as pd

    from src.models.training import ClimateModelTrainer

    config = get_pipeline_config()

    # Initialize trainer
    trainer = ClimateModelTrainer(
        data_dir=config["data_dir"],
        model_dir=config["model_dir"],
        experiment_name=config["experiment_name"],
    )

    # Load preprocessed data
    processed_file = f"{config['data_dir']}/processed/training_data_{config['start_date']}_{config['end_date']}.csv"  # noqa: E501
    df = pd.read_csv(processed_file, parse_dates=["date"])

    # Get all feature columns (exclude target and metadata)
    exclude_cols = [
        "LST_Day_C",
        "date",
        "longitude",
        "latitude",
        "LST_Day_1km",
        "LST_Night_1km",
        "QC_Day",
        "QC_Night",
    ]  # noqa: E501
    features = [col for col in df.columns if col not in exclude_cols]

    # Train model
    run_name = f"airflow_run_{context['ds']}"
    model, results = trainer.train_model(df, features, run_name=run_name)

    # Store training results
    context["task_instance"].xcom_push(
        key="training_results",
        value={
            "run_id": results["run_id"],
            "test_mae": results["test_metrics"]["mae"],
            "test_rmse": results["test_metrics"]["rmse"],
            "test_r2": results["test_metrics"]["r2"],
            "model_path": results["model_path"],
            "training_time": results["training_time"],
        },
    )

    return "Model training completed successfully"


def validate_model(**context):
    """Task to validate the trained model."""
    import sys

    sys.path.append("/opt/airflow/dags")

    import pandas as pd

    from src.models.climate_model import ClimatePredictor, evaluate_model

    config = get_pipeline_config()

    # Get training results
    training_results = context["task_instance"].xcom_pull(
        task_ids="train_model", key="training_results"
    )

    # Load model
    model = ClimatePredictor(model_dir=config["model_dir"])
    model_filename = Path(training_results["model_path"]).name
    model.load_model(model_filename)

    # Load test data (use a holdout set)
    processed_file = f"{config['data_dir']}/processed/training_data_{config['start_date']}_{config['end_date']}.csv"  # noqa: E501
    df = pd.read_csv(processed_file, parse_dates=["date"])

    # Use last 20% for validation
    val_size = int(len(df) * 0.2)
    val_df = df.tail(val_size)

    # Prepare features
    exclude_cols = [
        "LST_Day_C",
        "date",
        "longitude",
        "latitude",
        "LST_Day_1km",
        "LST_Night_1km",
        "QC_Day",
        "QC_Night",
    ]  # noqa: E501
    features = [col for col in df.columns if col not in exclude_cols]

    X_val = val_df[features]
    y_val = val_df["LST_Day_C"]

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate validation metrics
    val_metrics = evaluate_model(y_val.values, y_pred)

    # Validation thresholds
    mae_threshold = 3.0  # Maximum acceptable MAE
    r2_threshold = 0.5  # Minimum acceptable R²

    validation_passed = (
        val_metrics["mae"] <= mae_threshold and val_metrics["r2"] >= r2_threshold
    )

    validation_results = {
        "validation_passed": validation_passed,
        "val_metrics": val_metrics,
        "thresholds": {"mae_threshold": mae_threshold, "r2_threshold": r2_threshold},
    }

    # Store validation results
    context["task_instance"].xcom_push(
        key="validation_results", value=validation_results
    )

    if not validation_passed:
        mae = val_metrics['mae']
        r2 = val_metrics['r2']
        raise ValueError(
            f"Model validation failed: MAE={mae:.3f}, R²={r2:.3f}"
        )

    return "Model validation passed"


def deploy_model(**context):
    """Task to deploy the validated model."""
    import json
    import shutil

    config = get_pipeline_config()

    # Get training and validation results
    training_results = context["task_instance"].xcom_pull(
        task_ids="train_model", key="training_results"
    )

    validation_results = context["task_instance"].xcom_pull(
        task_ids="validate_model", key="validation_results"
    )

    # Copy model to production directory
    model_path = training_results["model_path"]
    production_model_path = (
        f"{config['model_dir']}/production/climate_model_latest.joblib"  # noqa: E501
    )

    # Create production directory
    Path(production_model_path).parent.mkdir(parents=True, exist_ok=True)

    # Copy model
    shutil.copy2(model_path, production_model_path)

    # Create model metadata
    metadata = {
        "deployed_at": context["ts"],
        "run_id": training_results["run_id"],
        "model_metrics": validation_results["val_metrics"],
        "training_date": config["start_date"],
        "data_end_date": config["end_date"],
        "model_version": f"v{context['ds_nodash']}",
    }

    metadata_path = f"{config['model_dir']}/production/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    context["task_instance"].xcom_push(
        key="deployment_results",
        value={
            "production_model_path": production_model_path,
            "metadata_path": metadata_path,
            "model_version": metadata["model_version"],
        },
    )

    return "Model deployed to production"


def monitor_model_performance(**context):
    """Task to set up monitoring for the deployed model."""
    import sys

    sys.path.append("/opt/airflow/dags")

    # Get deployment results
    deployment_results = context["task_instance"].xcom_pull(
        task_ids="deploy_model", key="deployment_results"
    )

    # Store monitoring configuration
    monitoring_config = {
        "model_version": deployment_results["model_version"],
        "monitoring_enabled": True,
        "alert_thresholds": {
            "prediction_latency_seconds": 1.0,
            "error_rate_percent": 5.0,
            "mae_threshold": 3.0,
            "data_drift_score": 0.7,
        },
        "dashboard_url": "http://grafana:3000/d/climate-model-dashboard",
    }

    context["task_instance"].xcom_push(key="monitoring_config", value=monitoring_config)

    return "Model monitoring configured"


def send_pipeline_notification(**context):
    """Task to send pipeline completion notification."""
    # Get all task results
    ingestion_stats = context["task_instance"].xcom_pull(
        task_ids="ingest_data", key="ingestion_stats"
    )

    training_results = context["task_instance"].xcom_pull(
        task_ids="train_model", key="training_results"
    )

    validation_results = context["task_instance"].xcom_pull(
        task_ids="validate_model", key="validation_results"
    )

    deployment_results = context["task_instance"].xcom_pull(
        task_ids="deploy_model", key="deployment_results"
    )

    # Create summary message
    message = f"""
    Climate Prediction Pipeline Completed Successfully!

    Execution Date: {context['ds']}
    Run ID: {context['run_id']}

    Data Ingestion:
    - Total Records: {ingestion_stats['total_records']:,}
    - Date Range: {ingestion_stats['date_range']}

    Model Training:
    - Test MAE: {training_results['test_mae']:.3f}°C
    - Test R²: {training_results['test_r2']:.3f}
    - Training Time: {training_results['training_time']:.1f}s

    Model Validation: {'✓ PASSED' if validation_results['validation_passed'] else '✗ FAILED'}  # noqa: E501

    Deployment:
    - Model Version: {deployment_results['model_version']}
    - Production Path: {deployment_results['production_model_path']}

    Dashboard: http://grafana:3000/d/climate-model-dashboard
    MLflow: http://mlflow:5000
    """

    print(message)

    # In production, send to Slack/email/etc.
    return "Pipeline notification sent"


# Define tasks
start_task = DummyOperator(
    task_id="start_pipeline",
    dag=dag,
)

# Data tasks
ingest_task = PythonOperator(
    task_id="ingest_data",
    python_callable=ingest_nasa_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)

# Model tasks
with TaskGroup("model_training", dag=dag) as model_group:
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    validate_task = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
    )

    train_task >> validate_task

# Deployment tasks
deploy_task = PythonOperator(
    task_id="deploy_model",
    python_callable=deploy_model,
    dag=dag,
)

monitor_task = PythonOperator(
    task_id="setup_monitoring",
    python_callable=monitor_model_performance,
    dag=dag,
)

# Notification task
notify_task = PythonOperator(
    task_id="send_notification",
    python_callable=send_pipeline_notification,
    dag=dag,
)

# End task
end_task = DummyOperator(
    task_id="end_pipeline",
    dag=dag,
)

# Define task dependencies
(
    start_task
    >> ingest_task
    >> preprocess_task
    >> model_group
    >> deploy_task
    >> monitor_task
    >> notify_task
    >> end_task
)  # noqa: E501
