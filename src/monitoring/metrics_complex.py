"""Simplified Model Monitoring and Metrics Module.

This module provides basic monitoring capabilities for the climate model
including performance tracking and simple metrics collection.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class SimpleModelMonitor:
    """Simple model monitoring system."""

    def __init__(self):
        """Initialize simple model monitor."""
        # Basic metrics storage
        self.predictions_log = []
        self.performance_metrics = []
        
        # Basic Prometheus metrics
        self.prediction_counter = Counter(
            'model_predictions_total', 
            'Total number of predictions made'
        )
        self.prediction_latency = Histogram(
            'model_prediction_duration_seconds', 
            'Time spent on predictions'
        )
        self.model_accuracy = Gauge(
            'model_accuracy_score', 
            'Current model accuracy'
        )
        
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions made',
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Time spent making predictions',
            registry=self.registry
        )
        
        self.prediction_error_gauge = Gauge(
            'model_prediction_error',
            'Current prediction error metric',
            registry=self.registry
        )
        
        self.data_drift_gauge = Gauge(
            'model_data_drift_score',
            'Data drift detection score',
            registry=self.registry
        )
        
        self.model_accuracy_gauge = Gauge(
            'model_accuracy',
            'Current model accuracy',
            registry=self.registry
        )
        
        logger.info("Model monitor initialized")

    def log_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        latency: float,
        actual_value: Optional[float] = None
    ) -> None:
        """Log a prediction for monitoring.
        
        Args:
            input_data: Input features used for prediction
            prediction: Model prediction output
            latency: Prediction latency in seconds
            actual_value: Actual target value (if available)
        """
        prediction_record = {
            "timestamp": datetime.now(),
            "input_data": input_data,
            "prediction": prediction,
            "latency": latency,
            "actual_value": actual_value
        }
        
        self.predictions_log.append(prediction_record)
        
        # Update Prometheus metrics
        self.prediction_counter.inc()
        self.prediction_latency.observe(latency)
        
        # Check for alerts
        self._check_latency_alert(latency)
        
        # Clean old records
        self._cleanup_old_records()
        
        logger.debug(f"Logged prediction: {prediction}")

    def log_batch_prediction(
        self,
        batch_size: int,
        total_latency: float
    ) -> None:
        """Log batch prediction metrics.
        
        Args:
            batch_size: Number of predictions in batch
            total_latency: Total processing time for batch
        """
        avg_latency = total_latency / batch_size
        
        self.prediction_counter.inc(batch_size)
        self.prediction_latency.observe(avg_latency)
        
        logger.debug(f"Logged batch prediction: {batch_size} predictions, {avg_latency:.3f}s avg latency")

    def calculate_model_performance(
        self,
        window_hours: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate model performance metrics over a time window.
        
        Args:
            window_hours: Time window in hours (default: monitoring window)
            
        Returns:
            Dictionary of performance metrics
        """
        window = timedelta(hours=window_hours) if window_hours else self.monitoring_window
        cutoff_time = datetime.now() - window
        
        # Filter recent predictions with actual values
        recent_predictions = [
            p for p in self.predictions_log
            if p["timestamp"] >= cutoff_time and p["actual_value"] is not None
        ]
        
        if not recent_predictions:
            logger.warning("No recent predictions with actual values for performance calculation")
            return {}
        
        # Extract predictions and actual values
        predicted_values = [p["prediction"]["predicted_temperature"] for p in recent_predictions]
        actual_values = [p["actual_value"] for p in recent_predictions]
        
        # Calculate metrics
        mae = np.mean(np.abs(np.array(predicted_values) - np.array(actual_values)))
        rmse = np.sqrt(np.mean((np.array(predicted_values) - np.array(actual_values)) ** 2))
        mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
        
        # Calculate correlation
        correlation = np.corrcoef(predicted_values, actual_values)[0, 1] if len(predicted_values) > 1 else 0
        
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "correlation": correlation,
            "sample_count": len(recent_predictions),
            "window_hours": window.total_seconds() / 3600
        }
        
        # Update Prometheus metrics
        self.prediction_error_gauge.set(mae)
        self.model_accuracy_gauge.set(correlation)
        
        # Store metrics
        self.performance_metrics.append({
            "timestamp": datetime.now(),
            "metrics": metrics
        })
        
        # Check for performance alerts
        self._check_performance_alerts(metrics)
        
        logger.info(f"Calculated performance metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, Correlation={correlation:.3f}")
        
        return metrics

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, float]:
        """Detect data drift between reference and current data.
        
        Args:
            reference_data: Reference/training data
            current_data: Current inference data
            features: List of features to check for drift
            
        Returns:
            Dictionary of drift scores per feature
        """
        drift_scores = {}
        
        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            # Calculate statistical distance (simplified KS test)
            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Simple drift detection using mean and std differences
            ref_mean, ref_std = ref_values.mean(), ref_values.std()
            curr_mean, curr_std = curr_values.mean(), curr_values.std()
            
            # Normalized difference
            mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            std_diff = abs(curr_std - ref_std) / (ref_std + 1e-8)
            
            drift_score = (mean_diff + std_diff) / 2
            drift_scores[feature] = drift_score
        
        # Overall drift score
        overall_drift = np.mean(list(drift_scores.values())) if drift_scores else 0
        drift_scores["overall"] = overall_drift
        
        # Update Prometheus metric
        self.data_drift_gauge.set(overall_drift)
        
        # Check for drift alerts
        self._check_drift_alerts(overall_drift)
        
        logger.info(f"Data drift detection completed: overall score={overall_drift:.3f}")
        
        return drift_scores

    def generate_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        metadata: Optional[Dict] = None
    ) -> None:
        """Generate an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, critical)
            metadata: Additional alert metadata
        """
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "metadata": metadata or {}
        }
        
        self.alerts_log.append(alert)
        
        # Log alert
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "critical": logging.ERROR
        }.get(severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{alert_type}]: {message}")
        
        # In production, this would send alerts to external systems
        # (e.g., PagerDuty, Slack, email)
        self._send_alert_notification(alert)

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard.
        
        Returns:
            Dictionary containing dashboard data
        """
        # Recent performance metrics
        recent_performance = self.performance_metrics[-1] if self.performance_metrics else {}
        
        # Recent alerts
        recent_alerts = self.alerts_log[-10:]  # Last 10 alerts
        
        # Prediction statistics
        recent_predictions = [
            p for p in self.predictions_log
            if p["timestamp"] >= datetime.now() - self.monitoring_window
        ]
        
        prediction_stats = {
            "total_predictions": len(recent_predictions),
            "avg_latency": np.mean([p["latency"] for p in recent_predictions]) if recent_predictions else 0,
            "error_rate": len([p for p in recent_predictions if p.get("error")]) / max(len(recent_predictions), 1) * 100
        }
        
        return {
            "performance_metrics": recent_performance,
            "prediction_stats": prediction_stats,
            "recent_alerts": recent_alerts,
            "monitoring_window_hours": self.monitoring_window.total_seconds() / 3600,
            "dashboard_timestamp": datetime.now()
        }

    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics.
        
        Returns:
            Prometheus metrics in text format
        """
        return generate_latest(self.registry).decode('utf-8')

    def _check_latency_alert(self, latency: float) -> None:
        """Check for latency alerts."""
        threshold = self.alert_thresholds.get("prediction_latency_seconds", 1.0)
        
        if latency > threshold:
            self.generate_alert(
                "high_latency",
                f"Prediction latency {latency:.3f}s exceeds threshold {threshold}s",
                "warning",
                {"latency": latency, "threshold": threshold}
            )

    def _check_performance_alerts(self, metrics: Dict[str, float]) -> None:
        """Check for performance-based alerts."""
        mae_threshold = self.alert_thresholds.get("mae_threshold", 3.0)
        
        if metrics.get("mae", 0) > mae_threshold:
            self.generate_alert(
                "high_prediction_error",
                f"Model MAE {metrics['mae']:.3f} exceeds threshold {mae_threshold}",
                "critical",
                {"mae": metrics["mae"], "threshold": mae_threshold}
            )

    def _check_drift_alerts(self, drift_score: float) -> None:
        """Check for data drift alerts."""
        threshold = self.alert_thresholds.get("data_drift_score", 0.7)
        
        if drift_score > threshold:
            self.generate_alert(
                "data_drift_detected",
                f"Data drift score {drift_score:.3f} exceeds threshold {threshold}",
                "warning",
                {"drift_score": drift_score, "threshold": threshold}
            )

    def _send_alert_notification(self, alert: Dict) -> None:
        """Send alert notification to external systems.
        
        Args:
            alert: Alert dictionary
        """
        # Placeholder for external alert systems
        # In production, implement integrations with:
        # - Slack
        # - PagerDuty
        # - Email
        # - SMS
        # - Webhook endpoints
        
        logger.debug(f"Alert notification sent: {alert['type']}")

    def _cleanup_old_records(self) -> None:
        """Clean up old monitoring records."""
        cutoff_time = datetime.now() - self.monitoring_window * 2  # Keep 2x window for analysis
        
        # Clean predictions log
        self.predictions_log = [
            p for p in self.predictions_log
            if p["timestamp"] >= cutoff_time
        ]
        
        # Clean performance metrics
        self.performance_metrics = [
            m for m in self.performance_metrics
            if m["timestamp"] >= cutoff_time
        ]
        
        # Keep alerts for longer (7 days)
        alert_cutoff = datetime.now() - timedelta(days=7)
        self.alerts_log = [
            a for a in self.alerts_log
            if a["timestamp"] >= alert_cutoff
        ]


class HealthChecker:
    """System health checking utilities."""
    
    @staticmethod
    def check_model_health(model) -> Dict[str, Any]:
        """Check model health status.
        
        Args:
            model: Model instance to check
            
        Returns:
            Health status dictionary
        """
        health = {
            "model_loaded": model is not None,
            "model_fitted": getattr(model, "is_fitted", False) if model else False,
            "timestamp": datetime.now()
        }
        
        if model and hasattr(model, "models"):
            health["ensemble_size"] = len(model.models)
            health["ensemble_weights"] = model.ensemble_weights
        
        return health
    
    @staticmethod
    def check_data_health(data_dir: str) -> Dict[str, Any]:
        """Check data availability and health.
        
        Args:
            data_dir: Data directory path
            
        Returns:
            Data health status
        """
        from pathlib import Path
        
        data_path = Path(data_dir)
        
        health = {
            "data_dir_exists": data_path.exists(),
            "raw_data_available": (data_path / "raw").exists(),
            "processed_data_available": (data_path / "processed").exists(),
            "timestamp": datetime.now()
        }
        
        if health["raw_data_available"]:
            raw_files = list((data_path / "raw").glob("*.csv"))
            health["raw_files_count"] = len(raw_files)
            health["latest_raw_file"] = max(raw_files, key=lambda p: p.stat().st_mtime).name if raw_files else None
        
        return health


def main() -> None:
    """Main function for testing monitoring functionality."""
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Simulate some predictions
    import random
    
    for i in range(100):
        input_data = {
            "longitude": random.uniform(-120, -115),
            "latitude": random.uniform(35, 40),
            "date": datetime.now()
        }
        
        prediction = {"predicted_temperature": random.uniform(15, 30)}
        latency = random.uniform(0.1, 2.0)
        actual_value = prediction["predicted_temperature"] + random.uniform(-2, 2)
        
        monitor.log_prediction(input_data, prediction, latency, actual_value)
    
    # Calculate performance
    performance = monitor.calculate_model_performance()
    print(f"Performance metrics: {performance}")
    
    # Get dashboard data
    dashboard_data = monitor.get_monitoring_dashboard_data()
    print(f"Dashboard data: {json.dumps(dashboard_data, default=str, indent=2)}")


if __name__ == "__main__":
    main()