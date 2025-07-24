"""Simplified Model Monitoring and Metrics Module.

This module provides basic monitoring capabilities for the climate model
including performance tracking and simple metrics collection.
"""

import logging
from datetime import datetime
from typing import Any, Dict

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
            "model_predictions_total", "Total number of predictions made"
        )
        self.prediction_latency = Histogram(
            "model_prediction_duration_seconds", "Time spent on predictions"
        )
        self.model_accuracy = Gauge(
            "model_accuracy_score", "Current model accuracy"
        )  # noqa: E501

    def log_prediction(
        self, prediction_data: Dict[str, Any], latency: float
    ) -> None:  # noqa: E501
        """Log a prediction for monitoring.

        Args:
            prediction_data: Information about the prediction
            latency: Time taken for prediction
        """
        # Update Prometheus metrics
        self.prediction_counter.inc()
        self.prediction_latency.observe(latency)

        # Store prediction data
        log_entry = {
            "timestamp": datetime.now(),
            "prediction": prediction_data.get("prediction"),
            "features": prediction_data.get("features", {}),
            "latency": latency,
        }

        self.predictions_log.append(log_entry)

        # Keep only recent entries (last 1000)
        if len(self.predictions_log) > 1000:
            self.predictions_log = self.predictions_log[-1000:]

    def log_performance(self, actual: float, predicted: float) -> None:
        """Log actual vs predicted values for performance tracking.

        Args:
            actual: Actual value
            predicted: Predicted value
        """
        error = abs(actual - predicted)

        performance_entry = {
            "timestamp": datetime.now(),
            "actual": actual,
            "predicted": predicted,
            "error": error,
        }

        self.performance_metrics.append(performance_entry)

        # Keep only recent entries
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]

        # Update accuracy metric (simple MAE over recent predictions)
        if len(self.performance_metrics) >= 10:
            recent_errors = [
                m["error"] for m in self.performance_metrics[-10:]
            ]  # noqa: E501
            avg_error = np.mean(recent_errors)
            # Convert MAE to accuracy-like score (lower error = higher score)
            accuracy_score = max(
                0, 1 - (avg_error / 10)
            )  # Normalize by expected error range
            self.model_accuracy.set(accuracy_score)

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic monitoring statistics.

        Returns:
            Dictionary of basic stats
        """
        if not self.performance_metrics:
            return {"status": "No performance data available"}

        recent_metrics = self.performance_metrics[
            -100:
        ]  # Last 100 predictions  # noqa: E501
        errors = [m["error"] for m in recent_metrics]

        return {
            "total_predictions": len(self.predictions_log),
            "recent_predictions": len(recent_metrics),
            "avg_error": np.mean(errors) if errors else 0,
            "max_error": np.max(errors) if errors else 0,
            "min_error": np.min(errors) if errors else 0,
            "last_updated": datetime.now(),
        }


def main() -> None:
    """Main function for testing monitoring."""
    monitor = SimpleModelMonitor()

    # Simulate some predictions
    for i in range(10):
        # Simulate prediction
        prediction_data = {
            "prediction": 20 + np.random.normal(0, 2),
            "features": {"temp": 15 + i, "humidity": 0.6},
        }
        latency = np.random.uniform(0.1, 0.5)

        monitor.log_prediction(prediction_data, latency)

        # Simulate actual value
        actual = prediction_data["prediction"] + np.random.normal(0, 1)
        monitor.log_performance(actual, prediction_data["prediction"])

    # Get stats
    stats = monitor.get_basic_stats()
    logger.info(f"Monitoring stats: {stats}")


if __name__ == "__main__":
    main()
