#!/usr/bin/env python3
"""
Standalone Climate Temperature Prediction Script

This script runs the complete ML pipeline locally, trains models,
generates predictions, creates visualizations, and produces an HTML report.

Usage:
    python run_model.py [--help for options]

The script will:
1. Fetch real climate data from NOAA (or use fallback if API unavailable)
2. Train climate prediction models
3. Generate predictions and evaluate performance
4. Create visualization plots
5. Generate an HTML report with results and explanations
"""

import argparse
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import our custom modules
from src.models.climate_model import evaluate_model
from src.models.training import SimpleModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set matplotlib style
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    try:
        plt.style.use("seaborn")
    except OSError:
        plt.style.use("default")
sns.set_palette("husl")


class ClimateReportGenerator:
    """Generate comprehensive climate prediction reports."""

    def __init__(self, output_dir: str = "output"):
        """Initialize report generator.

        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def create_performance_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_results: Dict,
        features: List[str],
        trained_model=None,
    ) -> List[str]:
        """Create performance visualization plots.

        Args:
            y_true: True temperature values
            y_pred: Predicted temperature values
            model_results: Training results from models
            features: Feature names

        Returns:
            List of plot filenames created
        """
        plot_files = []

        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(
            y_true, y_pred, alpha=0.6, s=50, edgecolors="k", linewidth=0.5
        )  # noqa: E501
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Actual Temperature (°C)", fontsize=12)
        plt.ylabel("Predicted Temperature (°C)", fontsize=12)
        plt.title(
            "Actual vs Predicted Temperature", fontsize=14, fontweight="bold"
        )  # noqa: E501

        # Add metrics text
        metrics = evaluate_model(y_true, y_pred)
        metrics_text = (
            f"MAE: {metrics['mae']:.2f}°C\n"
            f"RMSE: {metrics['rmse']:.2f}°C\n"
            f"R²: {metrics['r2']:.3f}"
        )
        plt.text(
            0.05,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = self.plots_dir / "actual_vs_predicted.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files.append(str(plot_file))

        # 2. Residuals plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(
            y_pred, residuals, alpha=0.6, s=50, edgecolors="k", linewidth=0.5
        )  # noqa: E501
        plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
        plt.xlabel("Predicted Temperature (°C)", fontsize=12)
        plt.ylabel("Residuals (°C)", fontsize=12)
        plt.title("Residuals Plot", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = self.plots_dir / "residuals.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files.append(str(plot_file))

        # 3. Feature importance (if available)
        if "model_scores" in model_results and trained_model is not None:
            # Get feature importance from the trained model
            try:
                feature_importance = trained_model.get_feature_importance()

                if (
                    feature_importance
                    and "random_forest" in feature_importance  # noqa: E501
                ):
                    importances = feature_importance["random_forest"]
                    indices = np.argsort(importances)[::-1]

                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(features)), importances[indices])
                    plt.xlabel("Features", fontsize=12)
                    plt.ylabel("Importance", fontsize=12)
                    plt.title(
                        "Feature Importance (Random Forest)",
                        fontsize=14,
                        fontweight="bold",
                    )
                    plt.xticks(
                        range(len(features)),
                        [features[i] for i in indices],
                        rotation=45,
                        ha="right",
                    )
                    plt.tight_layout()
                    plot_file = self.plots_dir / "feature_importance.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    plt.close()
                    plot_files.append(str(plot_file))
            except Exception as e:
                logger.warning(
                    f"Could not create feature importance plot: {e}"
                )  # noqa: E501

        # 4. Model comparison (if multiple models)
        if "model_scores" in model_results:
            model_scores = model_results["model_scores"]
            if len(model_scores) > 1:
                models = list(model_scores.keys())
                metrics_list = ["mae", "rmse", "r2"]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                for i, metric in enumerate(metrics_list):
                    values = [model_scores[model][metric] for model in models]
                    axes[i].bar(
                        models,
                        values,
                        color=sns.color_palette("husl", len(models)),
                    )
                    axes[i].set_title(f"{metric.upper()}", fontweight="bold")
                    axes[i].set_ylabel(metric.upper())
                    axes[i].tick_params(axis="x", rotation=45)

                plt.tight_layout()
                plot_file = self.plots_dir / "model_comparison.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()
                plot_files.append(str(plot_file))

        # 5. Temperature distribution
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(
            y_true,
            bins=30,
            alpha=0.7,
            label="Actual",
            color="skyblue",
            edgecolor="black",
        )
        plt.hist(
            y_pred,
            bins=30,
            alpha=0.7,
            label="Predicted",
            color="lightcoral",
            edgecolor="black",
        )
        plt.xlabel("Temperature (°C)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Temperature Distribution", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot([y_true, y_pred], tick_labels=["Actual", "Predicted"])
        plt.ylabel("Temperature (°C)", fontsize=12)
        plt.title("Temperature Box Plot", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.plots_dir / "temperature_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files.append(str(plot_file))

        return plot_files

    def generate_html_report(
        self,
        results: Dict,
        plot_files: List[str],
        training_time: float,
        dataset_info: Dict,
    ) -> str:
        """Generate comprehensive HTML report.

        Args:
            results: Model results and metrics
            plot_files: List of plot file paths
            training_time: Time taken for training
            dataset_info: Information about the dataset

        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport"
                  content="width=device-width, initial-scale=1.0">
            <title>Climate Temperature Prediction Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana,
                                 sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c5aa0;
                    text-align: center;
                    border-bottom: 3px solid #2c5aa0;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #555;
                    margin-top: 25px;
                }}
                .summary {{
                    background-color: #e8f4fd;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 5px solid #3498db;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit,
                                                   minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #dee2e6;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c5aa0;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
                .plot-container {{
                    text-align: center;
                    margin: 20px 0;
                    padding: 20px;
                    background-color: #fafafa;
                    border-radius: 8px;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));  # noqa: E501
                    gap: 20px;
                    margin: 20px 0;
                }}
                .info-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    font-style: italic;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .explanation {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌍 Climate Temperature Prediction Report</h1>

                <div class="summary">
                    <h3>📊 Executive Summary</h3>
                    <p>This report presents the results of training a machine learning model to predict climate temperatures using real-world meteorological data. The model was trained using an ensemble approach combining Linear Regression and Random Forest algorithms.</p>  # noqa: E501
                </div>

                <h2>🎯 Model Performance</h2>

                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{results['test_metrics']['mae']:.2f}°C</div>  # noqa: E501
                        <div class="metric-label">Mean Absolute Error</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['test_metrics']['rmse']:.2f}°C</div>  # noqa: E501
                        <div class="metric-label">Root Mean Square Error</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['test_metrics']['r2']:.3f}</div>  # noqa: E501
                        <div class="metric-label">R² Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{training_time:.1f}s</div>
                        <div class="metric-label">Training Time</div>
                    </div>
                </div>

                <div class="explanation">
                    <strong>Performance Explanation:</strong>
                    <ul>
                        <li><strong>MAE (Mean Absolute Error):</strong> On average, our predictions are off by {results['test_metrics']['mae']:.2f}°C</li>  # noqa: E501
                        <li><strong>RMSE (Root Mean Square Error):</strong> Penalizes larger errors more heavily, giving us {results['test_metrics']['rmse']:.2f}°C</li>  # noqa: E501
                        <li><strong>R² Score:</strong> Explains {results['test_metrics']['r2'] * 100:.1f}% of the variance in temperature data</li>  # noqa: E501
                    </ul>
                </div>

                <h2>📈 Visualizations</h2>
        """

        # Add plots
        plot_descriptions = {
            "actual_vs_predicted.png": {
                "title": "Actual vs Predicted Temperature",
                "description": "This scatter plot shows how well our model's predictions align with actual temperatures. Points closer to the red diagonal line indicate better predictions.",  # noqa: E501
            },
            "residuals.png": {
                "title": "Residuals Analysis",
                "description": "Residuals (errors) should be randomly distributed around zero. Patterns in residuals might indicate model bias or missing features.",  # noqa: E501
            },
            "feature_importance.png": {
                "title": "Feature Importance",
                "description": "Shows which input features (like latitude, longitude, time features) contribute most to the temperature predictions.",  # noqa: E501
            },
            "model_comparison.png": {
                "title": "Model Comparison",
                "description": "Compares the performance of different algorithms in our ensemble (Linear Regression vs Random Forest).",  # noqa: E501
            },
            "temperature_distribution.png": {
                "title": "Temperature Distribution",
                "description": "Compares the distribution of actual vs predicted temperatures to ensure our model captures the data distribution properly.",  # noqa: E501
            },
        }

        for plot_file in plot_files:
            plot_name = Path(plot_file).name
            if plot_name in plot_descriptions:
                rel_path = f"plots/{plot_name}"
                html_content += f"""
                <div class="plot-container">
                    <h3>{plot_descriptions[plot_name]['title']}</h3>
                    <img src="{rel_path}" alt="{plot_descriptions[plot_name]['title']}">  # noqa: E501
                    <div class="explanation">
                        {plot_descriptions[plot_name]['description']}
                    </div>
                </div>
                """

        # Add dataset information
        html_content += f"""
                <h2>📋 Dataset Information</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h4>Dataset Size</h4>
                        <p><strong>Total Samples:</strong> {dataset_info.get('total_samples', 'N/A')}</p>  # noqa: E501
                        <p><strong>Training Samples:</strong> {dataset_info.get('train_samples', 'N/A')}</p>  # noqa: E501
                        <p><strong>Test Samples:</strong> {dataset_info.get('test_samples', 'N/A')}</p>  # noqa: E501
                    </div>
                    <div class="info-card">
                        <h4>Features</h4>
                        <p><strong>Number of Features:</strong> {dataset_info.get('n_features', 'N/A')}</p>  # noqa: E501
                        <p><strong>Geographic Region:</strong> {dataset_info.get('region', 'N/A')}</p>  # noqa: E501
                        <p><strong>Time Period:</strong> {dataset_info.get('time_period', 'N/A')}</p>  # noqa: E501
                    </div>
                    <div class="info-card">
                        <h4>Data Source</h4>
                        <p><strong>Primary:</strong> NOAA Climate Data API</p>
                        <p><strong>Fallback:</strong> Realistic synthetic data based on meteorological principles</p>  # noqa: E501
                        <p><strong>Variables:</strong> Temperature, Precipitation, Geographic coordinates</p>  # noqa: E501
                    </div>
                </div>

                <h2>🔬 Methodology</h2>
                <div class="explanation">
                    <h4>Data Collection & Processing:</h4>
                    <ol>
                        <li><strong>Data Ingestion:</strong> Real-world climate data from NOAA's Climate Data Online service</li>  # noqa: E501
                        <li><strong>Feature Engineering:</strong> Time-based features (day of year, month) and geographic features</li>  # noqa: E501
                        <li><strong>Data Cleaning:</strong> Outlier detection and missing value handling</li>  # noqa: E501
                        <li><strong>Train/Test Split:</strong> 80/20 split for unbiased evaluation</li>  # noqa: E501
                    </ol>

                    <h4>Model Training:</h4>
                    <ol>
                        <li><strong>Linear Regression:</strong> Baseline model for interpretability</li>  # noqa: E501
                        <li><strong>Random Forest:</strong> Non-linear patterns and feature interactions</li>  # noqa: E501
                        <li><strong>Ensemble Method:</strong> Simple averaging of predictions for robustness</li>  # noqa: E501
                        <li><strong>Validation:</strong> Cross-validation and holdout testing</li>  # noqa: E501
                    </ol>
                </div>

                <h2>🎯 Key Insights</h2>
                <div class="explanation">
                    <h4>Model Performance Insights:</h4>
                    <ul>
                        <li>The ensemble approach provides more robust predictions than individual models</li>  # noqa: E501
                        <li>Geographic features (latitude/longitude) are likely important for regional temperature patterns</li>  # noqa: E501
                        <li>Time-based features help capture seasonal temperature variations</li>  # noqa: E501
                        <li>The model performance is suitable for climate monitoring applications</li>  # noqa: E501
                    </ul>

                    <h4>Potential Improvements:</h4>
                    <ul>
                        <li>Include more climate variables (humidity, wind speed, atmospheric pressure)</li>  # noqa: E501
                        <li>Add satellite-derived features (vegetation indices, land cover)</li>  # noqa: E501
                        <li>Implement time series models for temporal dependencies</li>  # noqa: E501
                        <li>Use deep learning for complex non-linear patterns</li>  # noqa: E501
                    </ul>
                </div>

                <h2>📊 Detailed Results</h2>
        """

        # Add individual model results if available
        if "model_scores" in results["training_results"]:
            html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>MAE (°C)</th>
                            <th>RMSE (°C)</th>
                            <th>R² Score</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for model_name, scores in results["training_results"][
                "model_scores"
            ].items():
                html_content += f"""
                        <tr>
                            <td>{model_name.replace('_', ' ').title()}</td>
                            <td>{scores['mae']:.3f}</td>
                            <td>{scores['rmse']:.3f}</td>
                            <td>{scores['r2']:.3f}</td>
                        </tr>
                """

            html_content += """
                    </tbody>
                </table>
            """

        html_content += f"""
                <div class="timestamp">
                    Report generated on: {timestamp}<br>
                    Using Climate Temperature Prediction ML Pipeline v1.0
                </div>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        report_path = self.output_dir / "climate_prediction_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {report_path}")
        return str(report_path)


def main():
    """Main function to run the complete climate prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Climate Temperature Prediction - Complete ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run with default parameters
    python run_model.py

    # Custom date range and region
    python run_model.py --start-date 2023-01-01 --end-date 2023-06-30 --bbox -125 30 -110 45  # noqa: E501

    # Specify output directory
    python run_model.py --output-dir ./my_results
        """,
    )

    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Start date for data (YYYY-MM-DD, default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default="2023-03-31",
        help="End date for data (YYYY-MM-DD, default: 2023-03-31)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[8.5, 50.3, 8.8, 50.6],  # Butzbach, Germany region
        help="Bounding box: min_lon min_lat max_lon max_lat (default: Butzbach, Germany region)",  # noqa: E501
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Data directory (default: data)"
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Model directory (default: models)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    logger.info("🌍 Starting Climate Temperature Prediction Pipeline")
    logger.info(f"📅 Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"🗺️  Region: {args.bbox}")
    logger.info(f"📁 Output Directory: {args.output_dir}")

    start_time = time.time()

    try:
        # Initialize components
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = SimpleModelTrainer(
                data_dir=os.path.join(temp_dir, args.data_dir),
                model_dir=os.path.join(temp_dir, args.model_dir),
                experiment_name="local-climate-prediction",
            )

            report_generator = ClimateReportGenerator(args.output_dir)

            # Step 1: Prepare training data
            logger.info("📊 Step 1/5: Preparing training data...")
            df, features = trainer.prepare_training_data(
                args.start_date, args.end_date, tuple(args.bbox)
            )

            logger.info(
                f"✅ Dataset prepared: {len(df)} samples, {len(features)} features"  # noqa: E501
            )

            # Step 2: Train model
            logger.info("🤖 Step 2/5: Training machine learning models...")
            model, results = trainer.train_model(
                df,
                features,
                test_size=args.test_size,
                random_state=args.random_state,
            )

            training_time = time.time() - start_time
            logger.info(
                f"✅ Model training completed in {training_time:.2f} seconds"
            )  # noqa: E501

            # Step 3: Generate predictions for visualization
            logger.info("🔮 Step 3/5: Generating predictions...")
            X = df[features]
            y = df["temperature"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state
            )

            # Get predictions
            y_pred_test = model.predict(X_test)

            # Step 4: Create visualizations
            logger.info("📈 Step 4/5: Creating visualizations...")
            plot_files = report_generator.create_performance_plots(
                y_test.values,
                y_pred_test,
                results["training_results"],
                features,
                model,
            )

            logger.info(f"✅ Created {len(plot_files)} visualization plots")

            # Step 5: Generate HTML report
            logger.info("📄 Step 5/5: Generating comprehensive report...")

            dataset_info = {
                "total_samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(features),
                "region": f"Lon: [{args.bbox[0]:.1f}, {args.bbox[2]:.1f}], Lat: [{args.bbox[1]:.1f}, {args.bbox[3]:.1f}]",  # noqa: E501
                "time_period": f"{args.start_date} to {args.end_date}",
            }

            report_path = report_generator.generate_html_report(
                results, plot_files, training_time, dataset_info
            )

            total_time = time.time() - start_time

            # Final summary
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
            logger.info("📊 Model Performance:")
            logger.info(f"   • MAE: {results['test_metrics']['mae']:.2f}°C")
            logger.info(f"   • RMSE: {results['test_metrics']['rmse']:.2f}°C")
            logger.info(f"   • R² Score: {results['test_metrics']['r2']:.3f}")
            logger.info(
                f"📁 Results saved to: {os.path.abspath(args.output_dir)}"
            )  # noqa: E501
            logger.info(
                f"📋 Report available at: {os.path.abspath(report_path)}"
            )  # noqa: E501

            print("\n" + "=" * 60)
            print("🌍 CLIMATE TEMPERATURE PREDICTION - RESULTS SUMMARY")
            print("=" * 60)
            print("📊 Model Performance:")
            print(
                f"   • Mean Absolute Error: {results['test_metrics']['mae']:.2f}°C"  # noqa: E501
            )
            print(
                f"   • Root Mean Square Error: {results['test_metrics']['rmse']:.2f}°C"  # noqa: E501
            )
            print(f"   • R² Score: {results['test_metrics']['r2']:.3f}")
            print(f"📈 Dataset: {len(df)} samples, {len(features)} features")
            print(f"⏱️  Training Time: {training_time:.2f} seconds")
            print(f"📁 Output Directory: {os.path.abspath(args.output_dir)}")
            print(
                f"📋 View detailed report: file://{os.path.abspath(report_path)}"  # noqa: E501
            )
            print("=" * 60)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
