"""End-to-end integration tests for the climate prediction pipeline."""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.ingestion import NOAAClimateDataClient
from src.data.preprocessing import SimpleClimatePreprocessor
from src.models.climate_model import SimpleSimpleClimatePredictor
from src.models.training import SimpleModelTrainer


class TestEndToEndPipeline:
    """Test the complete end-to-end ML pipeline."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration test."""
        temp_dir = tempfile.mkdtemp()
        workspace = {
            "base_dir": temp_dir,
            "data_dir": Path(temp_dir) / "data",
            "model_dir": Path(temp_dir) / "models"
        }
        
        # Create subdirectories
        (workspace["data_dir"] / "raw").mkdir(parents=True)
        (workspace["data_dir"] / "processed").mkdir(parents=True)
        workspace["model_dir"].mkdir(parents=True)
        
        yield workspace
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_complete_pipeline(self, temp_workspace):
        """Test the complete ML pipeline from data ingestion to prediction."""
        
        # 1. Data Ingestion
        print("Step 1: Data Ingestion")
        client = NOAAClimateDataClient(
            data_dir=str(temp_workspace["data_dir"] / "raw")
        )
        
        bbox = (-120, 35, -115, 40)
        start_date = "2023-01-01"
        end_date = "2023-01-10"  # Small date range for fast testing
        
        # Fetch climate indicators
        data = client.fetch_climate_data(start_date, end_date, bbox)
        
        # Verify data ingestion
        assert isinstance(data, dict)
        assert "lst" in data
        assert "precipitation" in data
        assert "vegetation" in data
        
        for dataset_name, df in data.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            print(f"  {dataset_name}: {len(df)} records")
        
        # 2. Data Preprocessing
        print("Step 2: Data Preprocessing")
        preprocessor = SimpleClimatePreprocessor(
            data_dir=str(temp_workspace["data_dir"])
        )
        
        # Create training data
        df, features = preprocessor.prepare_training_data(
            start_date, end_date, target_col="LST_Day_C"
        )
        
        # Verify preprocessing
        assert isinstance(df, pd.DataFrame)
        assert isinstance(features, list)
        assert len(df) > 0
        assert len(features) > 0
        assert "LST_Day_C" in df.columns
        
        print(f"  Processed dataset: {len(df)} samples, {len(features)} features")
        
        # Check that features don't include target or metadata
        exclude_cols = ["LST_Day_C", "date", "longitude", "latitude"]
        for col in exclude_cols:
            if col != "LST_Day_C":  # Target might be excluded from features
                assert col not in features or col == "LST_Day_C"
        
        # 3. Model Training
        print("Step 3: Model Training")
        predictor = SimpleClimatePredictor(
            model_dir=str(temp_workspace["model_dir"]),
            random_state=42
        )
        
        # Train model
        X = df[features]
        y = df["LST_Day_C"]
        
        training_results = predictor.fit(X, y, validation_split=0.3)
        
        # Verify training
        assert predictor.is_fitted
        assert "model_scores" in training_results
        assert "ensemble_scores" in training_results
        
        print(f"  Training completed. Ensemble MAE: {training_results['ensemble_scores']['val_mae']:.3f}")
        
        # 4. Model Evaluation
        print("Step 4: Model Evaluation")
        
        # Make predictions on the training data (for demonstration)
        predictions = predictor.predict(X)
        
        # Basic evaluation
        mae = np.mean(np.abs(predictions - y.values))
        rmse = np.sqrt(np.mean((predictions - y.values) ** 2))
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        
        # Verify predictions are reasonable
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
        
        # Check that predictions are in reasonable temperature range (Celsius)
        assert predictions.min() > -50  # Reasonable lower bound
        assert predictions.max() < 70   # Reasonable upper bound
        
        # 5. Model Persistence
        print("Step 5: Model Persistence")
        
        model_filename = "test_climate_model.joblib"
        predictor.save_model(model_filename)
        
        # Verify model file was created
        model_path = temp_workspace["model_dir"] / model_filename
        assert model_path.exists()
        
        # 6. Model Loading and Inference
        print("Step 6: Model Loading and Inference")
        
        # Create new predictor instance and load model
        new_predictor = SimpleClimatePredictor(
            model_dir=str(temp_workspace["model_dir"])
        )
        new_predictor.load_model(model_filename)
        
        # Verify loaded model
        assert new_predictor.is_fitted
        
        # Make predictions with loaded model
        new_predictions = new_predictor.predict(X.head(10))  # Test with first 10 samples
        original_predictions = predictor.predict(X.head(10))
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(
            new_predictions, original_predictions, decimal=5
        )
        
        print("  Model loading and inference verified")
        
        # 7. Feature Importance Analysis
        print("Step 7: Feature Importance Analysis")
        
        feature_importance = predictor.get_feature_importance()
        
        if feature_importance:
            for model_name, importance in feature_importance.items():
                assert len(importance) == len(features)
                # Check that importances are non-negative and sum to reasonable value
                assert all(imp >= 0 for imp in importance)
                print(f"  {model_name}: top feature importance = {importance.max():.3f}")
        
        print("Integration test completed successfully!")

    def test_trainer_integration(self, temp_workspace):
        """Test the integrated training pipeline using SimpleModelTrainer."""
        
        print("Testing SimpleModelTrainer integration")
        
        # Initialize trainer
        trainer = SimpleModelTrainer(
            data_dir=str(temp_workspace["data_dir"]),
            model_dir=str(temp_workspace["model_dir"]),
            experiment_name="test-integration"
        )
        
        # Prepare training data
        bbox = (-120, 35, -115, 40)
        start_date = "2023-01-01"
        end_date = "2023-01-05"  # Very small range for fast testing
        
        df, features = trainer.prepare_training_data(
            start_date, end_date, bbox, fetch_new_data=True
        )
        
        # Verify data preparation
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(features) > 0
        
        print(f"Data prepared: {len(df)} samples, {len(features)} features")
        
        # Train model (without MLflow for integration test)
        model, results = trainer.train_model(
            df, features, run_name="integration_test"
        )
        
        # Verify training results
        assert model.is_fitted
        assert "train_metrics" in results
        assert "test_metrics" in results
        assert "model_path" in results
        
        # Verify model file was created
        model_path = Path(results["model_path"])
        assert model_path.exists()
        
        print(f"Training completed: Test MAE = {results['test_metrics']['mae']:.3f}")
        print("SimpleModelTrainer integration test passed!")

    def test_data_pipeline_consistency(self, temp_workspace):
        """Test consistency of data pipeline across multiple runs."""
        
        print("Testing data pipeline consistency")
        
        # Configuration
        bbox = (-119, 36, -118, 37)  # Small area
        start_date = "2023-01-01"
        end_date = "2023-01-03"
        
        client = NOAAClimateDataClient(
            data_dir=str(temp_workspace["data_dir"] / "raw")
        )
        
        # Run data ingestion twice
        data1 = client.fetch_climate_data(start_date, end_date, bbox)
        data2 = client.fetch_climate_data(start_date, end_date, bbox)
        
        # Verify consistency (since we use random seed, results should be similar)
        for dataset_name in data1.keys():
            df1 = data1[dataset_name]
            df2 = data2[dataset_name]
            
            # Same number of records
            assert len(df1) == len(df2)
            
            # Same columns
            assert list(df1.columns) == list(df2.columns)
            
            # Same date range
            assert df1["date"].min() == df2["date"].min()
            assert df1["date"].max() == df2["date"].max()
        
        print("Data pipeline consistency verified!")

    def test_model_performance_threshold(self, temp_workspace):
        """Test that model meets minimum performance thresholds."""
        
        print("Testing model performance thresholds")
        
        # Quick training with minimal data
        client = NOAAClimateDataClient(
            data_dir=str(temp_workspace["data_dir"] / "raw")
        )
        
        # Generate data
        bbox = (-120, 35, -115, 40)
        data = client.fetch_climate_data("2023-01-01", "2023-01-15", bbox)
        
        # Preprocess
        preprocessor = SimpleClimatePreprocessor(
            data_dir=str(temp_workspace["data_dir"])
        )
        df, features = preprocessor.prepare_training_data(
            "2023-01-01", "2023-01-15", target_col="LST_Day_C"
        )
        
        # Train model
        predictor = SimpleClimatePredictor(random_state=42)
        X = df[features]
        y = df["LST_Day_C"]
        
        results = predictor.fit(X, y, validation_split=0.3)
        
        # Performance thresholds
        ensemble_mae = results["ensemble_scores"]["val_mae"]
        ensemble_r2 = results["ensemble_scores"]["val_r2"]
        
        # Reasonable thresholds for synthetic data
        assert ensemble_mae < 10.0, f"MAE {ensemble_mae:.3f} exceeds threshold"
        assert ensemble_r2 > 0.1, f"R² {ensemble_r2:.3f} below threshold"
        
        print(f"Performance thresholds met: MAE={ensemble_mae:.3f}, R²={ensemble_r2:.3f}")
        print("Model performance test passed!")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])