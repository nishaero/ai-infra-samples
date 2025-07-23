"""Simplified Data Preprocessing Module.

This module handles basic preprocessing of climate data for machine learning,
including data cleaning and simple feature engineering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SimpleClimatePreprocessor:
    """Simple preprocessor for climate data."""

    def __init__(self, data_dir: str = "data"):
        """Initialize preprocessor.
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()

    def load_raw_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load raw data files.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of raw DataFrames
        """
        data = {}
        
        # Load temperature data
        temp_file = self.raw_dir / f"temperature_{start_date}_{end_date}.csv"
        if temp_file.exists():
            data["temperature"] = pd.read_csv(temp_file, parse_dates=["date"])
            logger.info(f"Loaded temperature data: {len(data['temperature'])} records")
        
        # Load precipitation data  
        precip_file = self.raw_dir / f"precipitation_{start_date}_{end_date}.csv"
        if precip_file.exists():
            data["precipitation"] = pd.read_csv(precip_file, parse_dates=["date"])
            logger.info(f"Loaded precipitation data: {len(data['precipitation'])} records")
        
        return data

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic time-based features.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Extract basic time components
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        
        # Simple seasonal encoding
        df["season_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["season_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        
        return df

    def merge_datasets(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge climate datasets.
        
        Args:
            data: Dictionary of DataFrames to merge
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging climate datasets")
        
        if "temperature" not in data:
            raise ValueError("Temperature data is required")
        
        merged = data["temperature"].copy()
        
        # Add precipitation data
        if "precipitation" in data:
            merged = pd.merge(
                merged,
                data["precipitation"],
                on=["date", "longitude", "latitude"],
                how="left"
            )
            # Fill missing precipitation with 0
            merged["precipitation"] = merged["precipitation"].fillna(0)
            logger.info("Added precipitation data")
        
        return merged

    def prepare_training_data(
        self,
        start_date: str,
        end_date: str,
        target_col: str = "temperature"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create simple training dataset.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            target_col: Target column name
            
        Returns:
            Tuple of (processed_df, feature_columns)
        """
        logger.info(f"Creating training data from {start_date} to {end_date}")
        
        # Load raw data
        data = self.load_raw_data(start_date, end_date)
        
        if not data:
            raise ValueError("No data found for the specified date range")
        
        # Merge datasets
        merged_df = self.merge_datasets(data)
        
        # Create basic features
        processed_df = self.create_basic_features(merged_df)
        
        # Remove any rows with missing target
        processed_df = processed_df.dropna(subset=[target_col])
        
        # Define feature columns
        feature_cols = ["longitude", "latitude", "month", "day_of_year", 
                       "season_sin", "season_cos"]
        
        # Add precipitation if available
        if "precipitation" in processed_df.columns:
            feature_cols.append("precipitation")
        
        # Scale features
        processed_df[feature_cols] = self.scaler.fit_transform(processed_df[feature_cols])
        
        # Save processed data
        output_file = self.processed_dir / f"training_data_{start_date}_{end_date}.csv"
        processed_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed training data to {output_file}")
        
        logger.info(f"Created training dataset: {len(processed_df)} samples, {len(feature_cols)} features")
        
        return processed_df, feature_cols


def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Climate Data")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--target", default="temperature", help="Target column")
    
    args = parser.parse_args()
    
    preprocessor = SimpleClimatePreprocessor(data_dir=args.data_dir)
    
    # Create training data
    df, features = preprocessor.prepare_training_data(
        args.start_date,
        args.end_date,
        args.target
    )
    
    logger.info("Data preprocessing complete!")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {features}")


if __name__ == "__main__":
    main()