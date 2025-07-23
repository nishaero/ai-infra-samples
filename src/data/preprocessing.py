"""Data Preprocessing Module.

This module handles preprocessing of NASA Earth data for machine learning,
including data cleaning, feature engineering, and time series preparation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class ClimateDataPreprocessor:
    """Preprocessor for climate data."""

    def __init__(self, data_dir: str = "data"):
        """Initialize preprocessor.
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.scalers = {}
        self.imputers = {}

    def load_raw_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load raw data files.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of raw DataFrames
        """
        data = {}
        
        # Load LST data
        lst_file = self.raw_dir / f"MOD11A1_{start_date}_{end_date}.csv"
        if lst_file.exists():
            data["lst"] = pd.read_csv(lst_file, parse_dates=["date"])
            logger.info(f"Loaded LST data: {len(data['lst'])} records")
        
        # Load precipitation data
        precip_file = self.raw_dir / f"precipitation_{start_date}_{end_date}.csv"
        if precip_file.exists():
            data["precipitation"] = pd.read_csv(precip_file, parse_dates=["date"])
            logger.info(f"Loaded precipitation data: {len(data['precipitation'])} records")
        
        return data

    def clean_lst_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean MODIS LST data.
        
        Args:
            df: Raw LST DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning LST data")
        
        # Convert LST from Kelvin to Celsius
        df["LST_Day_C"] = df["LST_Day_1km"] - 273.15
        df["LST_Night_C"] = df["LST_Night_1km"] - 273.15
        
        # Filter out poor quality data
        df = df[(df["QC_Day"] == 0) & (df["QC_Night"] == 0)].copy()
        
        # Remove unrealistic temperature values
        df = df[
            (df["LST_Day_C"] >= -50) & (df["LST_Day_C"] <= 70) &
            (df["LST_Night_C"] >= -50) & (df["LST_Night_C"] <= 70)
        ].copy()
        
        # Calculate temperature anomalies
        daily_mean = df.groupby("date")["LST_Day_C"].mean()
        df["LST_Day_anomaly"] = df.apply(
            lambda row: row["LST_Day_C"] - daily_mean.loc[row["date"]], axis=1
        )
        
        logger.info(f"Cleaned LST data: {len(df)} records remaining")
        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Extract time components
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofyear"] = df["date"].dt.dayofyear
        df["week"] = df["date"].dt.isocalendar().week
        
        # Cyclical encoding for seasonal patterns
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
        
        return df

    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features.
        
        Args:
            df: DataFrame with longitude and latitude columns
            
        Returns:
            DataFrame with additional spatial features
        """
        df = df.copy()
        
        # Distance from centroid
        center_lon = df["longitude"].mean()
        center_lat = df["latitude"].mean()
        
        df["distance_from_center"] = np.sqrt(
            (df["longitude"] - center_lon) ** 2 + (df["latitude"] - center_lat) ** 2
        )
        
        # Elevation proxy (simplified)
        df["elevation_proxy"] = (
            100 * np.sin(df["latitude"] * np.pi / 180) + 
            50 * np.cos(df["longitude"] * np.pi / 180) +
            np.random.normal(0, 20, len(df))
        )
        
        return df

    def merge_datasets(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple climate datasets.
        
        Args:
            data: Dictionary of DataFrames to merge
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging climate datasets")
        
        # Start with LST data as base
        if "lst" not in data:
            raise ValueError("LST data is required as base dataset")
        
        merged = data["lst"].copy()
        
        # Add precipitation data
        if "precipitation" in data:
            precip_df = data["precipitation"].copy()
            merged = pd.merge(
                merged,
                precip_df,
                on=["date", "longitude", "latitude"],
                how="left"
            )
            logger.info("Added precipitation data")
        
        # Add vegetation data
        if "vegetation" in data:
            veg_df = data["vegetation"].copy()
            merged = pd.merge(
                merged,
                veg_df,
                on=["date", "longitude", "latitude"],
                how="left"
            )
            logger.info("Added vegetation data")
        
        return merged

    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        lags: List[int] = [1, 2, 3, 7, 14]
    ) -> pd.DataFrame:
        """Create lag features for time series.
        
        Args:
            df: DataFrame with time series data
            target_col: Column to create lags for
            lags: List of lag values
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        # Sort by location and date
        df = df.sort_values(["longitude", "latitude", "date"])
        
        for lag in lags:
            lag_col = f"{target_col}_lag_{lag}"
            df[lag_col] = df.groupby(["longitude", "latitude"])[target_col].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 14]:
            roll_mean = df.groupby(["longitude", "latitude"])[target_col].rolling(
                window=window, min_periods=1
            ).mean()
            df[f"{target_col}_roll_mean_{window}"] = roll_mean.values
            
            roll_std = df.groupby(["longitude", "latitude"])[target_col].rolling(
                window=window, min_periods=1
            ).std()
            df[f"{target_col}_roll_std_{window}"] = roll_std.values
        
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning.
        
        Args:
            df: Merged climate DataFrame
            
        Returns:
            DataFrame ready for ML
        """
        logger.info("Preparing features for ML")
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create spatial features
        df = self.create_spatial_features(df)
        
        # Create lag features for temperature
        df = self.create_lag_features(df, "LST_Day_C")
        df = self.create_lag_features(df, "LST_Night_C")
        
        # Create interaction features
        if "precipitation" in df.columns:
            df["temp_precip_interaction"] = df["LST_Day_C"] * df["precipitation"]
        
        if "ndvi" in df.columns:
            df["temp_vegetation_interaction"] = df["LST_Day_C"] * df["ndvi"]
        
        return df

    def scale_features(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        method: str = "standard"
    ) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: DataFrame to scale
            feature_cols: List of columns to scale
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Store scaler for later use
        self.scalers[f"{method}_scaler"] = scaler
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values")
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        
        # Impute numerical columns with median
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy="median")
            df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
            self.imputers["numerical"] = num_imputer
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.imputers["categorical"] = cat_imputer
        
        logger.info(f"Handled missing values: {df.isnull().sum().sum()} remaining")
        return df

    def create_training_data(
        self,
        start_date: str,
        end_date: str,
        target_col: str = "LST_Day_C"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create training dataset.
        
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
        
        # Clean LST data
        if "lst" in data:
            data["lst"] = self.clean_lst_data(data["lst"])
        
        # Merge datasets
        merged_df = self.merge_datasets(data)
        
        # Prepare features
        processed_df = self.prepare_features(merged_df)
        
        # Handle missing values
        processed_df = self.handle_missing_values(processed_df)
        
        # Identify feature columns (exclude target and metadata)
        exclude_cols = [
            target_col, "date", "longitude", "latitude", 
            "LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"
        ]
        feature_cols = [col for col in processed_df.columns if col not in exclude_cols]
        
        # Scale numerical features
        numerical_features = processed_df[feature_cols].select_dtypes(include=[np.number]).columns
        if len(numerical_features) > 0:
            processed_df = self.scale_features(processed_df, numerical_features.tolist())
        
        # Drop rows with NaN in target
        processed_df = processed_df.dropna(subset=[target_col])
        
        # Save processed data
        output_file = self.processed_dir / f"training_data_{start_date}_{end_date}.csv"
        processed_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed training data to {output_file}")
        
        logger.info(f"Created training dataset: {len(processed_df)} samples, {len(feature_cols)} features")
        
        return processed_df, feature_cols


def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess NASA Earth Data")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--target", default="LST_Day_C", help="Target column")
    
    args = parser.parse_args()
    
    preprocessor = ClimateDataPreprocessor(data_dir=args.data_dir)
    
    # Create training data
    df, features = preprocessor.create_training_data(
        args.start_date,
        args.end_date,
        args.target
    )
    
    logger.info("Data preprocessing complete!")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {len(features)}")


if __name__ == "__main__":
    main()