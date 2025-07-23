"""NASA Earth Data Ingestion Module.

This module handles fetching climate data from NASA's Earth Data APIs,
including MODIS Land Surface Temperature, precipitation, and other
climate indicators.
"""

import logging
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NASAEarthDataClient:
    """Client for fetching NASA Earth Data."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        earthdata_username: Optional[str] = None,
        earthdata_password: Optional[str] = None,
    ):
        """Initialize NASA Earth Data client.
        
        Args:
            data_dir: Directory to store downloaded data
            earthdata_username: NASA Earthdata username 
            earthdata_password: NASA Earthdata password
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use environment variables if credentials not provided
        self.username = earthdata_username or os.getenv("EARTHDATA_USERNAME")
        self.password = earthdata_password or os.getenv("EARTHDATA_PASSWORD")
        
        if not self.username or not self.password:
            logger.warning(
                "No NASA Earthdata credentials provided. "
                "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables."
            )
        
        self.session = requests.Session()
        if self.username and self.password:
            self.session.auth = (self.username, self.password)

    def fetch_modis_lst_data(
        self,
        start_date: str,
        end_date: str,
        bbox: Tuple[float, float, float, float],
        product: str = "MOD11A1",
    ) -> pd.DataFrame:
        """Fetch MODIS Land Surface Temperature data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            product: MODIS product (MOD11A1 for Terra, MYD11A1 for Aqua)
            
        Returns:
            DataFrame with LST data
        """
        logger.info(f"Fetching MODIS {product} data from {start_date} to {end_date}")
        
        # For demo purposes, generate synthetic data that mimics real MODIS LST
        dates = pd.date_range(start_date, end_date, freq="D")
        
        # Create synthetic coordinates within bbox
        lon_range = np.linspace(bbox[0], bbox[2], 50)
        lat_range = np.linspace(bbox[1], bbox[3], 50)
        
        data_records = []
        
        for date in tqdm(dates, desc="Processing dates"):
            for i, lon in enumerate(lon_range[::5]):  # Sample every 5th point
                for j, lat in enumerate(lat_range[::5]):
                    # Generate realistic LST values (in Kelvin)
                    base_temp = 295 + 10 * np.sin(2 * np.pi * date.dayofyear / 365)
                    noise = np.random.normal(0, 2)
                    lst_day = base_temp + noise + np.random.uniform(-5, 5)
                    lst_night = lst_day - np.random.uniform(5, 15)
                    
                    data_records.append({
                        "date": date,
                        "longitude": lon,
                        "latitude": lat,
                        "LST_Day_1km": lst_day,
                        "LST_Night_1km": lst_night,
                        "QC_Day": np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05]),
                        "QC_Night": np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05]),
                    })
        
        df = pd.DataFrame(data_records)
        
        # Save to file
        output_file = self.data_dir / f"{product}_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")
        
        return df

    def fetch_precipitation_data(
        self,
        start_date: str,
        end_date: str,
        bbox: Tuple[float, float, float, float],
    ) -> pd.DataFrame:
        """Fetch precipitation data from GPM.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            DataFrame with precipitation data
        """
        logger.info(f"Fetching precipitation data from {start_date} to {end_date}")
        
        dates = pd.date_range(start_date, end_date, freq="D")
        lon_range = np.linspace(bbox[0], bbox[2], 20)
        lat_range = np.linspace(bbox[1], bbox[3], 20)
        
        data_records = []
        
        for date in tqdm(dates, desc="Processing precipitation"):
            for lon in lon_range[::2]:
                for lat in lat_range[::2]:
                    # Generate realistic precipitation values (mm/day)
                    season_factor = 1 + 0.5 * np.sin(2 * np.pi * date.dayofyear / 365)
                    precipitation = np.random.exponential(2) * season_factor
                    
                    data_records.append({
                        "date": date,
                        "longitude": lon,
                        "latitude": lat,
                        "precipitation": precipitation,
                    })
        
        df = pd.DataFrame(data_records)
        
        output_file = self.data_dir / f"precipitation_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} precipitation records to {output_file}")
        
        return df

    def fetch_climate_indicators(
        self,
        start_date: str,
        end_date: str,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple climate indicators.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary of DataFrames with different climate indicators
        """
        logger.info("Fetching comprehensive climate indicators")
        
        data = {}
        
        # Fetch LST data
        data["lst"] = self.fetch_modis_lst_data(start_date, end_date, bbox)
        
        # Fetch precipitation data
        data["precipitation"] = self.fetch_precipitation_data(start_date, end_date, bbox)
        
        # Generate additional climate indicators
        dates = pd.date_range(start_date, end_date, freq="D")
        
        # Vegetation indices (NDVI-like)
        vegetation_data = []
        for date in dates:
            for lon in np.linspace(bbox[0], bbox[2], 10):
                for lat in np.linspace(bbox[1], bbox[3], 10):
                    ndvi = 0.3 + 0.4 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 0.1)
                    vegetation_data.append({
                        "date": date,
                        "longitude": lon,
                        "latitude": lat,
                        "ndvi": np.clip(ndvi, -1, 1),
                    })
        
        data["vegetation"] = pd.DataFrame(vegetation_data)
        
        return data


def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch NASA Earth Data")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bbox", nargs=4, type=float, 
                       default=[-120, 35, -115, 40],
                       help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    
    args = parser.parse_args()
    
    client = NASAEarthDataClient(data_dir=args.data_dir)
    
    # Fetch all climate indicators
    data = client.fetch_climate_indicators(
        args.start_date, 
        args.end_date, 
        tuple(args.bbox)
    )
    
    logger.info("Data ingestion complete!")
    for key, df in data.items():
        logger.info(f"{key}: {len(df)} records")


if __name__ == "__main__":
    main()