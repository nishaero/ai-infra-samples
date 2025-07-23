"""Simplified NASA Earth Data Ingestion Module.

This module handles generating synthetic climate data for learning purposes,
simulating NASA MODIS Land Surface Temperature and climate indicators.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateDataGenerator:
    """Simple climate data generator for learning purposes."""

    def __init__(self, data_dir: str = "data/raw"):
        """Initialize climate data generator.
        
        Args:
            data_dir: Directory to store generated data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_temperature_data(
        self,
        start_date: str,
        end_date: str,
        bbox: Tuple[float, float, float, float],
    ) -> pd.DataFrame:
        """Generate synthetic temperature data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            DataFrame with temperature data
        """
        logger.info(f"Generating temperature data from {start_date} to {end_date}")
        
        dates = pd.date_range(start_date, end_date, freq="D")
        
        # Create coordinates within bbox
        lon_range = np.linspace(bbox[0], bbox[2], 10)  # Simplified grid
        lat_range = np.linspace(bbox[1], bbox[3], 10)
        
        data_records = []
        
        for date in dates:
            for lon in lon_range:
                for lat in lat_range:
                    # Generate realistic temperature values (in Celsius)
                    base_temp = 20 + 15 * np.sin(2 * np.pi * date.dayofyear / 365)
                    noise = np.random.normal(0, 3)
                    temperature = base_temp + noise
                    
                    data_records.append({
                        "date": date,
                        "longitude": lon,
                        "latitude": lat,
                        "temperature": temperature,
                    })
        
        df = pd.DataFrame(data_records)
        
        # Save to file
        output_file = self.data_dir / f"temperature_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} temperature records to {output_file}")
        
        return df

    def generate_climate_data(
        self,
        start_date: str,
        end_date: str,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic climate data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary of DataFrames with climate data
        """
        logger.info("Generating climate data")
        
        data = {}
        
        # Generate temperature data
        data["temperature"] = self.generate_temperature_data(start_date, end_date, bbox)
        
        # Generate precipitation data
        dates = pd.date_range(start_date, end_date, freq="D")
        precipitation_data = []
        
        for date in dates:
            for lon in np.linspace(bbox[0], bbox[2], 10):
                for lat in np.linspace(bbox[1], bbox[3], 10):
                    # Simple precipitation pattern
                    precip = max(0, np.random.normal(2, 1))
                    
                    precipitation_data.append({
                        "date": date,
                        "longitude": lon,
                        "latitude": lat,
                        "precipitation": precip,
                    })
        
        data["precipitation"] = pd.DataFrame(precipitation_data)
        
        return data


def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Climate Data")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bbox", nargs=4, type=float, 
                       default=[-120, 35, -115, 40],
                       help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    
    args = parser.parse_args()
    
    generator = ClimateDataGenerator(data_dir=args.data_dir)
    
    # Generate climate data
    data = generator.generate_climate_data(
        args.start_date, 
        args.end_date, 
        tuple(args.bbox)
    )
    
    logger.info("Data generation complete!")
    for key, df in data.items():
        logger.info(f"{key}: {len(df)} records")


if __name__ == "__main__":
    main()