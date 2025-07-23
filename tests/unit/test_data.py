"""Unit tests for data ingestion module."""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.data.ingestion import NASAEarthDataClient


class TestNASAEarthDataClient:
    """Test NASA Earth Data client."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def client(self, temp_dir):
        """Create client instance for testing."""
        return NASAEarthDataClient(data_dir=temp_dir)

    def test_init(self, temp_dir):
        """Test client initialization."""
        client = NASAEarthDataClient(data_dir=temp_dir)
        
        assert client.data_dir == Path(temp_dir)
        assert client.data_dir.exists()
        assert client.username is None  # No credentials in test
        assert client.password is None

    def test_init_with_credentials(self, temp_dir):
        """Test client initialization with credentials."""
        client = NASAEarthDataClient(
            data_dir=temp_dir,
            earthdata_username="test_user",
            earthdata_password="test_pass"
        )
        
        assert client.username == "test_user"
        assert client.password == "test_pass"

    def test_fetch_modis_lst_data(self, client):
        """Test MODIS LST data fetching."""
        bbox = (-120, 35, -115, 40)
        
        df = client.fetch_modis_lst_data(
            "2023-01-01", "2023-01-03", bbox
        )
        
        # Check data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check required columns
        required_cols = [
            "date", "longitude", "latitude", 
            "LST_Day_1km", "LST_Night_1km", 
            "QC_Day", "QC_Night"
        ]
        for col in required_cols:
            assert col in df.columns
        
        # Check data ranges
        assert df["longitude"].min() >= bbox[0]
        assert df["longitude"].max() <= bbox[2]
        assert df["latitude"].min() >= bbox[1]
        assert df["latitude"].max() <= bbox[3]
        
        # Check temperature values are reasonable (in Kelvin)
        assert df["LST_Day_1km"].min() > 250  # Above absolute zero
        assert df["LST_Day_1km"].max() < 350  # Below extreme values

    def test_fetch_precipitation_data(self, client):
        """Test precipitation data fetching."""
        bbox = (-120, 35, -115, 40)
        
        df = client.fetch_precipitation_data(
            "2023-01-01", "2023-01-03", bbox
        )
        
        # Check data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check required columns
        required_cols = ["date", "longitude", "latitude", "precipitation"]
        for col in required_cols:
            assert col in df.columns
        
        # Check precipitation values are non-negative
        assert df["precipitation"].min() >= 0

    def test_fetch_climate_indicators(self, client):
        """Test comprehensive climate indicators fetching."""
        bbox = (-120, 35, -115, 40)
        
        data = client.fetch_climate_indicators(
            "2023-01-01", "2023-01-02", bbox
        )
        
        # Check returned data structure
        assert isinstance(data, dict)
        assert "lst" in data
        assert "precipitation" in data
        assert "vegetation" in data
        
        # Check each dataset
        for key, df in data.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "date" in df.columns
            assert "longitude" in df.columns
            assert "latitude" in df.columns

    def test_data_file_creation(self, client, temp_dir):
        """Test that data files are created correctly."""
        bbox = (-120, 35, -115, 40)
        
        client.fetch_modis_lst_data("2023-01-01", "2023-01-02", bbox)
        
        # Check that file was created
        expected_file = Path(temp_dir) / "MOD11A1_2023-01-01_2023-01-02.csv"
        assert expected_file.exists()
        
        # Check file contents
        df = pd.read_csv(expected_file)
        assert len(df) > 0

    @patch.dict('os.environ', {'EARTHDATA_USERNAME': 'env_user', 'EARTHDATA_PASSWORD': 'env_pass'})
    def test_credentials_from_environment(self, temp_dir):
        """Test loading credentials from environment variables."""
        client = NASAEarthDataClient(data_dir=temp_dir)
        
        assert client.username == "env_user"
        assert client.password == "env_pass"