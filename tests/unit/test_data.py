"""Unit tests for data ingestion module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.ingestion import NOAAClimateDataClient


class TestNOAAClimateDataClient:
    """Test NOAA Climate Data client."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def client(self, temp_dir):
        """Create client instance for testing."""
        return NOAAClimateDataClient(data_dir=temp_dir)

    def test_init(self, temp_dir):
        """Test client initialization."""
        client = NOAAClimateDataClient(data_dir=temp_dir)

        assert client.data_dir == Path(temp_dir)
        assert client.data_dir.exists()
        assert client.base_url == "https://www.ncei.noaa.gov/cdo-web/api/v2"

    def test_init_with_api_token(self, temp_dir):
        """Test client initialization with API token."""
        client = NOAAClimateDataClient(
            data_dir=temp_dir, api_token="test_token"
        )  # noqa: E501

        assert "token" in client.headers
        assert client.headers["token"] == "test_token"

    def test_fetch_temperature_data(self, client):
        """Test temperature data fetching (with fallback)."""
        bbox = (-120, 35, -115, 40)

        df = client.fetch_temperature_data("2023-01-01", "2023-01-03", bbox)

        # Check data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check required columns
        required_cols = ["date", "longitude", "latitude", "temperature"]
        for col in required_cols:
            assert col in df.columns

        # Check data ranges
        assert df["longitude"].min() >= bbox[0]
        assert df["longitude"].max() <= bbox[2]
        assert df["latitude"].min() >= bbox[1]
        assert df["latitude"].max() <= bbox[3]

        # Check temperature values are reasonable (in Celsius)
        assert df["temperature"].min() > -50  # Reasonable minimum
        assert df["temperature"].max() < 60  # Reasonable maximum

    def test_fetch_climate_data(self, client):
        """Test comprehensive climate data fetching."""
        bbox = (-120, 35, -115, 40)

        data = client.fetch_climate_data("2023-01-01", "2023-01-02", bbox)

        # Check returned data structure
        assert isinstance(data, dict)
        assert "temperature" in data
        assert "precipitation" in data

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

        client.fetch_temperature_data("2023-01-01", "2023-01-02", bbox)

        # Check that file was created
        expected_file = (
            Path(temp_dir) / "noaa_temperature_2023-01-01_2023-01-02.csv"
        )  # noqa: E501
        assert expected_file.exists()

        # Check file contents
        df = pd.read_csv(expected_file, parse_dates=["date"])
        assert len(df) > 0

    def test_fallback_data_generation(self, client):
        """Test fallback data generation when API fails."""
        bbox = (-118, 34, -116, 36)

        # Test the fallback method directly
        df = client._generate_fallback_data("2023-01-01", "2023-01-03", bbox)

        # Check data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check required columns
        required_cols = ["date", "longitude", "latitude", "temperature"]
        for col in required_cols:
            assert col in df.columns

        # Check that data follows realistic patterns
        # Should have seasonal variation (January should be cooler in Northern Hemisphere)  # noqa: E501
        temps = df["temperature"].values
        assert len(set(temps)) > 1  # Should have variation

        # Check geographic bounds
        assert df["longitude"].min() >= bbox[0]
        assert df["longitude"].max() <= bbox[2]
        assert df["latitude"].min() >= bbox[1]
        assert df["latitude"].max() <= bbox[3]

    @patch("src.data.ingestion.requests.get")
    def test_api_request_error_handling(self, mock_get, client):
        """Test API error handling."""
        # Mock a failed request
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        # Should handle error gracefully
        result = client._make_request("test", {})
        assert result == {"results": []}

    def test_get_stations_fallback(self, client):
        """Test station fallback when API fails."""
        bbox = (-120, 35, -115, 40)

        stations = client.get_stations_in_bbox(bbox)

        # Should return fallback stations
        assert isinstance(stations, list)
        assert len(stations) > 0

        # Should be valid station IDs
        for station in stations:
            assert station.startswith("GHCND:")
