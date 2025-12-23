"""Tests for ETL pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from etl.extract import generate_synthetic_passenger_data
from etl.transform import clean_passenger_data


def test_generate_passenger_data():
    """Test passenger data generation."""
    df = generate_synthetic_passenger_data("2020-01-01", "2020-12-31")
    
    assert len(df) > 0
    assert "date" in df.columns
    assert "passenger_count" in df.columns
    assert df["passenger_count"].min() >= 0
    assert df["date"].min() == pd.to_datetime("2020-01-01")


def test_clean_passenger_data():
    """Test passenger data cleaning."""
    # Create test data with issues
    test_data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10),
        "passenger_count": [1000, -100, 2000, None, 1500, 3000, 5000, 10000, 8000, 9000],
        "airport_code": ["PVG"] * 10,
        "terminal": ["T1"] * 10
    })
    
    cleaned = clean_passenger_data(test_data)
    
    # Should remove negatives
    assert (cleaned["passenger_count"] >= 0).all()
    
    # Should handle missing values
    assert cleaned["passenger_count"].notna().all()
    
    # Should have same or fewer rows (duplicates removed)
    assert len(cleaned) <= len(test_data)

