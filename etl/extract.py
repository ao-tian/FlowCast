"""Data extraction module for ingesting raw data."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR


def generate_synthetic_passenger_data(start_date="2020-01-01", end_date="2024-12-31", airport_code="PVG"):
    """
    Generate synthetic passenger traffic data with realistic patterns.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        airport_code: Airport code
        
    Returns:
        DataFrame with passenger traffic data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    
    # Base trend (gradual increase)
    base_trend = np.linspace(50000, 70000, n_days)
    
    # Weekly seasonality (higher on weekends)
    day_of_week = dates.dayofweek
    weekly_pattern = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_week / 7) + 0.1 * (day_of_week >= 5)
    
    # Monthly seasonality (higher in summer, holidays)
    month = dates.month
    monthly_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * (month - 1) / 12) + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)
    
    # Yearly trend with some randomness
    yearly_noise = np.random.normal(1.0, 0.05, n_days)
    
    # Combine patterns
    passenger_counts = (base_trend * weekly_pattern * monthly_pattern * yearly_noise).astype(int)
    
    # Convert to numpy array to ensure proper indexing
    passenger_counts = np.array(passenger_counts)
    
    # Add some outliers (special events)
    outlier_indices = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
    outlier_multiplier = np.random.uniform(1.3, 1.8, len(outlier_indices))
    passenger_counts[outlier_indices] = (passenger_counts[outlier_indices] * outlier_multiplier).astype(int)
    
    # Ensure no negatives
    passenger_counts = np.maximum(passenger_counts, 0)
    
    df = pd.DataFrame({
        "date": dates,
        "passenger_count": passenger_counts,
        "airport_code": airport_code,
        "terminal": np.random.choice(["T1", "T2", "T3"], n_days)
    })
    
    return df


def generate_synthetic_weather_data(start_date="2020-01-01", end_date="2024-12-31"):
    """
    Generate synthetic weather data.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        
    Returns:
        DataFrame with weather data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    
    # Temperature with seasonality
    month = dates.month
    base_temp = 15 + 10 * np.sin(2 * np.pi * (month - 1) / 12 - np.pi / 2)
    temperature = base_temp + np.random.normal(0, 5, n_days)
    
    # Precipitation (more in summer)
    precipitation_chance = 0.2 + 0.3 * (month >= 6) * (month <= 9)
    precipitation = np.where(
        np.random.random(n_days) < precipitation_chance,
        np.random.exponential(5, n_days),
        0
    )
    
    # Humidity (higher in summer)
    humidity = 50 + 20 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 10, n_days)
    humidity = np.clip(humidity, 0, 100)
    
    # Weather condition
    conditions = ["Clear", "Cloudy", "Rain", "Fog"]
    condition_probs = [0.5, 0.3, 0.15, 0.05]
    weather_condition = np.random.choice(conditions, n_days, p=condition_probs)
    
    df = pd.DataFrame({
        "date": dates,
        "temperature_celsius": np.round(temperature, 1),
        "precipitation_mm": np.round(precipitation, 1),
        "humidity_percent": np.round(humidity, 1),
        "weather_condition": weather_condition
    })
    
    return df


def generate_synthetic_holidays(year_range=(2020, 2024)):
    """
    Generate synthetic holiday calendar.
    
    Args:
        year_range: Tuple of (start_year, end_year)
        
    Returns:
        DataFrame with holiday data
    """
    holidays = []
    
    for year in range(year_range[0], year_range[1] + 1):
        # New Year
        holidays.append({"date": f"{year}-01-01", "holiday_name": "New Year's Day", "holiday_type": "National", "is_national": True})
        # Spring Festival (simplified - early February)
        holidays.append({"date": f"{year}-02-10", "holiday_name": "Spring Festival", "holiday_type": "National", "is_national": True})
        holidays.append({"date": f"{year}-02-11", "holiday_name": "Spring Festival", "holiday_type": "National", "is_national": True})
        holidays.append({"date": f"{year}-02-12", "holiday_name": "Spring Festival", "holiday_type": "National", "is_national": True})
        # Labor Day
        holidays.append({"date": f"{year}-05-01", "holiday_name": "Labor Day", "holiday_type": "National", "is_national": True})
        # National Day
        holidays.append({"date": f"{year}-10-01", "holiday_name": "National Day", "holiday_type": "National", "is_national": True})
        holidays.append({"date": f"{year}-10-02", "holiday_name": "National Day", "holiday_type": "National", "is_national": True})
        holidays.append({"date": f"{year}-10-03", "holiday_name": "National Day", "holiday_type": "National", "is_national": True})
        # Mid-Autumn Festival (simplified)
        holidays.append({"date": f"{year}-09-15", "holiday_name": "Mid-Autumn Festival", "holiday_type": "National", "is_national": True})
    
    df = pd.DataFrame(holidays)
    df["date"] = pd.to_datetime(df["date"])
    
    return df


def generate_synthetic_retail_sales(passenger_data, start_date="2020-01-01", end_date="2024-12-31"):
    """
    Generate synthetic retail sales data correlated with passenger volume.
    
    Args:
        passenger_data: DataFrame with passenger traffic
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with retail sales data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Merge with passenger data to get correlation
    passenger_daily = passenger_data.groupby("date")["passenger_count"].sum().reset_index()
    passenger_daily = passenger_daily.set_index("date").reindex(dates, fill_value=0)
    
    # Sales correlated with passenger volume (higher conversion on weekends/holidays)
    day_of_week = dates.dayofweek
    weekend_multiplier = 1.2 * (day_of_week >= 5) + 1.0
    
    # Base conversion rate (percentage of passengers making purchases)
    base_conversion = 0.15
    conversion_rate = base_conversion * weekend_multiplier * np.random.uniform(0.8, 1.2, len(dates))
    
    # Average transaction value
    avg_transaction = np.random.uniform(50, 150, len(dates))
    
    # Calculate sales
    sales_count = (passenger_daily["passenger_count"].values * conversion_rate).astype(int)
    sales_amount = sales_count * avg_transaction
    
    # Add some noise
    sales_amount = sales_amount * np.random.uniform(0.9, 1.1, len(dates))
    
    categories = ["Duty Free", "Food & Beverage", "Retail", "Services"]
    terminals = ["T1", "T2", "T3"]
    
    records = []
    for i, date in enumerate(dates):
        # Distribute sales across categories and terminals
        total_sales = sales_amount[i]
        total_count = sales_count[i]
        
        for category in categories:
            for terminal in terminals:
                share = np.random.uniform(0.15, 0.35)
                records.append({
                    "date": date,
                    "sales_amount": np.round(total_sales * share / len(terminals), 2),
                    "sales_count": int(total_count * share / len(terminals)),
                    "category": category,
                    "terminal": terminal
                })
    
    df = pd.DataFrame(records)
    return df


def extract_all_data(regenerate=True):
    """
    Extract/generate all data sources.
    
    Args:
        regenerate: If True, regenerate synthetic data
        
    Returns:
        Dictionary of DataFrames
    """
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    
    if regenerate:
        print("Generating synthetic passenger data...")
        passenger_df = generate_synthetic_passenger_data(start_date, end_date)
        passenger_df.to_csv(RAW_DATA_DIR / "passenger_traffic.csv", index=False)
        
        print("Generating synthetic weather data...")
        weather_df = generate_synthetic_weather_data(start_date, end_date)
        weather_df.to_csv(RAW_DATA_DIR / "weather_data.csv", index=False)
        
        print("Generating synthetic holidays...")
        holidays_df = generate_synthetic_holidays((2020, 2024))
        holidays_df.to_csv(RAW_DATA_DIR / "holidays.csv", index=False)
        
        print("Generating synthetic retail sales...")
        retail_df = generate_synthetic_retail_sales(passenger_df, start_date, end_date)
        retail_df.to_csv(RAW_DATA_DIR / "retail_sales.csv", index=False)
        
        print("Data extraction complete!")
    else:
        # Load existing data
        passenger_df = pd.read_csv(RAW_DATA_DIR / "passenger_traffic.csv")
        weather_df = pd.read_csv(RAW_DATA_DIR / "weather_data.csv")
        holidays_df = pd.read_csv(RAW_DATA_DIR / "holidays.csv")
        retail_df = pd.read_csv(RAW_DATA_DIR / "retail_sales.csv")
        
        # Convert date columns
        for df in [passenger_df, weather_df, holidays_df, retail_df]:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
    
    return {
        "passenger_traffic": passenger_df,
        "weather": weather_df,
        "holidays": holidays_df,
        "retail_sales": retail_df
    }


if __name__ == "__main__":
    extract_all_data(regenerate=True)

