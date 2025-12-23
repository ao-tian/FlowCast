"""Data transformation and cleaning module."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR


def clean_passenger_data(df):
    """Clean and standardize passenger traffic data."""
    df = df.copy()
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["date", "airport_code", "terminal"], keep="last")
    
    # Handle missing values
    df["passenger_count"] = df["passenger_count"].fillna(0)
    df["airport_code"] = df["airport_code"].fillna("UNKNOWN")
    df["terminal"] = df["terminal"].fillna("UNKNOWN")
    
    # Remove negative passenger counts
    df = df[df["passenger_count"] >= 0]
    
    # Cap outliers (values > 3 standard deviations)
    mean = df["passenger_count"].mean()
    std = df["passenger_count"].std()
    upper_bound = mean + 3 * std
    df.loc[df["passenger_count"] > upper_bound, "passenger_count"] = upper_bound
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def clean_weather_data(df):
    """Clean and standardize weather data."""
    df = df.copy()
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["date"], keep="last")
    
    # Handle missing values with forward fill
    numeric_cols = ["temperature_celsius", "precipitation_mm", "humidity_percent"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Fill categorical with mode
    if "weather_condition" in df.columns:
        df["weather_condition"] = df["weather_condition"].fillna("Unknown")
    
    # Ensure reasonable ranges
    if "humidity_percent" in df.columns:
        df["humidity_percent"] = df["humidity_percent"].clip(0, 100)
    
    if "precipitation_mm" in df.columns:
        df["precipitation_mm"] = df["precipitation_mm"].clip(0, None)
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def clean_holidays_data(df):
    """Clean and standardize holidays data."""
    df = df.copy()
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["date", "holiday_name"], keep="last")
    
    # Fill missing values
    df["holiday_name"] = df["holiday_name"].fillna("Unknown")
    df["holiday_type"] = df["holiday_type"].fillna("Other")
    df["is_national"] = df["is_national"].fillna(False)
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def clean_retail_sales_data(df):
    """Clean and standardize retail sales data."""
    df = df.copy()
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Remove duplicates (keep last)
    df = df.drop_duplicates(keep="last")
    
    # Handle missing values
    df["sales_amount"] = df["sales_amount"].fillna(0)
    df["sales_count"] = df["sales_count"].fillna(0)
    df["category"] = df["category"].fillna("Unknown")
    df["terminal"] = df["terminal"].fillna("UNKNOWN")
    
    # Remove negative values
    df = df[(df["sales_amount"] >= 0) & (df["sales_count"] >= 0)]
    
    # Cap extreme outliers
    if "sales_amount" in df.columns:
        q99 = df["sales_amount"].quantile(0.99)
        df.loc[df["sales_amount"] > q99, "sales_amount"] = q99
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def merge_datasets(passenger_df, weather_df, holidays_df, retail_df):
    """
    Merge all datasets into a unified dataset for modeling.
    
    Args:
        passenger_df: Passenger traffic DataFrame
        weather_df: Weather DataFrame
        holidays_df: Holidays DataFrame
        retail_df: Retail sales DataFrame
        
    Returns:
        Merged DataFrame
    """
    # Aggregate passenger data by date
    passenger_daily = passenger_df.groupby("date").agg({
        "passenger_count": "sum"
    }).reset_index()
    
    # Aggregate retail sales by date
    retail_daily = retail_df.groupby("date").agg({
        "sales_amount": "sum",
        "sales_count": "sum"
    }).reset_index()
    
    # Create holiday indicator
    holidays_df["is_holiday"] = True
    holiday_dates = holidays_df[["date", "is_holiday"]].drop_duplicates("date")
    
    # Get all unique dates
    all_dates = pd.date_range(
        start=min(passenger_df["date"].min(), weather_df["date"].min()),
        end=max(passenger_df["date"].max(), weather_df["date"].max()),
        freq="D"
    )
    
    # Create base DataFrame
    merged_df = pd.DataFrame({"date": all_dates})
    
    # Merge all datasets
    merged_df = merged_df.merge(passenger_daily, on="date", how="left")
    merged_df = merged_df.merge(weather_df, on="date", how="left")
    merged_df = merged_df.merge(holiday_dates, on="date", how="left")
    merged_df = merged_df.merge(retail_daily, on="date", how="left")
    
    # Fill missing values
    merged_df["is_holiday"] = merged_df["is_holiday"].fillna(False)
    merged_df["passenger_count"] = merged_df["passenger_count"].fillna(0)
    merged_df["sales_amount"] = merged_df["sales_amount"].fillna(0)
    merged_df["sales_count"] = merged_df["sales_count"].fillna(0)
    
    # Add time-based features
    merged_df["year"] = merged_df["date"].dt.year
    merged_df["month"] = merged_df["date"].dt.month
    merged_df["day_of_week"] = merged_df["date"].dt.dayofweek
    merged_df["day_of_year"] = merged_df["date"].dt.dayofyear
    merged_df["is_weekend"] = merged_df["day_of_week"] >= 5
    
    # Sort by date
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    
    return merged_df


def transform_all_data(raw_data_dict):
    """
    Transform all raw data.
    
    Args:
        raw_data_dict: Dictionary of raw DataFrames
        
    Returns:
        Dictionary of cleaned DataFrames and merged dataset
    """
    print("Cleaning passenger data...")
    passenger_clean = clean_passenger_data(raw_data_dict["passenger_traffic"])
    passenger_clean.to_csv(PROCESSED_DATA_DIR / "passenger_traffic_clean.csv", index=False)
    
    print("Cleaning weather data...")
    weather_clean = clean_weather_data(raw_data_dict["weather"])
    weather_clean.to_csv(PROCESSED_DATA_DIR / "weather_data_clean.csv", index=False)
    
    print("Cleaning holidays data...")
    holidays_clean = clean_holidays_data(raw_data_dict["holidays"])
    holidays_clean.to_csv(PROCESSED_DATA_DIR / "holidays_clean.csv", index=False)
    
    print("Cleaning retail sales data...")
    retail_clean = clean_retail_sales_data(raw_data_dict["retail_sales"])
    retail_clean.to_csv(PROCESSED_DATA_DIR / "retail_sales_clean.csv", index=False)
    
    print("Merging datasets...")
    merged_df = merge_datasets(passenger_clean, weather_clean, holidays_clean, retail_clean)
    merged_df.to_csv(PROCESSED_DATA_DIR / "merged_dataset.csv", index=False)
    
    print("Data transformation complete!")
    
    return {
        "passenger_traffic": passenger_clean,
        "weather": weather_clean,
        "holidays": holidays_clean,
        "retail_sales": retail_clean,
        "merged": merged_df
    }


if __name__ == "__main__":
    from extract import extract_all_data
    
    raw_data = extract_all_data(regenerate=False)
    transformed_data = transform_all_data(raw_data)

