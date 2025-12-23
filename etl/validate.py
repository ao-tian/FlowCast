"""Data validation module using Great Expectations."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, DB_URL
from sqlalchemy import create_engine
import json


def validate_passenger_data(df):
    """Validate passenger traffic data."""
    checks = []
    
    # Check 1: No negative passenger counts
    negative_count = (df["passenger_count"] < 0).sum()
    checks.append({
        "check_name": "no_negative_passenger_counts",
        "status": "PASS" if negative_count == 0 else "FAIL",
        "message": f"Found {negative_count} negative passenger counts",
        "rows_checked": len(df),
        "rows_failed": negative_count
    })
    
    # Check 2: Reasonable range (not too high)
    upper_bound = 200000  # Reasonable max for a single day
    too_high = (df["passenger_count"] > upper_bound).sum()
    checks.append({
        "check_name": "passenger_count_reasonable_range",
        "status": "PASS" if too_high == 0 else "FAIL",
        "message": f"Found {too_high} records with passenger count > {upper_bound}",
        "rows_checked": len(df),
        "rows_failed": too_high
    })
    
    # Check 3: No missing dates
    missing_dates = df["date"].isna().sum()
    checks.append({
        "check_name": "no_missing_dates",
        "status": "PASS" if missing_dates == 0 else "FAIL",
        "message": f"Found {missing_dates} missing dates",
        "rows_checked": len(df),
        "rows_failed": missing_dates
    })
    
    # Check 4: Date range is valid
    date_range_valid = (df["date"].min() < df["date"].max())
    checks.append({
        "check_name": "valid_date_range",
        "status": "PASS" if date_range_valid else "FAIL",
        "message": "Date range is valid" if date_range_valid else "Invalid date range",
        "rows_checked": len(df),
        "rows_failed": 0 if date_range_valid else len(df)
    })
    
    return checks


def validate_weather_data(df):
    """Validate weather data."""
    checks = []
    
    # Check 1: Temperature in reasonable range
    if "temperature_celsius" in df.columns:
        temp_range = ((df["temperature_celsius"] >= -30) & (df["temperature_celsius"] <= 50)).all()
        checks.append({
            "check_name": "temperature_reasonable_range",
            "status": "PASS" if temp_range else "FAIL",
            "message": "Temperature within reasonable range (-30 to 50°C)" if temp_range else "Temperature out of range",
            "rows_checked": len(df),
            "rows_failed": (~((df["temperature_celsius"] >= -30) & (df["temperature_celsius"] <= 50))).sum()
        })
    
    # Check 2: Humidity in 0-100 range
    if "humidity_percent" in df.columns:
        humidity_range = ((df["humidity_percent"] >= 0) & (df["humidity_percent"] <= 100)).all()
        checks.append({
            "check_name": "humidity_reasonable_range",
            "status": "PASS" if humidity_range else "FAIL",
            "message": "Humidity within 0-100%" if humidity_range else "Humidity out of range",
            "rows_checked": len(df),
            "rows_failed": (~((df["humidity_percent"] >= 0) & (df["humidity_percent"] <= 100))).sum()
        })
    
    # Check 3: No negative precipitation
    if "precipitation_mm" in df.columns:
        negative_precip = (df["precipitation_mm"] < 0).sum()
        checks.append({
            "check_name": "no_negative_precipitation",
            "status": "PASS" if negative_precip == 0 else "FAIL",
            "message": f"Found {negative_precip} negative precipitation values",
            "rows_checked": len(df),
            "rows_failed": negative_precip
        })
    
    return checks


def validate_retail_sales_data(df):
    """Validate retail sales data."""
    checks = []
    
    # Check 1: No negative sales
    negative_sales = ((df["sales_amount"] < 0) | (df["sales_count"] < 0)).sum()
    checks.append({
        "check_name": "no_negative_sales",
        "status": "PASS" if negative_sales == 0 else "FAIL",
        "message": f"Found {negative_sales} records with negative sales",
        "rows_checked": len(df),
        "rows_failed": negative_sales
    })
    
    # Check 2: Sales amount and count consistency (rough check)
    if len(df) > 0:
        avg_transaction = df["sales_amount"] / (df["sales_count"] + 1)  # +1 to avoid div by zero
        reasonable_transaction = ((avg_transaction >= 1) & (avg_transaction <= 1000)).all()
        checks.append({
            "check_name": "sales_amount_count_consistency",
            "status": "PASS" if reasonable_transaction else "FAIL",
            "message": "Average transaction value is reasonable (1-1000)" if reasonable_transaction else "Unreasonable average transaction values",
            "rows_checked": len(df),
            "rows_failed": (~reasonable_transaction).sum()
        })
    
    return checks


def calculate_data_quality_score(checks):
    """Calculate overall data quality score."""
    if not checks:
        return 0.0
    
    total_checks = len(checks)
    passed_checks = sum(1 for check in checks if check["status"] == "PASS")
    
    score = (passed_checks / total_checks) * 100
    return round(score, 2)


def save_validation_results(checks, check_type="general"):
    """Save validation results to database."""
    try:
        engine = create_engine(DB_URL)
        
        for check in checks:
            check["check_type"] = check_type
            check_df = pd.DataFrame([check])
            check_df.to_sql(
                "data_quality_checks",
                engine,
                if_exists="append",
                index=False,
                method="multi"
            )
        
        print(f"Saved {len(checks)} validation checks to database")
    except Exception as e:
        print(f"Warning: Could not save validation results to database: {e}")
        # Save to JSON as backup
        output_path = PROCESSED_DATA_DIR / f"validation_results_{check_type}.json"
        with open(output_path, "w") as f:
            json.dump(checks, f, indent=2, default=str)
        print(f"Saved validation results to {output_path}")


def validate_all_data(transformed_data_dict):
    """
    Run all validation checks.
    
    Args:
        transformed_data_dict: Dictionary of transformed DataFrames
        
    Returns:
        Dictionary of validation results
    """
    print("Running validation checks...")
    
    all_checks = []
    
    # Validate passenger data
    if "passenger_traffic" in transformed_data_dict:
        passenger_checks = validate_passenger_data(transformed_data_dict["passenger_traffic"])
        all_checks.extend(passenger_checks)
        save_validation_results(passenger_checks, "passenger_traffic")
    
    # Validate weather data
    if "weather" in transformed_data_dict:
        weather_checks = validate_weather_data(transformed_data_dict["weather"])
        all_checks.extend(weather_checks)
        save_validation_results(weather_checks, "weather")
    
    # Validate retail sales data
    if "retail_sales" in transformed_data_dict:
        retail_checks = validate_retail_sales_data(transformed_data_dict["retail_sales"])
        all_checks.extend(retail_checks)
        save_validation_results(retail_checks, "retail_sales")
    
    # Calculate overall quality score
    quality_score = calculate_data_quality_score(all_checks)
    
    results = {
        "checks": all_checks,
        "quality_score": quality_score,
        "total_checks": len(all_checks),
        "passed_checks": sum(1 for check in all_checks if check["status"] == "PASS"),
        "failed_checks": sum(1 for check in all_checks if check["status"] == "FAIL")
    }
    
    print(f"\nValidation Summary:")
    print(f"  Total checks: {results['total_checks']}")
    print(f"  Passed: {results['passed_checks']}")
    print(f"  Failed: {results['failed_checks']}")
    print(f"  Quality Score: {quality_score}%")
    
    return results


if __name__ == "__main__":
    from transform import transform_all_data
    from extract import extract_all_data
    
    raw_data = extract_all_data(regenerate=False)
    transformed_data = transform_all_data(raw_data)
    validation_results = validate_all_data(transformed_data)

