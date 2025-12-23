"""Utility functions for the dashboard."""
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import DB_URL, PROCESSED_DATA_DIR, MODEL_DIR


def get_db_connection():
    """Get database connection."""
    try:
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def load_passenger_data():
    """Load passenger traffic data."""
    try:
        engine = get_db_connection()
        if engine is None:
            # Fallback to CSV
            df = pd.read_csv(PROCESSED_DATA_DIR / "merged_dataset.csv")
        else:
            query = """
            SELECT date, SUM(passenger_count) as passenger_count
            FROM passenger_traffic
            GROUP BY date
            ORDER BY date
            """
            df = pd.read_sql(query, engine)
        
        # Always ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # If we have multiple rows per date, aggregate
        if "passenger_count" in df.columns and len(df) > 0:
            df = df.groupby("date")["passenger_count"].sum().reset_index()
        
        return df
    except Exception as e:
        # Fallback to CSV
        try:
            df = pd.read_csv(PROCESSED_DATA_DIR / "merged_dataset.csv")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            if "passenger_count" in df.columns:
                df = df.groupby("date")["passenger_count"].sum().reset_index()
            return df
        except:
            return pd.DataFrame()


def load_forecast_data():
    """Load forecast results."""
    try:
        forecast_path = MODEL_DIR / "forecast_results.csv"
        if forecast_path.exists():
            df = pd.read_csv(forecast_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def load_retail_data():
    """Load retail sales data."""
    try:
        engine = get_db_connection()
        if engine is None:
            # Try loading from retail_sales_clean.csv first (has category info)
            retail_path = PROCESSED_DATA_DIR / "retail_sales_clean.csv"
            if retail_path.exists():
                try:
                    df = pd.read_csv(retail_path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    return df
                except Exception as e:
                    pass
            
            # Fallback to merged dataset
            merged_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
            if merged_path.exists():
                df = pd.read_csv(merged_path)
                if "date" in df.columns and "sales_amount" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    # Create a dummy category column if missing
                    if "category" not in df.columns:
                        df["category"] = "All"
                    return df[["date", "sales_amount", "sales_count", "category"]].dropna(subset=["sales_amount"])
            return pd.DataFrame()
        
        # Try database
        query = """
        SELECT date, SUM(sales_amount) as sales_amount, SUM(sales_count) as sales_count, category
        FROM retail_sales
        GROUP BY date, category
        ORDER BY date
        """
        df = pd.read_sql(query, engine)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        # Final fallback to CSV
        try:
            retail_path = PROCESSED_DATA_DIR / "retail_sales_clean.csv"
            if retail_path.exists():
                df = pd.read_csv(retail_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df
        except:
            pass
        return pd.DataFrame()


def load_model_performance():
    """Load model performance metrics."""
    try:
        engine = get_db_connection()
        if engine is None:
            return pd.DataFrame()
        query = """
        SELECT model_name, metric_name, metric_value, evaluation_date
        FROM model_performance
        ORDER BY evaluation_date DESC, model_name, metric_name
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        return pd.DataFrame()


def load_data_quality_checks():
    """Load data quality check results."""
    try:
        engine = get_db_connection()
        if engine is None:
            # Load from JSON files
            quality_checks = []
            
            # Load validation results from all JSON files
            for json_file in PROCESSED_DATA_DIR.glob("validation_results_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        checks = json.load(f)
                        if isinstance(checks, list):
                            # Extract check type from filename
                            check_type = json_file.stem.replace("validation_results_", "")
                            for check in checks:
                                check["check_type"] = check_type
                                quality_checks.append(check)
                except Exception as e:
                    continue
            
            if quality_checks:
                df = pd.DataFrame(quality_checks)
                # Add check_date if missing (use current timestamp)
                if "check_date" not in df.columns:
                    df["check_date"] = pd.Timestamp.now()
                # Ensure check_date is datetime if it exists
                if "check_date" in df.columns:
                    df["check_date"] = pd.to_datetime(df["check_date"], errors='coerce')
                return df
            return pd.DataFrame()
        
        # Try database
        query = """
        SELECT check_name, check_type, status, message, rows_checked, rows_failed, check_date
        FROM data_quality_checks
        ORDER BY check_date DESC
        LIMIT 100
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        # Final fallback to JSON
        try:
            quality_checks = []
            for json_file in PROCESSED_DATA_DIR.glob("validation_results_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        checks = json.load(f)
                        if isinstance(checks, list):
                            check_type = json_file.stem.replace("validation_results_", "")
                            for check in checks:
                                check["check_type"] = check_type
                                quality_checks.append(check)
                except:
                    continue
            if quality_checks:
                df = pd.DataFrame(quality_checks)
                # Add check_date if missing (use current timestamp)
                if "check_date" not in df.columns:
                    df["check_date"] = pd.Timestamp.now()
                # Ensure check_date is datetime if it exists
                if "check_date" in df.columns:
                    df["check_date"] = pd.to_datetime(df["check_date"], errors='coerce')
                return df
        except:
            pass
        return pd.DataFrame()


def calculate_yoy_growth(df, date_col="date", value_col="passenger_count"):
    """Calculate year-over-year growth."""
    df = df.copy()
    df = df.sort_values(date_col)
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    
    # Group by month-day and calculate YoY
    df["prev_year_value"] = df.groupby(["month", "day"])[value_col].shift(1)
    df["yoy_growth"] = ((df[value_col] - df["prev_year_value"]) / (df["prev_year_value"] + 1)) * 100
    
    return df

