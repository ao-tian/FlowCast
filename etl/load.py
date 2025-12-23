"""Data loading module for database operations."""
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import DB_URL, PROCESSED_DATA_DIR


def load_to_database(df, table_name, if_exists="replace"):
    """
    Load DataFrame to PostgreSQL database.
    
    Args:
        df: DataFrame to load
        table_name: Target table name
        if_exists: What to do if table exists ('replace', 'append', 'fail')
    """
    try:
        engine = create_engine(DB_URL)
        
        # Ensure date columns are properly formatted
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = df[col].dt.strftime('%Y-%m-%d')
        
        df.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=False,
            method="multi"
        )
        
        print(f"Loaded {len(df)} rows to {table_name}")
        return True
    except Exception as e:
        print(f"Error loading {table_name}: {e}")
        return False


def load_all_data(transformed_data_dict):
    """
    Load all transformed data to database.
    
    Args:
        transformed_data_dict: Dictionary of transformed DataFrames
    """
    print("Loading data to database...")
    
    # Load passenger traffic
    if "passenger_traffic" in transformed_data_dict:
        load_to_database(transformed_data_dict["passenger_traffic"], "passenger_traffic", if_exists="replace")
    
    # Load weather data
    if "weather" in transformed_data_dict:
        load_to_database(transformed_data_dict["weather"], "weather_data", if_exists="replace")
    
    # Load holidays
    if "holidays" in transformed_data_dict:
        load_to_database(transformed_data_dict["holidays"], "holidays", if_exists="replace")
    
    # Load retail sales
    if "retail_sales" in transformed_data_dict:
        load_to_database(transformed_data_dict["retail_sales"], "retail_sales", if_exists="replace")
    
    print("Data loading complete!")


def test_connection():
    """Test database connection."""
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("Database connection successful!")
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    from transform import transform_all_data
    from extract import extract_all_data
    
    if test_connection():
        raw_data = extract_all_data(regenerate=False)
        transformed_data = transform_all_data(raw_data)
        load_all_data(transformed_data)

