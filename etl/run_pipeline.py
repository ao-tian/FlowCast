"""Main ETL pipeline orchestration script."""
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from etl.extract import extract_all_data
from etl.transform import transform_all_data
from etl.validate import validate_all_data
from etl.load import load_all_data, test_connection


def run_etl_pipeline(regenerate_data=True):
    """
    Run the complete ETL pipeline.
    
    Args:
        regenerate_data: If True, regenerate synthetic data
    """
    print("=" * 60)
    print("Starting ETL Pipeline")
    print("=" * 60)
    
    # Step 1: Extract
    print("\n[1/4] EXTRACT: Generating/loading raw data...")
    raw_data = extract_all_data(regenerate=regenerate_data)
    
    # Step 2: Transform
    print("\n[2/4] TRANSFORM: Cleaning and merging data...")
    transformed_data = transform_all_data(raw_data)
    
    # Step 3: Validate
    print("\n[3/4] VALIDATE: Running data quality checks...")
    validation_results = validate_all_data(transformed_data)
    
    # Step 4: Load
    print("\n[4/4] LOAD: Loading data to database...")
    if test_connection():
        load_all_data(transformed_data)
    else:
        print("Warning: Database connection failed. Skipping database load.")
        print("Data saved to processed/ directory instead.")
    
    print("\n" + "=" * 60)
    print("ETL Pipeline Complete!")
    print("=" * 60)
    print(f"Data Quality Score: {validation_results['quality_score']}%")
    print(f"Total Records Processed: {len(transformed_data.get('merged', pd.DataFrame()))}")
    
    return transformed_data, validation_results


if __name__ == "__main__":
    import pandas as pd
    run_etl_pipeline(regenerate_data=True)

