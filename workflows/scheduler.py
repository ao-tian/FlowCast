"""Prefect workflow scheduler for ETL and model training."""
from prefect import flow, task
from prefect.schedules import CronSchedule
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


@task
def run_etl_pipeline():
    """Task to run ETL pipeline."""
    from etl.run_pipeline import run_etl_pipeline
    print(f"[{datetime.now()}] Running ETL pipeline...")
    run_etl_pipeline(regenerate_data=False)
    print(f"[{datetime.now()}] ETL pipeline completed")


@task
def train_models():
    """Task to train forecasting models."""
    from models.train_models import train_all_models, load_data_from_db
    print(f"[{datetime.now()}] Training models...")
    df = load_data_from_db()
    train_all_models(df, test_size=90)
    print(f"[{datetime.now()}] Model training completed")


@flow(name="Weekly ETL Pipeline")
def weekly_etl_flow():
    """Weekly ETL pipeline workflow."""
    run_etl_pipeline()


@flow(name="Monthly Model Training")
def monthly_model_training_flow():
    """Monthly model training workflow."""
    run_etl_pipeline()  # Refresh data first
    train_models()


if __name__ == "__main__":
    # Schedule workflows
    # Weekly ETL (every Monday at 2 AM)
    weekly_schedule = CronSchedule(cron="0 2 * * 1", timezone="UTC")
    
    # Monthly model training (1st of month at 3 AM)
    monthly_schedule = CronSchedule(cron="0 3 1 * *", timezone="UTC")
    
    print("Workflow scheduler configured:")
    print("- Weekly ETL: Every Monday at 2 AM UTC")
    print("- Monthly Model Training: 1st of each month at 3 AM UTC")
    print("\nTo run workflows, use Prefect Cloud/Server or run manually:")
    print("  python -m workflows.scheduler")
    print("\nOr deploy with:")
    print("  prefect deploy workflows/scheduler.py:weekly_etl_flow")
    print("  prefect deploy workflows/scheduler.py:monthly_model_training_flow")

