"""Configuration management for the application."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "airport_db"),
    "user": os.getenv("DB_USER", "airport_user"),
    "password": os.getenv("DB_PASSWORD", "airport_password"),
}

# Database connection string
DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Model configuration
MODEL_CONFIG = {
    "forecast_horizon": 90,  # days
    "train_test_split": 0.8,
    "validation_window": 30,  # days for rolling validation
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "lag_features": [1, 7, 30, 90],  # days
    "rolling_windows": [7, 30, 90],  # days
}

