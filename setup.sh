#!/bin/bash

# Setup script for Airport Operations & Retail Demand Intelligence Platform

echo "=========================================="
echo "Setting up FlowCast Project"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=airport_db
DB_USER=airport_user
DB_PASSWORD=airport_password

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO

# Data Paths
RAW_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
MODEL_PATH=./data/models
OUTPUT_PATH=./data/outputs
EOF
    echo ".env file created. Please review and update if needed."
else
    echo ".env file already exists. Skipping creation."
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw data/processed data/models data/outputs

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install setuptools (if needed): pip install setuptools wheel"
echo "2. Run ETL pipeline: python -m etl.run_pipeline"
echo "3. Train models: python -m models.train_models"
echo "4. Launch dashboard: streamlit run app/main.py"
echo ""

