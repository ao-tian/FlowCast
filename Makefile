# Makefile for project automation

.PHONY: help install setup etl train dashboard test clean

help:
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies"
	@echo "  make etl        - Run ETL pipeline"
	@echo "  make train      - Train forecasting models"
	@echo "  make dashboard  - Launch Streamlit dashboard"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean generated files"

setup:
	pip install -r requirements.txt

etl:
	python -m etl.run_pipeline

train:
	python -m models.train_models

dashboard:
	streamlit run app/main.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

