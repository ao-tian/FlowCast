"""Main script for training all forecasting models."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from config import DB_URL, MODEL_DIR, MODEL_CONFIG, PROCESSED_DATA_DIR
from sqlalchemy import create_engine
from models.baseline import seasonal_naive_forecast, evaluate_baseline
from models.arima import auto_arima_forecast
from models.ml_models import prepare_ml_features, train_xgboost_model, forecast_xgboost, evaluate_ml_model
from models.ensemble import weighted_ensemble, inverse_error_weighted_ensemble
from models.evaluate import calculate_metrics


def load_data_from_db():
    """Load merged dataset from database."""
    try:
        engine = create_engine(DB_URL)
        query = """
        SELECT p.date, p.passenger_count, 
               w.temperature_celsius, w.precipitation_mm, w.humidity_percent,
               CASE WHEN h.date IS NOT NULL THEN 1 ELSE 0 END as is_holiday,
               r.sales_amount, r.sales_count
        FROM passenger_traffic p
        LEFT JOIN weather_data w ON p.date = w.date
        LEFT JOIN holidays h ON p.date = h.date AND h.is_national = true
        LEFT JOIN (
            SELECT date, SUM(sales_amount) as sales_amount, SUM(sales_count) as sales_count
            FROM retail_sales
            GROUP BY date
        ) r ON p.date = r.date
        ORDER BY p.date
        """
        df = pd.read_sql(query, engine)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"Error loading from database: {e}")
        print("Loading from CSV instead...")
        df = pd.read_csv(PROCESSED_DATA_DIR / "merged_dataset.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df


def prepare_data(df):
    """Prepare data for modeling."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Aggregate passenger count by date (in case of multiple terminals)
    if "passenger_count" in df.columns:
        passenger_daily = df.groupby("date")["passenger_count"].sum().reset_index()
        df = df.merge(passenger_daily, on="date", suffixes=("", "_daily"))
        df["passenger_count"] = df["passenger_count_daily"]
        df = df.drop_duplicates("date").reset_index(drop=True)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df


def train_all_models(df, test_size=90):
    """
    Train all forecasting models.
    
    Args:
        df: Input DataFrame
        test_size: Size of test set (days)
        
    Returns:
        Dictionary of trained models and forecasts
    """
    print("=" * 60)
    print("Training Forecasting Models")
    print("=" * 60)
    
    # Prepare data
    df = prepare_data(df)
    
    # Split into train and test
    split_idx = len(df) - test_size
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    target_col = "passenger_count"
    horizon = len(test_df)
    
    print(f"\nTraining period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Horizon: {horizon} days")
    
    results = {}
    
    # 1. Baseline: Seasonal Naive
    print("\n[1/4] Training Seasonal Naive Baseline...")
    train_ts = train_df[target_col].values
    test_ts = test_df[target_col].values
    
    baseline_forecast = seasonal_naive_forecast(train_ts, seasonal_period=365, horizon=horizon)
    baseline_metrics = evaluate_baseline(test_ts, baseline_forecast, "Seasonal Naive")
    results["seasonal_naive"] = {
        "forecast": baseline_forecast,
        "metrics": baseline_metrics,
        "model": None
    }
    print(f"  MAE: {baseline_metrics['mae']}, RMSE: {baseline_metrics['rmse']}, MAPE: {baseline_metrics['mape']:.2f}%")
    
    # 2. ARIMA/SARIMA
    print("\n[2/4] Training ARIMA/SARIMA...")
    try:
        arima_result = auto_arima_forecast(train_ts, horizon=horizon, seasonal=True)
        arima_forecast = arima_result["forecast"]
        arima_metrics = evaluate_baseline(test_ts, arima_forecast, "SARIMA")
        results["sarima"] = {
            "forecast": arima_forecast,
            "lower_bound": arima_result.get("lower_bound"),
            "upper_bound": arima_result.get("upper_bound"),
            "metrics": arima_metrics,
            "model": arima_result["model"],
            "order": arima_result.get("order"),
            "seasonal_order": arima_result.get("seasonal_order")
        }
        print(f"  MAE: {arima_metrics['mae']}, RMSE: {arima_metrics['rmse']}, MAPE: {arima_metrics['mape']:.2f}%")
        print(f"  Order: {arima_result.get('order')}, Seasonal: {arima_result.get('seasonal_order')}")
    except Exception as e:
        print(f"  Error training ARIMA: {e}")
        results["sarima"] = None
    
    # 3. XGBoost
    print("\n[3/4] Training XGBoost...")
    try:
        # Prepare features
        X_train, y_train, feature_cols = prepare_ml_features(
            train_df, target_col=target_col, date_col="date"
        )
        X_test, y_test, _ = prepare_ml_features(
            test_df, target_col=target_col, date_col="date"
        )
        
        # Ensure feature alignment
        missing_cols = set(feature_cols) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[feature_cols]
        
        # Train model
        xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test)
        xgb_forecast = forecast_xgboost(xgb_model, X_test, feature_cols)
        xgb_metrics = evaluate_ml_model(test_ts, xgb_forecast, "XGBoost")
        results["xgboost"] = {
            "forecast": xgb_forecast,
            "metrics": xgb_metrics,
            "model": xgb_model,
            "feature_cols": feature_cols
        }
        print(f"  MAE: {xgb_metrics['mae']}, RMSE: {xgb_metrics['rmse']}, MAPE: {xgb_metrics['mape']:.2f}%")
        
        # Save model
        model_path = MODEL_DIR / "xgboost_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": xgb_model, "feature_cols": feature_cols}, f)
        print(f"  Model saved to {model_path}")
    except Exception as e:
        print(f"  Error training XGBoost: {e}")
        import traceback
        traceback.print_exc()
        results["xgboost"] = None
    
    # 4. Ensemble
    print("\n[4/4] Creating Ensemble...")
    try:
        forecasts_dict = {}
        errors_dict = {}
        
        if results.get("sarima") is not None:
            forecasts_dict["SARIMA"] = results["sarima"]["forecast"]
            errors_dict["SARIMA"] = results["sarima"]["metrics"]["mae"]
        
        if results.get("xgboost") is not None:
            forecasts_dict["XGBoost"] = results["xgboost"]["forecast"]
            errors_dict["XGBoost"] = results["xgboost"]["metrics"]["mae"]
        
        forecasts_dict["Seasonal Naive"] = results["seasonal_naive"]["forecast"]
        errors_dict["Seasonal Naive"] = results["seasonal_naive"]["metrics"]["mae"]
        
        if len(forecasts_dict) > 1:
            ensemble_forecast, ensemble_weights = inverse_error_weighted_ensemble(forecasts_dict, errors_dict)
            ensemble_metrics = evaluate_baseline(test_ts, ensemble_forecast, "Ensemble")
            results["ensemble"] = {
                "forecast": ensemble_forecast,
                "metrics": ensemble_metrics,
                "weights": ensemble_weights
            }
            print(f"  MAE: {ensemble_metrics['mae']}, RMSE: {ensemble_metrics['rmse']}, MAPE: {ensemble_metrics['mape']:.2f}%")
            print(f"  Weights: {ensemble_weights}")
        else:
            results["ensemble"] = None
    except Exception as e:
        print(f"  Error creating ensemble: {e}")
        results["ensemble"] = None
    
    # Save results
    results_df = pd.DataFrame({
        "date": test_df["date"].values,
        "actual": test_ts,
        **{f"{k}_forecast": v["forecast"] for k, v in results.items() if v is not None and "forecast" in v}
    })
    results_df.to_csv(MODEL_DIR / "forecast_results.csv", index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    metrics_list = []
    for model_name, result in results.items():
        if result is not None and "metrics" in result:
            metrics_list.append(result["metrics"])
    
    if metrics_list:
        comparison_df = pd.DataFrame(metrics_list)
        comparison_df = comparison_df.sort_values("mae")
        print(comparison_df.to_string(index=False))
    
    return results, test_df


def save_model_performance(results):
    """Save model performance metrics to database."""
    try:
        engine = create_engine(DB_URL)
        metrics_records = []
        
        for model_name, result in results.items():
            if result is not None and "metrics" in result:
                metrics = result["metrics"]
                for metric_name, metric_value in metrics.items():
                    if metric_name != "model_name":
                        metrics_records.append({
                            "model_name": metrics["model_name"],
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                            "evaluation_date": datetime.now().date()
                        })
        
        if metrics_records:
            metrics_df = pd.DataFrame(metrics_records)
            metrics_df.to_sql(
                "model_performance",
                engine,
                if_exists="append",
                index=False
            )
            print(f"Saved {len(metrics_records)} performance metrics to database")
    except Exception as e:
        print(f"Warning: Could not save performance metrics to database: {e}")


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = load_data_from_db()
    print(f"Loaded {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    # Train models
    results, test_df = train_all_models(df, test_size=90)
    
    # Save performance metrics
    save_model_performance(results)
    
    print("\nModel training complete!")

