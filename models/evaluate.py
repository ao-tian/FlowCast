"""Model evaluation utilities."""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Dict, Tuple


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1))) * 100
    
    # Additional metrics
    mpe = np.mean((y_true - y_pred) / (np.abs(y_true) + 1)) * 100  # Mean percentage error
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "mpe": round(mpe, 2),
        "correlation": round(corr, 4)
    }


def rolling_time_series_cv(df, target_col, train_size=365, test_size=30, step_size=30, 
                          model_func=None, model_params=None):
    """
    Perform rolling time series cross-validation.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        train_size: Size of training window
        test_size: Size of test window
        step_size: Step size for rolling window
        model_func: Function to train model
        model_params: Parameters for model function
        
    Returns:
        List of evaluation results
    """
    results = []
    df = df.sort_values("date").reset_index(drop=True)
    
    start_idx = 0
    end_idx = len(df) - test_size
    
    while start_idx + train_size + test_size <= len(df):
        # Split data
        train_end = start_idx + train_size
        test_end = train_end + test_size
        
        train_data = df.iloc[start_idx:train_end]
        test_data = df.iloc[train_end:test_end]
        
        # Train model (placeholder - should be replaced with actual model training)
        if model_func is not None:
            # This is a placeholder - actual implementation depends on model type
            pass
        
        # Move window
        start_idx += step_size
    
    return results


def backtest_forecast(forecast_df, actual_df, date_col="date", forecast_col="forecast", actual_col="passenger_count"):
    """
    Perform backtesting of forecasts.
    
    Args:
        forecast_df: DataFrame with forecasts
        actual_df: DataFrame with actual values
        date_col: Name of date column
        forecast_col: Name of forecast column
        actual_col: Name of actual values column
        
    Returns:
        Evaluation metrics
    """
    # Merge forecasts with actuals
    merged = forecast_df.merge(
        actual_df[[date_col, actual_col]],
        on=date_col,
        how="inner"
    )
    
    if len(merged) == 0:
        return None
    
    y_true = merged[actual_col].values
    y_pred = merged[forecast_col].values
    
    metrics = calculate_metrics(y_true, y_pred)
    
    return metrics


def compare_models(model_results: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        model_results: List of dictionaries with model results
        
    Returns:
        DataFrame comparing models
    """
    comparison_df = pd.DataFrame(model_results)
    comparison_df = comparison_df.sort_values("mae")
    return comparison_df


if __name__ == "__main__":
    # Example usage
    pass

