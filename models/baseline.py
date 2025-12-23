"""Baseline forecasting models."""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def seasonal_naive_forecast(ts, seasonal_period=365, horizon=30):
    """
    Seasonal naive forecast: use value from same period last year.
    
    Args:
        ts: Time series array
        seasonal_period: Seasonal period (default: 365 days)
        horizon: Forecast horizon
        
    Returns:
        Forecast array
    """
    if len(ts) < seasonal_period:
        # If not enough data, use simple average
        return np.full(horizon, ts.mean())
    
    # Use values from same period last year
    forecast = []
    for i in range(horizon):
        idx = len(ts) - seasonal_period + (i % seasonal_period)
        forecast.append(ts[idx])
    
    return np.array(forecast)


def simple_naive_forecast(ts, horizon=30):
    """
    Simple naive forecast: use last observed value.
    
    Args:
        ts: Time series array
        horizon: Forecast horizon
        
    Returns:
        Forecast array
    """
    last_value = ts[-1] if len(ts) > 0 else 0
    return np.full(horizon, last_value)


def moving_average_forecast(ts, window=30, horizon=30):
    """
    Moving average forecast: use average of last N values.
    
    Args:
        ts: Time series array
        window: Window size for moving average
        horizon: Forecast horizon
        
    Returns:
        Forecast array
    """
    if len(ts) < window:
        window = len(ts)
    
    avg_value = np.mean(ts[-window:])
    return np.full(horizon, avg_value)


def evaluate_baseline(actual, forecast, model_name="Baseline"):
    """
    Evaluate baseline model performance.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        model_name: Name of the model
        
    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / (actual + 1))) * 100  # +1 to avoid div by zero
    
    return {
        "model_name": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2)
    }

