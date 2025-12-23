"""ARIMA/SARIMA time series forecasting models."""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


def fit_arima_model(ts, order=(1, 1, 1)):
    """
    Fit ARIMA model.
    
    Args:
        ts: Time series array or Series
        order: (p, d, q) order tuple
        
    Returns:
        Fitted ARIMA model
    """
    try:
        model = ARIMA(ts, order=order)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None


def fit_sarima_model(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Fit SARIMA model with seasonality.
    
    Args:
        ts: Time series array or Series
        order: (p, d, q) order tuple
        seasonal_order: (P, D, Q, s) seasonal order tuple
        
    Returns:
        Fitted SARIMA model
    """
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        return fitted_model
    except Exception as e:
        print(f"Error fitting SARIMA model: {e}")
        return None


def auto_arima_forecast(ts, horizon=30, max_p=3, max_d=2, max_q=3, seasonal=True):
    """
    Automatically select and fit ARIMA/SARIMA model.
    
    Args:
        ts: Time series array or Series
        horizon: Forecast horizon
        max_p, max_d, max_q: Maximum orders to try
        seasonal: Whether to try seasonal models
        
    Returns:
        Forecast array and fitted model
    """
    ts = pd.Series(ts) if not isinstance(ts, pd.Series) else ts
    
    best_aic = np.inf
    best_model = None
    best_order = None
    best_seasonal_order = None
    
    # Try different ARIMA orders
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    if seasonal and len(ts) > 24:
                        # Try SARIMA with monthly seasonality
                        seasonal_order = (1, 1, 1, 12)
                        model = SARIMAX(ts, order=(p, d, q), seasonal_order=seasonal_order, 
                                       enforce_stationarity=False, enforce_invertibility=False)
                        fitted = model.fit(disp=False)
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                            best_seasonal_order = seasonal_order
                    else:
                        # Try ARIMA
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                            best_seasonal_order = None
                except:
                    continue
    
    if best_model is None:
        # Fallback to simple model
        best_order = (1, 1, 1)
        model = ARIMA(ts, order=best_order)
        best_model = model.fit()
    
    # Generate forecast
    forecast = best_model.forecast(steps=horizon)
    forecast_conf_int = best_model.get_forecast(steps=horizon).conf_int()
    
    return {
        "forecast": forecast.values,
        "lower_bound": forecast_conf_int.iloc[:, 0].values,
        "upper_bound": forecast_conf_int.iloc[:, 1].values,
        "model": best_model,
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "aic": best_aic
    }


def decompose_time_series(ts, period=365):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        ts: Time series array or Series
        period: Seasonal period
        
    Returns:
        Decomposition object
    """
    ts = pd.Series(ts) if not isinstance(ts, pd.Series) else ts
    
    if len(ts) < 2 * period:
        period = len(ts) // 2
    
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        return decomposition
    except Exception as e:
        print(f"Error in decomposition: {e}")
        return None

