"""Ensemble forecasting methods."""
import numpy as np
import pandas as pd


def weighted_ensemble(forecasts_dict, weights=None):
    """
    Create weighted ensemble of forecasts.
    
    Args:
        forecasts_dict: Dictionary of {model_name: forecast_array}
        weights: Dictionary of weights (default: equal weights)
        
    Returns:
        Ensemble forecast array
    """
    if weights is None:
        # Equal weights
        n_models = len(forecasts_dict)
        weights = {model: 1.0 / n_models for model in forecasts_dict.keys()}
    
    # Ensure all forecasts have same length
    forecast_lengths = [len(f) for f in forecasts_dict.values()]
    if len(set(forecast_lengths)) > 1:
        min_length = min(forecast_lengths)
        forecasts_dict = {k: v[:min_length] for k, v in forecasts_dict.items()}
    
    # Calculate weighted average
    ensemble_forecast = np.zeros(len(list(forecasts_dict.values())[0]))
    
    for model_name, forecast in forecasts_dict.items():
        weight = weights.get(model_name, 0.0)
        ensemble_forecast += weight * forecast
    
    return ensemble_forecast


def inverse_error_weighted_ensemble(forecasts_dict, errors_dict):
    """
    Create ensemble weighted by inverse of errors (lower error = higher weight).
    
    Args:
        forecasts_dict: Dictionary of {model_name: forecast_array}
        errors_dict: Dictionary of {model_name: error_value}
        
    Returns:
        Ensemble forecast array
    """
    # Calculate inverse errors (with small epsilon to avoid division by zero)
    epsilon = 1e-6
    inv_errors = {model: 1.0 / (error + epsilon) for model, error in errors_dict.items()}
    
    # Normalize to sum to 1
    total_inv_error = sum(inv_errors.values())
    weights = {model: inv_error / total_inv_error for model, inv_error in inv_errors.items()}
    
    return weighted_ensemble(forecasts_dict, weights), weights


def simple_average_ensemble(forecasts_dict):
    """
    Simple average ensemble.
    
    Args:
        forecasts_dict: Dictionary of {model_name: forecast_array}
        
    Returns:
        Ensemble forecast array
    """
    return weighted_ensemble(forecasts_dict, weights=None)


def median_ensemble(forecasts_dict):
    """
    Median ensemble (more robust to outliers).
    
    Args:
        forecasts_dict: Dictionary of {model_name: forecast_array}
        
    Returns:
        Ensemble forecast array
    """
    # Ensure all forecasts have same length
    forecast_lengths = [len(f) for f in forecasts_dict.values()]
    if len(set(forecast_lengths)) > 1:
        min_length = min(forecast_lengths)
        forecasts_dict = {k: v[:min_length] for k, v in forecasts_dict.items()}
    
    # Stack forecasts and take median
    forecasts_array = np.array(list(forecasts_dict.values()))
    ensemble_forecast = np.median(forecasts_array, axis=0)
    
    return ensemble_forecast

