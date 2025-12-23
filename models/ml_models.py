"""Machine learning models for time series forecasting."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import FEATURE_CONFIG


def create_lag_features(df, target_col, lags=[1, 7, 30, 90]):
    """
    Create lag features for time series.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df, target_col, windows=[7, 30, 90]):
    """
    Create rolling window features.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
        df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window=window, min_periods=1).max()
        df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window=window, min_periods=1).min()
    
    return df


def create_time_features(df, date_col="date"):
    """
    Create time-based features.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        DataFrame with time features added
    """
    df = df.copy()
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day_of_week"] = df[date_col].dt.dayofweek
        df["day_of_year"] = df[date_col].dt.dayofyear
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["quarter"] = df[date_col].dt.quarter
        
        # Cyclical encoding for month and day of week
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    return df


def prepare_ml_features(df, target_col="passenger_count", date_col="date", 
                       include_lags=True, include_rolling=True, include_time=True):
    """
    Prepare feature matrix for ML models.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        date_col: Name of date column
        include_lags: Whether to include lag features
        include_rolling: Whether to include rolling features
        include_time: Whether to include time features
        
    Returns:
        Feature DataFrame and target Series
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create time features
    if include_time:
        df = create_time_features(df, date_col)
    
    # Create lag features
    if include_lags:
        df = create_lag_features(df, target_col, lags=FEATURE_CONFIG["lag_features"])
    
    # Create rolling features
    if include_rolling:
        df = create_rolling_features(df, target_col, windows=FEATURE_CONFIG["rolling_windows"])
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [target_col, date_col]]
    feature_cols = [col for col in feature_cols if not col.startswith(target_col)]  # Remove target variations
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    return X, y, feature_cols


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train XGBoost model for time series forecasting.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: XGBoost parameters
        
    Returns:
        Trained XGBoost model
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]
    
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get("n_estimators", 100),
        evals=evals,
        early_stopping_rounds=10 if X_val is not None else None,
        verbose_eval=False
    )
    
    return model


def forecast_xgboost(model, X_future, feature_cols):
    """
    Generate forecasts using trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_future: Future feature matrix
        feature_cols: List of feature column names
        
    Returns:
        Forecast array
    """
    # Ensure X_future has same columns as training
    X_future_df = pd.DataFrame(X_future, columns=feature_cols)
    
    # Fill missing columns with 0
    for col in feature_cols:
        if col not in X_future_df.columns:
            X_future_df[col] = 0
    
    # Reorder columns
    X_future_df = X_future_df[feature_cols]
    
    # Make predictions
    dtest = xgb.DMatrix(X_future_df)
    forecast = model.predict(dtest)
    
    return forecast


def evaluate_ml_model(y_true, y_pred, model_name="ML Model"):
    """
    Evaluate ML model performance.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    return {
        "model_name": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2)
    }

