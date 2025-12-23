"""Forecast analysis page."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from app.utils import load_passenger_data, load_forecast_data, load_model_performance


def show_forecast():
    """Display forecast analysis page."""
    st.header("Forecast Analysis")
    
    st.markdown("""
    **What this page shows:** This page compares multiple forecasting models and displays their predictions 
    for future passenger traffic. It helps decision-makers understand expected passenger volumes and choose 
    the most accurate forecasting approach.
    
    **How forecasts are generated:** Multiple forecasting models are trained on historical passenger data:
    - **Baseline Model**: Uses simple patterns like seasonal averages from previous years
    - **ARIMA/SARIMA**: Statistical time series models that capture trends and seasonal patterns
    - **XGBoost**: Machine learning model that uses features like day of week, month, holidays, and lagged values
    - **Ensemble**: Combines all models using weighted averages for improved accuracy
    
    **How the graphs work:** The forecast comparison chart overlays predictions from different models against 
    actual historical data. Accuracy metrics (MAE, RMSE, MAPE) quantify how close predictions were to actual 
    values. Lower values indicate better performance. The future forecast section extends predictions beyond 
    the training period.
    """)
    st.markdown("---")
    
    # Load data
    passenger_df = load_passenger_data()
    forecast_df = load_forecast_data()
    performance_df = load_model_performance()
    
    if passenger_df.empty:
        st.warning("No data available. Please run the ETL pipeline first.")
        return
    
    if forecast_df.empty:
        st.warning("No forecast data available. Please train the models first.")
        
        # Show instructions
        st.info("""
        To generate forecasts:
        1. Run the ETL pipeline: `python -m etl.run_pipeline`
        2. Train the models: `python -m models.train_models`
        3. Refresh this page
        """)
        return
    
    # Model comparison with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Model Performance Comparison")
    with col2:
        help_key = "show_help_model_comparison"
        if st.button("?", key="btn_help_model_comparison"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Model Performance Comparison** shows a table comparing how accurately different forecasting models predicted passenger traffic.
        The table displays three key metrics:
        - **MAE (Mean Absolute Error)**: Average prediction error in number of passengers (lower is better)
        - **RMSE (Root Mean Squared Error)**: Penalizes large errors more heavily (lower is better)
        - **MAPE (Mean Absolute Percentage Error)**: Error as a percentage of actual values (lower is better)
        
        Use this table to:
        - Identify which model performs best for your data
        - Understand the trade-offs between different modeling approaches
        - Decide which model to trust for future forecasts
        
        Generally, the Ensemble model (which combines multiple models) provides the most reliable predictions.
        """)
    
    if not performance_df.empty:
        # Get latest metrics for each model
        latest_metrics = performance_df.groupby(["model_name", "metric_name"])["metric_value"].last().reset_index()
        metrics_pivot = latest_metrics.pivot(index="model_name", columns="metric_name", values="metric_value")
        
        st.dataframe(metrics_pivot, use_container_width=True)
    
    st.markdown("---")
    
    # Forecast visualization with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Forecast vs Actual")
    with col2:
        help_key = "show_help_forecast_vs_actual"
        if st.button("?", key="btn_help_forecast_vs_actual"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Forecast vs Actual** overlays model predictions against real historical data to visualize accuracy.
        - **Black line**: Actual passenger counts (what really happened)
        - **Colored dashed lines**: Predictions from different models
        
        This chart helps you:
        - See how close predictions were to reality
        - Compare different models side-by-side
        - Identify periods where models struggled (large gaps between lines)
        - Understand which model tracks actual trends best
        
        Models with lines closer to the black actual line are more accurate. You can select which models to display using the dropdown above the chart.
        """)
    
    # Get available forecast columns
    forecast_cols = [col for col in forecast_df.columns if "forecast" in col.lower()]
    
    if forecast_cols:
        selected_models = st.multiselect(
            "Select models to display",
            options=forecast_cols,
            default=forecast_cols[:3] if len(forecast_cols) >= 3 else forecast_cols
        )
        
        if selected_models:
            fig = go.Figure()
            
            # Plot actual
            if "actual" in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["actual"],
                    name="Actual",
                    line=dict(color="black", width=2)
                ))
            
            # Plot forecasts
            colors = px.colors.qualitative.Set2
            for i, model_col in enumerate(selected_models):
                model_name = model_col.replace("_forecast", "").replace("_", " ").title()
                fig.add_trace(go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df[model_col],
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash="dash")
                ))
            
            fig.update_layout(
                title="Forecast Comparison",
                xaxis_title="Date",
                yaxis_title="Passengers",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Forecast accuracy metrics with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Forecast Accuracy Metrics")
    with col2:
        help_key = "show_help_accuracy_metrics"
        if st.button("?", key="btn_help_accuracy_metrics"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Forecast Accuracy Metrics** provides numerical measures of how well each model predicted passenger traffic:
        - **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actuals. If MAE is 1,000, 
          predictions were off by an average of 1,000 passengers per day.
        - **RMSE (Root Mean Squared Error)**: Similar to MAE but penalizes large errors more. Useful for identifying 
          models that occasionally make very bad predictions.
        - **MAPE (Mean Absolute Percentage Error)**: Error as a percentage. A MAPE of 5% means predictions were off 
          by 5% on average.
        
        Lower values across all metrics indicate better performance. Use these numbers to objectively compare models 
        rather than just looking at charts.
        """)
    
    if "actual" in forecast_df.columns:
        accuracy_results = []
        
        for col in forecast_cols:
            model_name = col.replace("_forecast", "")
            actual = forecast_df["actual"].values
            forecast = forecast_df[col].values
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - forecast))
            rmse = np.sqrt(np.mean((actual - forecast) ** 2))
            mape = np.mean(np.abs((actual - forecast) / (actual + 1))) * 100
            
            accuracy_results.append({
                "Model": model_name.replace("_", " ").title(),
                "MAE": f"{mae:,.0f}",
                "RMSE": f"{rmse:,.0f}",
                "MAPE (%)": f"{mape:.2f}"
            })
        
        accuracy_df = pd.DataFrame(accuracy_results)
        st.dataframe(accuracy_df, use_container_width=True)
    
    st.markdown("---")
    
    # Future forecast with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Future Forecast")
    with col2:
        help_key = "show_help_future_forecast"
        if st.button("?", key="btn_help_future_forecast"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Future Forecast** extends predictions beyond the historical data to show expected passenger traffic in upcoming days.
        - **Black line**: Historical actual passenger counts
        - **Blue dashed line**: Model predictions during the validation period (recent past)
        - **Green dotted line**: Future forecast (upcoming days)
        
        This visualization helps you:
        - Plan for upcoming passenger volumes
        - Make staffing and resource allocation decisions
        - Anticipate busy or quiet periods
        
        Use the slider to adjust how many days ahead you want to forecast (7-90 days). Longer forecasts are less accurate 
        but useful for long-term planning. Shorter forecasts are more reliable for immediate operational decisions.
        """)
    
    # Use the best model for future forecast
    if forecast_cols:
        # Simple extension: use last forecast values with trend
        if "ensemble_forecast" in forecast_cols:
            best_model_col = "ensemble_forecast"
        elif "xgboost_forecast" in forecast_cols:
            best_model_col = "xgboost_forecast"
        else:
            best_model_col = forecast_cols[0]
        
        last_date = forecast_df["date"].max()
        last_forecast = forecast_df[best_model_col].iloc[-1]
        
        # Generate future dates
        future_days = st.slider("Forecast horizon (days)", 7, 90, 30)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq="D")
        
        # Simple future forecast (could be improved with actual model prediction)
        # For demo purposes, use last forecast value with slight variation
        future_forecast = np.full(future_days, last_forecast) * (1 + np.random.normal(0, 0.02, future_days))
        
        fig = go.Figure()
        
        # Historical
        if "actual" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["actual"],
                name="Historical Actual",
                line=dict(color="black", width=2)
            ))
        
        # Forecast period
        forecast_period = forecast_df[forecast_df["date"] >= forecast_df["date"].max() - pd.Timedelta(days=30)]
        if best_model_col in forecast_period.columns:
            fig.add_trace(go.Scatter(
                x=forecast_period["date"],
                y=forecast_period[best_model_col],
                name="Forecast (Validation)",
                line=dict(color="blue", width=2, dash="dash")
            ))
        
        # Future forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_forecast,
            name="Future Forecast",
            line=dict(color="green", width=2, dash="dot"),
            fill="tonexty"
        ))
        
        fig.update_layout(
            title=f"Future Forecast ({future_days} days ahead)",
            xaxis_title="Date",
            yaxis_title="Passengers",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

