"""Overview page for the dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.utils import load_passenger_data, load_forecast_data, load_model_performance


def show_overview():
    """Display overview page."""
    st.header("Project Overview")
    
    # Project description
    st.markdown("""
    ### About This Platform
    
    This **Airport Operations & Retail Demand Intelligence Platform** demonstrates end-to-end 
    data analytics capabilities for airport operations:
    
    - **ETL Pipeline**: Automated data ingestion, cleaning, validation, and storage
    - **Forecasting Models**: Time series models (ARIMA/SARIMA) combined with ML models (XGBoost) 
      to predict passenger flow
    - **Dashboards**: Interactive visualizations for operational decisions and retail demand planning
    - **Data Quality**: Automated validation and monitoring
    
    ### Key Features
    
    **Automated Data Pipeline**: ETL workflows that extract, clean, validate, and store data  
    **Multiple Forecasting Models**: Baseline, ARIMA, XGBoost, and Ensemble models for accurate predictions  
    **Interactive Dashboards**: Real-time insights and forecasts through web-based visualizations  
    **Data Quality Monitoring**: Automated checks ensuring data accuracy and completeness  
    **Workflow Automation**: Scheduled ETL and model retraining capabilities  
    """)
    
    st.markdown("---")
    
    # Load data
    passenger_df = load_passenger_data()
    forecast_df = load_forecast_data()
    performance_df = load_model_performance()
    
    if passenger_df.empty:
        st.warning("No data available. Please run the ETL pipeline first.")
        return
    
    # Key metrics with help
    help_key_metrics = "show_help_key_metrics"
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Key Metrics")
    with col2:
        if st.button("?", key="btn_help_key_metrics"):
            st.session_state[help_key_metrics] = not st.session_state.get(help_key_metrics, False)
    if st.session_state.get(help_key_metrics, False):
        st.info("""
        **Key Metrics** displays the most important summary statistics for the entire dataset:
        - **Total Passengers**: The sum of all passenger counts across all dates in the dataset
        - **Avg Daily Passengers**: The average number of passengers per day, calculated by dividing total passengers by the number of days
        - **Data Coverage**: The time span of your data, showing how many days of historical data you have
        - **Forecast Horizon**: How many days into the future the forecasting models are predicting
        
        These metrics give you a quick overview of the scale and scope of your data.
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_passengers = passenger_df["passenger_count"].sum()
        st.metric("Total Passengers", f"{total_passengers:,.0f}")
    
    with col2:
        avg_daily = passenger_df["passenger_count"].mean()
        st.metric("Avg Daily Passengers", f"{avg_daily:,.0f}")
    
    with col3:
        # Ensure date column is datetime
        if passenger_df["date"].dtype == 'object' or passenger_df["date"].dtype.name == 'object':
            passenger_df["date"] = pd.to_datetime(passenger_df["date"])
        date_range = (passenger_df["date"].max() - passenger_df["date"].min()).days
        st.metric("Data Coverage", f"{date_range} days")
    
    with col4:
        if not forecast_df.empty and "actual" in forecast_df.columns:
            latest_forecast = len(forecast_df)
            st.metric("Forecast Horizon", f"{latest_forecast} days")
        else:
            st.metric("Forecast Horizon", "N/A")
    
    st.markdown("---")
    
    # Data overview chart with help
    help_key_traffic = "show_help_traffic_overview"
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Passenger Traffic Overview")
    with col2:
        if st.button("?", key="btn_help_traffic_overview"):
            st.session_state[help_key_traffic] = not st.session_state.get(help_key_traffic, False)
    if st.session_state.get(help_key_traffic, False):
        st.info("""
        **Passenger Traffic Overview** shows a line chart of daily passenger counts over time. 
        This visualization helps you:
        - Identify trends (increasing, decreasing, or stable passenger volumes)
        - Spot seasonal patterns (holidays, weekends, peak travel seasons)
        - See overall data patterns at a glance
        - Detect any unusual spikes or drops in passenger traffic
        
        The x-axis shows dates, and the y-axis shows the number of passengers. You can hover over any point to see the exact date and passenger count.
        """)
    
    fig = px.line(
        passenger_df,
        x="date",
        y="passenger_count",
        title="Daily Passenger Traffic Over Time",
        labels={"passenger_count": "Passengers", "date": "Date"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance summary with help
    if not performance_df.empty:
        help_key_perf = "show_help_model_performance"
        col1, col2 = st.columns([20, 1])
        with col1:
            st.subheader("Model Performance Summary")
        with col2:
            if st.button("?", key="btn_help_model_performance"):
                st.session_state[help_key_perf] = not st.session_state.get(help_key_perf, False)
        if st.session_state.get(help_key_perf, False):
            st.info("""
            **Model Performance Summary** compares how well different forecasting models performed:
            - **MAE (Mean Absolute Error)**: Average difference between predicted and actual values (lower is better)
            - **RMSE (Root Mean Squared Error)**: Penalizes larger errors more (lower is better)
            - **MAPE (Mean Absolute Percentage Error)**: Error as a percentage of actual values (lower is better)
            
            This table helps you understand which model is most accurate. Models with lower values across all metrics are generally better. 
            The Ensemble model often performs best as it combines predictions from multiple models.
            """)
        
        # Pivot performance data
        performance_pivot = performance_df.pivot_table(
            index="model_name",
            columns="metric_name",
            values="metric_value",
            aggfunc="last"
        )
        
        st.dataframe(performance_pivot, use_container_width=True)
    
    st.markdown("---")
    
    # Architecture diagram (text-based with better styling)
    st.subheader("System Architecture")
    st.markdown("""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 14px; line-height: 1.8; background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; border: 1px solid #e0e0e0;">
    <div style="font-weight: 600; color: #2c3e50;">Raw Data (CSV/API)</div>
    <div style="text-align: center; color: #7f8c8d;">│</div>
    <div style="text-align: center; color: #7f8c8d;">▼</div>
    <div style="font-weight: 600; color: #2c3e50; margin-top: 0.5rem;">ETL Pipeline</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Extract (Data Ingestion)</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Transform (Cleaning & Standardization)</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Validate (Data Quality Checks)</div>
    <div style="margin-left: 1rem; color: #34495e;">└── Load (CSV Storage)</div>
    <div style="text-align: center; color: #7f8c8d; margin-top: 0.5rem;">│</div>
    <div style="text-align: center; color: #7f8c8d;">▼</div>
    <div style="font-weight: 600; color: #2c3e50; margin-top: 0.5rem;">Forecasting Models</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Baseline (Seasonal Naive)</div>
    <div style="margin-left: 1rem; color: #34495e;">├── SARIMA (Time Series)</div>
    <div style="margin-left: 1rem; color: #34495e;">├── XGBoost (ML with Feature Engineering)</div>
    <div style="margin-left: 1rem; color: #34495e;">└── Ensemble (Weighted Combination)</div>
    <div style="text-align: center; color: #7f8c8d; margin-top: 0.5rem;">│</div>
    <div style="text-align: center; color: #7f8c8d;">▼</div>
    <div style="font-weight: 600; color: #2c3e50; margin-top: 0.5rem;">Streamlit Dashboard</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Overview</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Operations Dashboard</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Forecast Analysis</div>
    <div style="margin-left: 1rem; color: #34495e;">├── Retail Demand</div>
    <div style="margin-left: 1rem; color: #34495e;">└── Data Quality</div>
    </div>
    """, unsafe_allow_html=True)

