"""Operations dashboard page."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app.utils import load_passenger_data, calculate_yoy_growth


def show_operations():
    """Display operations dashboard."""
    st.header("Operations Dashboard")
    
    st.markdown("""
    **What this page shows:** This dashboard provides operational insights into passenger traffic patterns, 
    trends, and anomalies. It helps airport operations teams understand daily passenger volumes, identify 
    peak periods, and detect unusual patterns that may require attention.
    
    **How the data is generated:** The passenger traffic data is generated synthetically with realistic patterns 
    including weekly seasonality (higher on weekends), monthly seasonality (holiday periods), and gradual 
    trends over time. The data is cleaned to remove errors and outliers, then aggregated by date for analysis.
    
    **How the graphs work:** The line charts show passenger counts over time, allowing you to visually identify 
    trends and patterns. Year-over-year growth calculations compare current period performance to the same 
    period in previous years. Anomaly detection uses statistical methods (standard deviations) to flag 
    days with unusually high or low passenger counts.
    """)
    st.markdown("---")
    
    # Load data
    passenger_df = load_passenger_data()
    
    if passenger_df.empty:
        st.warning("No data available. Please run the ETL pipeline first.")
        return
    
    # Ensure date column is datetime
    if passenger_df["date"].dtype == 'object' or passenger_df["date"].dtype.name == 'object':
        passenger_df["date"] = pd.to_datetime(passenger_df["date"])
    
    # Date filters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=passenger_df["date"].min().date() if len(passenger_df) > 0 else pd.Timestamp.now().date(),
            min_value=passenger_df["date"].min().date() if len(passenger_df) > 0 else pd.Timestamp.now().date() - pd.Timedelta(days=365),
            max_value=passenger_df["date"].max().date() if len(passenger_df) > 0 else pd.Timestamp.now().date()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=passenger_df["date"].max().date() if len(passenger_df) > 0 else pd.Timestamp.now().date(),
            min_value=passenger_df["date"].min().date() if len(passenger_df) > 0 else pd.Timestamp.now().date() - pd.Timedelta(days=365),
            max_value=passenger_df["date"].max().date() if len(passenger_df) > 0 else pd.Timestamp.now().date()
        )
    
    # Filter data
    filtered_df = passenger_df[
        (passenger_df["date"].dt.date >= start_date) & 
        (passenger_df["date"].dt.date <= end_date)
    ].copy()
    
    if filtered_df.empty:
        st.warning("No data in selected date range.")
        return
    
    # Key metrics with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Operational Metrics")
    with col2:
        help_key = "show_help_operational_metrics"
        if st.button("?", key="btn_help_operational_metrics"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Operational Metrics** shows key statistics for the selected date range:
        - **Total Passengers**: Sum of all passengers in the selected period
        - **Avg Daily**: Average number of passengers per day in the period
        - **Peak Day**: The day with the highest passenger count and its date
        - **Lowest Day**: The day with the lowest passenger count and its date
        
        These metrics help operations teams understand capacity needs and identify peak periods that may require additional staffing or resources.
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_passengers = filtered_df["passenger_count"].sum()
        st.metric("Total Passengers", f"{total_passengers:,.0f}")
    
    with col2:
        avg_daily = filtered_df["passenger_count"].mean()
        st.metric("Avg Daily", f"{avg_daily:,.0f}")
    
    with col3:
        max_daily = filtered_df["passenger_count"].max()
        max_date = filtered_df.loc[filtered_df["passenger_count"].idxmax(), "date"]
        st.metric("Peak Day", f"{max_daily:,.0f}", delta=f"{max_date.strftime('%Y-%m-%d')}")
    
    with col4:
        min_daily = filtered_df["passenger_count"].min()
        min_date = filtered_df.loc[filtered_df["passenger_count"].idxmin(), "date"]
        st.metric("Lowest Day", f"{min_daily:,.0f}", delta=f"{min_date.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # Passenger trend with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Passenger Traffic Trend")
    with col2:
        help_key = "show_help_traffic_trend"
        if st.button("?", key="btn_help_traffic_trend"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Passenger Traffic Trend** displays a line chart showing daily passenger counts over the selected time period.
        This visualization helps you:
        - See day-to-day variations in passenger volume
        - Identify weekly patterns (weekends vs weekdays)
        - Spot trends over time (increasing, decreasing, or cyclical)
        - Compare different time periods by adjusting the date filters
        
        Use the date filters at the top to focus on specific time ranges for analysis.
        """)
    
    fig = px.line(
        filtered_df,
        x="date",
        y="passenger_count",
        title="Daily Passenger Traffic",
        labels={"passenger_count": "Passengers", "date": "Date"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year comparison with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Year-over-Year Growth")
    with col2:
        help_key = "show_help_yoy_growth"
        if st.button("?", key="btn_help_yoy_growth"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Year-over-Year Growth** compares current period performance to the same period in the previous year.
        - **Positive values (above 0%)**: Passenger traffic increased compared to last year
        - **Negative values (below 0%)**: Passenger traffic decreased compared to last year
        - **Zero line**: No change from the previous year
        
        This metric helps identify long-term trends and growth patterns. For example, if you see consistent positive growth, 
        it indicates increasing passenger demand over time. Seasonal patterns will show similar growth rates at the same time each year.
        """)
    
    yoy_df = calculate_yoy_growth(filtered_df)
    yoy_df_filtered = yoy_df[yoy_df["yoy_growth"].notna()]
    
    if not yoy_df_filtered.empty:
        fig = px.line(
            yoy_df_filtered,
            x="date",
            y="yoy_growth",
            title="Year-over-Year Growth (%)",
            labels={"yoy_growth": "YoY Growth (%)", "date": "Date"}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Growth")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly aggregation with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Monthly Aggregation")
    with col2:
        help_key = "show_help_monthly_agg"
        if st.button("?", key="btn_help_monthly_agg"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Monthly Aggregation** groups passenger data by month to show:
        - **Total**: Sum of all passengers in each month (bar chart)
        - **Average**: Average daily passengers in each month (line chart)
        
        This view helps you:
        - Identify seasonal patterns (which months are busiest)
        - Compare monthly performance across different years
        - Plan for seasonal variations in passenger volume
        - See overall trends at a higher level than daily data
        
        The bars show total monthly volume, while the line shows the average daily rate, helping you understand both absolute volume and intensity.
        """)
    
    filtered_df["year_month"] = filtered_df["date"].dt.to_period("M")
    monthly_df = filtered_df.groupby("year_month")["passenger_count"].agg(["sum", "mean", "max", "min"]).reset_index()
    monthly_df["year_month"] = monthly_df["year_month"].astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_df["year_month"], y=monthly_df["sum"], name="Total"))
    fig.add_trace(go.Scatter(x=monthly_df["year_month"], y=monthly_df["mean"], name="Average", mode="lines+markers"))
    fig.update_layout(
        title="Monthly Passenger Traffic",
        xaxis_title="Month",
        yaxis_title="Passengers",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly detection with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Anomaly Detection")
    with col2:
        help_key = "show_help_anomaly"
        if st.button("?", key="btn_help_anomaly"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Anomaly Detection** identifies days with unusually high or low passenger counts using statistical methods.
        - **Method**: Flags days where passenger counts are more than 2 standard deviations away from the average
        - **Upper threshold**: Average + (2 × standard deviation) - flags unusually busy days
        - **Lower threshold**: Average - (2 × standard deviation) - flags unusually quiet days
        
        Anomalies might indicate:
        - Special events (concerts, sports events, holidays)
        - Data quality issues (errors in data collection)
        - Unexpected disruptions (weather, strikes, etc.)
        
        Reviewing anomalies helps operations teams prepare for unusual days and identify potential data issues.
        """)
    
    mean_val = filtered_df["passenger_count"].mean()
    std_val = filtered_df["passenger_count"].std()
    threshold_high = mean_val + 2 * std_val
    threshold_low = mean_val - 2 * std_val
    
    filtered_df["anomaly"] = (filtered_df["passenger_count"] > threshold_high) | (filtered_df["passenger_count"] < threshold_low)
    anomalies = filtered_df[filtered_df["anomaly"]]
    
    if not anomalies.empty:
        st.warning(f"Found {len(anomalies)} anomalies (outside ±2 standard deviations)")
        
        fig = px.scatter(
            filtered_df,
            x="date",
            y="passenger_count",
            color="anomaly",
            title="Passenger Traffic with Anomalies Highlighted",
            labels={"passenger_count": "Passengers", "date": "Date"}
        )
        fig.add_hline(y=threshold_high, line_dash="dash", line_color="red", annotation_text="Upper Threshold")
        fig.add_hline(y=threshold_low, line_dash="dash", line_color="red", annotation_text="Lower Threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(anomalies[["date", "passenger_count"]].sort_values("passenger_count", ascending=False))
    else:
        st.info("No anomalies detected in the selected period.")

