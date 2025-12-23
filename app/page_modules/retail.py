"""Retail demand analysis page."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.utils import load_passenger_data, load_retail_data


def show_retail():
    """Display retail demand page."""
    st.header("Retail Demand Analysis")
    
    st.markdown("""
    **What this page shows:** This page analyzes retail sales performance and its relationship to passenger 
    volume. It helps retail managers understand sales trends, identify opportunities, and make staffing decisions.
    
    **How the data is generated:** Retail sales data is synthetically generated based on passenger volume with 
    realistic conversion rates (percentage of passengers making purchases). Sales are higher on weekends and 
    holidays when passenger volumes peak. The data includes sales amounts, transaction counts, and category 
    breakdowns (Duty Free, Food & Beverage, Retail, Services).
    
    **How the graphs work:** Sales trend charts show daily sales over time. The correlation scatter plot 
    demonstrates the relationship between passenger volume and sales - you'll see a positive correlation 
    where more passengers generally leads to more sales. Category breakdowns show which retail segments 
    contribute most to revenue. Staffing recommendations use simple heuristics based on expected passenger 
    volume to suggest optimal staffing levels.
    """)
    st.markdown("---")
    
    # Load data
    passenger_df = load_passenger_data()
    retail_df = load_retail_data()
    
    if passenger_df.empty:
        st.warning("No passenger data available. Please run the ETL pipeline first.")
        return
    
    if retail_df.empty:
        st.warning("No retail data available. Please run the ETL pipeline first.")
        return
    
    # Aggregate retail by date
    retail_daily = retail_df.groupby("date").agg({
        "sales_amount": "sum",
        "sales_count": "sum"
    }).reset_index()
    
    # Merge with passenger data
    merged_df = passenger_df.merge(retail_daily, on="date", how="inner")
    
    # Key metrics with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Retail Performance Metrics")
    with col2:
        help_key = "show_help_retail_metrics"
        if st.button("?", key="btn_help_retail_metrics"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Retail Performance Metrics** shows key statistics about retail sales:
        - **Total Sales**: Sum of all sales revenue across the entire dataset
        - **Avg Daily Sales**: Average sales per day, useful for understanding typical daily revenue
        - **Total Transactions**: Total number of purchases made, indicating customer engagement
        - **Conversion Rate**: Percentage of passengers who made a purchase (transactions ÷ passengers × 100)
        
        These metrics help retail managers understand overall performance and identify opportunities for improvement. 
        A low conversion rate might indicate a need for better product placement or marketing.
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = retail_daily["sales_amount"].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        avg_daily_sales = retail_daily["sales_amount"].mean()
        st.metric("Avg Daily Sales", f"${avg_daily_sales:,.0f}")
    
    with col3:
        total_transactions = retail_daily["sales_count"].sum()
        st.metric("Total Transactions", f"{total_transactions:,.0f}")
    
    with col4:
        if len(merged_df) > 0:
            conversion_rate = (merged_df["sales_count"].sum() / merged_df["passenger_count"].sum()) * 100
            st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    
    st.markdown("---")
    
    # Sales trend with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Sales Trend")
    with col2:
        help_key = "show_help_sales_trend"
        if st.button("?", key="btn_help_sales_trend"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Sales Trend** displays daily retail sales over time as a line chart.
        This visualization helps you:
        - Identify sales patterns and trends
        - See how sales change day-to-day
        - Spot seasonal variations (holidays, weekends, peak travel seasons)
        - Compare sales performance across different time periods
        
        Higher sales typically correlate with higher passenger volumes. Use this chart to identify peak sales periods 
        and plan inventory and staffing accordingly.
        """)
    
    fig = px.line(
        retail_daily,
        x="date",
        y="sales_amount",
        title="Daily Retail Sales",
        labels={"sales_amount": "Sales ($)", "date": "Date"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sales vs Passengers correlation with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Sales vs Passenger Volume")
    with col2:
        help_key = "show_help_sales_correlation"
        if st.button("?", key="btn_help_sales_correlation"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Sales vs Passenger Volume** shows the relationship between passenger traffic and retail sales using a scatter plot.
        - Each point represents one day, with x-axis showing passengers and y-axis showing sales
        - **Correlation Coefficient**: A number between -1 and 1 indicating the strength of the relationship
          - Close to 1: Strong positive relationship (more passengers = more sales)
          - Close to 0: Weak relationship (passenger volume doesn't strongly predict sales)
          - Close to -1: Strong negative relationship (more passengers = fewer sales, which would be unusual)
        
        A positive correlation confirms that passenger volume drives sales. The trendline (if visible) shows the expected 
        sales for a given passenger count. Days far from the trendline might indicate special promotions, events, or other factors.
        """)
    
    if len(merged_df) > 0:
        try:
            # Try with trendline first (may fail on Python 3.12 due to scipy compatibility)
            fig = px.scatter(
                merged_df,
                x="passenger_count",
                y="sales_amount",
                trendline="ols",
                title="Sales vs Passenger Count",
                labels={"sales_amount": "Sales ($)", "passenger_count": "Passengers"}
            )
        except Exception:
            # Fallback without trendline if there's a compatibility issue
            fig = px.scatter(
                merged_df,
                x="passenger_count",
                y="sales_amount",
                title="Sales vs Passenger Count",
                labels={"sales_amount": "Sales ($)", "passenger_count": "Passengers"}
            )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        correlation = merged_df["passenger_count"].corr(merged_df["sales_amount"])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    # Category breakdown with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Sales by Category")
    with col2:
        help_key = "show_help_category_breakdown"
        if st.button("?", key="btn_help_category_breakdown"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Sales by Category** breaks down total sales revenue by retail category:
        - **Duty Free**: Tax-free shopping items (alcohol, cosmetics, electronics)
        - **Food & Beverage**: Restaurants, cafes, and food vendors
        - **Retail**: General retail stores (clothing, souvenirs, books)
        - **Services**: Other services (spas, lounges, currency exchange)
        
        This bar chart helps you:
        - Identify which categories generate the most revenue
        - Understand customer spending preferences
        - Make decisions about which categories to expand or promote
        - Allocate resources and floor space based on performance
        
        Categories with taller bars contribute more to overall revenue.
        """)
    
    if "category" in retail_df.columns:
        category_sales = retail_df.groupby("category")["sales_amount"].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=category_sales.index,
            y=category_sales.values,
            title="Total Sales by Category",
            labels={"x": "Category", "y": "Total Sales ($)"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Staffing recommendations with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Staffing Recommendations")
    with col2:
        help_key = "show_help_staffing"
        if st.button("?", key="btn_help_staffing"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Staffing Recommendations** suggests optimal staffing levels based on expected passenger volume.
        The recommendation uses a simple formula:
        - **Base Staff**: 10 employees (minimum staffing for basic operations)
        - **Variable Staff**: 2 additional employees per 1,000 passengers
        - **Calculation**: Base Staff + (Average Passengers ÷ 1,000 × 2)
        
        The recommendation is based on the 30-day average passenger volume to smooth out daily fluctuations.
        This is a starting point - actual staffing needs may vary based on:
        - Store layout and size
        - Peak hours and shift patterns
        - Special events or promotions
        - Individual store performance
        
        Use this as a guideline and adjust based on your specific operational needs.
        """)
    
    if len(merged_df) > 0:
        # Simple staffing model: base staff + variable staff based on forecasted passengers
        base_staff = 10
        staff_per_1000_passengers = 2
        
        # Use recent average for forecast
        recent_avg_passengers = merged_df["passenger_count"].tail(30).mean()
        recommended_staff = base_staff + int((recent_avg_passengers / 1000) * staff_per_1000_passengers)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recent Avg Passengers", f"{recent_avg_passengers:,.0f}")
        with col2:
            st.metric("Recommended Staff", f"{recommended_staff}")
        
        st.info("""
        **Staffing Recommendation Logic:**
        - Base staff: 10 employees
        - Additional staff: 2 per 1,000 passengers
        - Recommendation based on 30-day average passenger volume
        """)
    
    # Monthly sales analysis with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Monthly Sales Analysis")
    with col2:
        help_key = "show_help_monthly_sales"
        if st.button("?", key="btn_help_monthly_sales"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Monthly Sales Analysis** groups sales data by month to show total monthly revenue.
        This view helps you:
        - Identify seasonal patterns (which months are busiest for retail)
        - Compare monthly performance across different years
        - Plan inventory and marketing campaigns for peak months
        - Understand long-term sales trends
        
        The bar chart shows total sales for each month. Taller bars indicate higher revenue months. 
        Use this to identify peak shopping seasons and plan accordingly.
        """)
    
    retail_daily["year_month"] = retail_daily["date"].dt.to_period("M")
    monthly_sales = retail_daily.groupby("year_month")["sales_amount"].sum().reset_index()
    monthly_sales["year_month"] = monthly_sales["year_month"].astype(str)
    
    fig = px.bar(
        monthly_sales,
        x="year_month",
        y="sales_amount",
        title="Monthly Sales",
        labels={"sales_amount": "Sales ($)", "year_month": "Month"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

