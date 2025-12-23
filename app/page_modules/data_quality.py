"""Data quality monitoring page."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app.utils import load_data_quality_checks, get_db_connection
from sqlalchemy import text


def show_data_quality():
    """Display data quality page."""
    st.header("Data Quality Monitoring")
    
    st.markdown("""
    **What this page shows:** This page monitors the quality and integrity of the data used throughout the 
    platform. It provides transparency into data validation checks and helps ensure reliable analytics.
    
    **How data quality is checked:** Automated validation rules are applied to all datasets during the ETL 
    process. These checks include:
    - No negative values (passenger counts, sales amounts cannot be negative)
    - Reasonable ranges (values within expected bounds)
    - Data completeness (no missing critical fields)
    - Date validity (dates are in proper format and chronological order)
    - Consistency checks (relationships between related data points make sense)
    
    **How the quality score works:** The overall quality score is calculated as the percentage of validation 
    checks that pass. For example, if 9 out of 10 checks pass, the quality score is 90%. This provides a 
    quick indicator of data reliability. Individual check results show which validations passed or failed, 
    helping identify specific data issues that need attention.
    """)
    st.markdown("---")
    
    # Load data quality checks
    quality_df = load_data_quality_checks()
    
    if quality_df.empty:
        st.warning("No data quality checks available. Please run the ETL pipeline first.")
        return
    
    # Overall quality score with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Overall Data Quality Score")
    with col2:
        help_key = "show_help_quality_score"
        if st.button("?", key="btn_help_quality_score"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Overall Data Quality Score** is a percentage indicating how many validation checks passed.
        - **Calculation**: (Number of Passed Checks ÷ Total Checks) × 100
        - **100%**: All checks passed - data is in excellent condition
        - **80-99%**: Most checks passed - data is generally reliable
        - **Below 80%**: Multiple checks failed - data quality issues need attention
        
        The gauge chart provides a visual indicator:
        - **Green (80-100%)**: Good quality
        - **Yellow (50-80%)**: Acceptable but needs monitoring
        - **Red (0-50%)**: Poor quality, requires immediate attention
        
        A high quality score ensures that your analytics and forecasts are based on reliable data.
        """)
    
    # Calculate quality score from recent checks
    recent_checks = quality_df.head(20)  # Last 20 checks
    if not recent_checks.empty:
        total_checks = len(recent_checks)
        passed_checks = len(recent_checks[recent_checks["status"] == "PASS"])
        quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality Score", f"{quality_score:.1f}%")
        with col2:
            st.metric("Total Checks", total_checks)
        with col3:
            st.metric("Passed Checks", passed_checks, delta=f"{total_checks - passed_checks} failed")
        
        # Quality score visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent checks with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Recent Data Quality Checks")
    with col2:
        help_key = "show_help_recent_checks"
        if st.button("?", key="btn_help_recent_checks"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Recent Data Quality Checks** shows a detailed table of all validation checks performed on your data.
        Each check verifies a specific data quality rule:
        - **Check Name**: What was being validated (e.g., "no_negative_passenger_counts")
        - **Check Type**: Which dataset was checked (passenger_traffic, retail_sales, weather_data)
        - **Status**: PASS (check succeeded) or FAIL (check found issues)
        - **Message**: Description of what was found
        - **Rows Checked**: How many records were examined
        - **Rows Failed**: How many records failed the check (0 for PASS status)
        
        Use the filters to:
        - Focus on specific datasets (Check Type filter)
        - See only failed checks (Status filter) to identify problems quickly
        
        Reviewing this table helps you understand exactly what validation is happening and identify any data issues.
        """)
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        check_type_filter = st.selectbox(
            "Filter by Check Type",
            options=["All"] + quality_df["check_type"].unique().tolist()
        )
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "PASS", "FAIL"]
        )
    
    # Apply filters
    filtered_df = quality_df.copy()
    if check_type_filter != "All":
        filtered_df = filtered_df[filtered_df["check_type"] == check_type_filter]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["status"] == status_filter]
    
    # Display checks
    if not filtered_df.empty:
        # Select only columns that exist
        available_cols = ["check_name", "check_type", "status", "message", "rows_checked", "rows_failed", "check_date"]
        display_cols = [col for col in available_cols if col in filtered_df.columns]
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True
        )
    else:
        st.info("No checks match the selected filters.")
    
    st.markdown("---")
    
    # Check status by type with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Check Status by Type")
    with col2:
        help_key = "show_help_check_status"
        if st.button("?", key="btn_help_check_status"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Check Status by Type** groups validation checks by dataset type and shows how many passed or failed.
        This bar chart helps you:
        - Quickly see which datasets have quality issues
        - Compare data quality across different data sources
        - Identify datasets that need attention
        
        The chart shows:
        - **X-axis**: Different dataset types (passenger_traffic, retail_sales, weather_data)
        - **Y-axis**: Number of checks
        - **Bars**: Grouped by status (PASS in one color, FAIL in another)
        
        If you see many FAIL bars for a particular dataset type, that dataset may have systematic data quality issues 
        that need to be addressed in the ETL pipeline.
        """)
    
    status_by_type = quality_df.groupby(["check_type", "status"]).size().reset_index(name="count")
    status_pivot = status_by_type.pivot(index="check_type", columns="status", values="count").fillna(0)
    
    if not status_pivot.empty:
        fig = px.bar(
            status_pivot.reset_index(),
            x="check_type",
            y=status_pivot.columns.tolist(),
            title="Check Status by Type",
            labels={"value": "Count", "check_type": "Check Type"},
            barmode="group"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Failed checks details with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Failed Checks Details")
    with col2:
        help_key = "show_help_failed_checks"
        if st.button("?", key="btn_help_failed_checks"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Failed Checks Details** lists all validation checks that found data quality issues.
        This table focuses specifically on problems that need attention:
        - **Check Name**: What validation rule failed
        - **Check Type**: Which dataset has the issue
        - **Message**: Description of what went wrong
        - **Rows Checked**: Total records examined
        - **Rows Failed**: How many records have the problem
        
        Use this section to:
        - Quickly identify data quality problems
        - Understand the scope of issues (how many records are affected)
        - Prioritize which issues to fix first (focus on checks with many failed rows)
        
        If this section is empty, congratulations! All your data quality checks are passing.
        """)
    
    failed_checks = quality_df[quality_df["status"] == "FAIL"]
    
    if not failed_checks.empty:
        st.warning(f"Found {len(failed_checks)} failed checks")
        # Select only columns that exist
        available_cols = ["check_name", "check_type", "message", "rows_checked", "rows_failed", "check_date"]
        display_cols = [col for col in available_cols if col in failed_checks.columns]
        st.dataframe(
            failed_checks[display_cols],
            use_container_width=True
        )
    else:
        st.success("All checks passed!")
    
    st.markdown("---")
    
    # Data freshness with help
    col1, col2 = st.columns([20, 1])
    with col1:
        st.subheader("Data Freshness")
    with col2:
        help_key = "show_help_data_freshness"
        if st.button("?", key="btn_help_data_freshness"):
            st.session_state[help_key] = not st.session_state.get(help_key, False)
    if st.session_state.get(help_key, False):
        st.info("""
        **Data Freshness** shows when each dataset was last updated and how current the data is.
        The table displays:
        - **Dataset**: Which data source (Merged Dataset, Passenger Traffic, Weather, Retail Sales)
        - **Last Data Date**: The most recent date in the actual data
        - **File Last Modified**: When the data file was last updated on disk
        - **Days Since Update**: How many days ago the file was last modified
        
        This information helps you:
        - Ensure data is being updated regularly
        - Identify stale datasets that may need attention
        - Understand data currency for decision-making
        
        A warning appears if any dataset hasn't been updated in more than 7 days, indicating the ETL pipeline 
        may need to be run or checked for issues.
        """)
    
    try:
        from config import PROCESSED_DATA_DIR
        from pathlib import Path
        
        freshness_data = []
        
        # Check merged dataset (has all data)
        merged_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
        if merged_path.exists():
            try:
                df = pd.read_csv(merged_path, nrows=1)  # Just check if file exists
                # Get file modification time
                file_mtime = Path(merged_path).stat().st_mtime
                last_update = datetime.fromtimestamp(file_mtime).date()
                days_old = (datetime.now().date() - last_update).days
                
                # Get actual last date from data
                df_full = pd.read_csv(merged_path)
                if "date" in df_full.columns:
                    df_full["date"] = pd.to_datetime(df_full["date"])
                    last_data_date = df_full["date"].max().date()
                    freshness_data.append({
                        "Dataset": "Merged Dataset",
                        "Last Data Date": last_data_date.strftime("%Y-%m-%d"),
                        "File Last Modified": last_update.strftime("%Y-%m-%d"),
                        "Days Since Update": days_old
                    })
            except Exception as e:
                pass
        
        # Check individual datasets
        datasets = {
            "passenger_traffic": "passenger_traffic_clean.csv",
            "weather": "weather_data_clean.csv",
            "retail_sales": "retail_sales_clean.csv"
        }
        
        for dataset_name, filename in datasets.items():
            file_path = PROCESSED_DATA_DIR / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        last_data_date = df["date"].max().date()
                        file_mtime = Path(file_path).stat().st_mtime
                        last_update = datetime.fromtimestamp(file_mtime).date()
                        days_old = (datetime.now().date() - last_update).days
                        
                        freshness_data.append({
                            "Dataset": dataset_name.replace("_", " ").title(),
                            "Last Data Date": last_data_date.strftime("%Y-%m-%d"),
                            "File Last Modified": last_update.strftime("%Y-%m-%d"),
                            "Days Since Update": days_old
                        })
                except Exception as e:
                    continue
        
        if freshness_data:
            freshness_df = pd.DataFrame(freshness_data)
            st.dataframe(freshness_df, use_container_width=True)
            
            # Warn if data is stale (older than 7 days)
            if "Days Since Update" in freshness_df.columns:
                stale_datasets = freshness_df[freshness_df["Days Since Update"] > 7]
                if not stale_datasets.empty:
                    st.warning(f"Some datasets have not been updated in more than 7 days")
        else:
            st.info("No data files found. Please run the ETL pipeline first.")
    except Exception as e:
        st.info(f"Could not retrieve data freshness information: {str(e)}")

