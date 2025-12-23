"""Main Streamlit dashboard application."""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Airport Operations & Retail Demand Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("Airport Operations & Retail Demand Intelligence Platform")
st.markdown("---")

# Sidebar navigation with professional styling
st.sidebar.markdown("### Navigation Menu")
st.sidebar.markdown("---")

# Navigation options
nav_options = ["Overview", "Operations Dashboard", "Forecast Analysis", "Retail Demand", "Data Quality"]

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Overview"

# Create navigation buttons - each option is its own button
current_page = st.session_state.current_page

for page_name in nav_options:
    # Highlight the current page with primary button, others with secondary
    button_type = "primary" if page_name == current_page else "secondary"
    
    # Create button for each option
    if st.sidebar.button(
        page_name,
        key=f"nav_{page_name}",
        use_container_width=True,
        type=button_type
    ):
        # Update session state when button is clicked
        st.session_state.current_page = page_name
        st.rerun()

# Get current page from session state
page = st.session_state.current_page

# Add spacing
st.sidebar.markdown("---")

# Route to appropriate page
try:
    if page == "Overview":
        from app.page_modules.overview import show_overview
        show_overview()
    elif page == "Operations Dashboard":
        from app.page_modules.operations import show_operations
        show_operations()
    elif page == "Forecast Analysis":
        from app.page_modules.forecast import show_forecast
        show_forecast()
    elif page == "Retail Demand":
        from app.page_modules.retail import show_retail
        show_retail()
    elif page == "Data Quality":
        from app.page_modules.data_quality import show_data_quality
        show_data_quality()
except Exception as e:
    st.error(f"Error loading page: {str(e)}")
    st.exception(e)
