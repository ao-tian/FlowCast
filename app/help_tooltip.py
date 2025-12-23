"""Helper function for creating help tooltips."""
import streamlit as st


def section_with_help(header_text, help_text, key_suffix=""):
    """
    Create a section header with a help tooltip.
    
    Args:
        header_text: The text for the section header
        help_text: The explanation text to show in the tooltip
        key_suffix: Unique suffix for the session state key
    
    Returns:
        A column layout with header and help button
    """
    col1, col2 = st.columns([20, 1])
    
    with col1:
        st.subheader(header_text)
    
    with col2:
        help_key = f"help_{key_suffix}"
        if st.button("❓", key=help_key, help=help_text):
            # This will show the help text in a tooltip
            pass
    
    # Show expandable help section
    help_expander_key = f"help_expander_{key_suffix}"
    with st.expander("", expanded=False):
        st.info(help_text)
    
    # Actually, let's use a simpler approach - just show the help text in an info box when button is clicked
    if help_key in st.session_state and st.session_state[help_key]:
        st.info(help_text)

