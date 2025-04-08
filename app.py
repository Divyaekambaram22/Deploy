import streamlit as st
import Forecasting
import project_description
import suggestions
import Download
from streamlit_option_menu import option_menu
st.set_page_config(page_title="Vellore Water Supply Data Analysis", page_icon="💧", layout="centered")
with st.sidebar:
    selected_menu = option_menu(
        menu_title="CONTENT",
        options=["Home", "Project Description","Download CSV_file", "Any Suggestions"],
        icons=["house-fill","exclamation-circle-fill","cloud-arrow-down","envelope-fill"],
        default_index=0,
    )
        # Page navigation
if selected_menu == "Home":
    Forecasting.home()

if selected_menu == "Project Description":
    project_description.detail()

if selected_menu == "Download CSV_file":
    Download.download()

if selected_menu == "Any Suggestions":
    suggestions.simple_feedback_form()



