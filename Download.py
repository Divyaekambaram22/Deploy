import streamlit as st
import pandas as pd
def download():
    st.subheader("Download Dataset")
    
    df = pd.read_csv("Vellore_Water_Supply_1981_2024.csv")
    # Download the data
    csv_filename = "Vellore_Water_Supply.csv"
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Here", data=csv_data, file_name=csv_filename, mime="text/csv")
    st.markdown("---")
