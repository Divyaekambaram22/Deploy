import streamlit as st

def detail():
    st.title("Project Description")
    
    st.write("""
    This project is a **web-based application** developed using **Streamlit** to analyze and predict water supply in Vellore. 
    The goal is to provide **data-driven insights** into water availability and forecast future trends using **machine learning (XGBoost)**.
    """)
    
    st.header("🔹 Key Features:")
    
    st.subheader("✔️ Data Upload & Preprocessing")
    st.write("""
    - Users can upload a CSV file containing water supply data.
    - Data is cleaned and formatted, with dates indexed properly.
    """)
    
    st.subheader("✔️ Statistical Analysis & Visualization")
    st.write("""
    - **Histograms** to understand data distribution.
    - **Box plots** to detect anomalies in water supply.
    - **Scatter plots** to observe relationships between total and actual water supply.
    """)
    
    st.subheader("✔️ XGBoost-Based Machine Learning Model")
    st.write("""
    - Uses past water supply data to train an **XGBoost regression model**.
    - **Performance Metrics:** MAE, MSE, RMSE to evaluate model accuracy.
    - **Future Predictions:** Forecasts the next **5 years (60 months)** of water supply with added random fluctuations to simulate real-world variations.
    """)
    
    st.subheader("✔️ Future Water Supply Forecasting")
    st.write("""
    - Predicts future water supply trends using the trained **XGBoost model**.
    - Generates **dynamic forecasts** based on historical data patterns.
    - Visualizes predictions in an **interactive line plot**.
    """)
    
    st.header("🎯 Objective:")
    st.write("""
    This project aims to assist **water management authorities** and **policymakers** in better planning and allocation of water resources in Vellore. 
    By leveraging **machine learning and data visualization**, it provides **actionable insights** to ensure sustainable water distribution.
    """)
