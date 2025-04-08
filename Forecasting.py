import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def home():
    # Streamlit Interface
    st.title('Vellore Water Supply Data Analysis')

    # Dataset file uploader
    file = st.file_uploader("📂 Upload your file (Excel)", type=["csv"])
    # Load Data
    if file is not None:
        df = pd.read_csv(file)
        st.snow()

    
        st.write(df.head(8))

        # Preprocessing
        df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01").dt.date
        df.set_index("Date", inplace=True)
        df.drop(columns=['Month', 'Year'], inplace=True)
        st.write("### Clean & Preprocessed Data")
        st.write(df.head(6))

        st.write("### Statical Analysis")
        st.write("***Histogram***")
        sns.set_style("dark")
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis object
        df.hist(ax=ax, bins=10, edgecolor='black')  # Pass the axis object to the histogram
        plt.suptitle("Histogram of Dataset Features", fontsize=14)
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig) 


        st.write("***Box Plot***")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 5))
        # Draw Box Plot
        sns.boxplot(data=df[['TotalWaterSupply_MLD', 'ActualWaterSupplied_MLD']], ax=ax)
        # Set labels and title
        ax.set_title("Box Plot: Total vs. Actual Water Supply")
        ax.set_ylabel("Water Supplied (MLD)")
        # Display in Streamlit
        st.pyplot(fig)



        st.write("***Scatter Plot***")
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 5))
        # Create scatter plot on the axis
        sns.scatterplot(x=df['TotalWaterSupply_MLD'], y=df['ActualWaterSupplied_MLD'], ax=ax)
        ax.set_title("Scatter Plot: Total Water Supply vs. Actual Water Supplied")
        ax.set_xlabel("Total Water Supply (MLD)")
        ax.set_ylabel("Actual Water Supplied (MLD)")
        st.pyplot(fig)  # Use st.pyplot(fig) instead of plt.show()
        st.write("***Scatter Plot***")

        # XGBoost Model
        st.write("### XGBoost Model Results")

        # Create lag features for XGBoost (lags of 1)
        df['Lag1'] = df['ActualWaterSupplied_MLD'].shift(1)
        df = df.dropna()  # Drop the first row as it will have a NaN for Lag1

        X = df[['Lag1']]  # Use Lag1 as the input feature
        y = df['ActualWaterSupplied_MLD']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the XGBoost model
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
        xgb_model.fit(X_train, y_train)

        # Make predictions with XGBoost
        y_pred_xgb = xgb_model.predict(X_test)

        # Calculate XGBoost error metrics
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mse_xgb)

        st.write(f"XGBoost MAE: {mae_xgb}")
        st.write(f"XGBoost MSE: {mse_xgb}")
        st.write(f"XGBoost RMSE: {rmse_xgb}")


        # Prepare the future dates (next 5 years or 60 months)
        future_dates = pd.date_range(start=df.index.max(), periods=61, freq='M')[1:]  # Excluding the last date from current data

        # Create a DataFrame for the next 5 years
        future_df = pd.DataFrame({
            'Lag1': df['ActualWaterSupplied_MLD'].iloc[-1]  # Use the last known value for Lag1
        }, index=future_dates)

        future_df.index = future_df.index.strftime('%Y-%m-%d')


        # Predict future values with XGBoost
        future_predictions = xgb_model.predict(future_df[['Lag1']])

        # Introduce larger fluctuations (range between -20 and +20 or adjust as needed)
        random_fluctuations = np.random.uniform(-20, 20, size=len(future_predictions))  # Random fluctuation range from -20 to +20
        future_predictions_adjusted = future_predictions + random_fluctuations

        # Ensure values don't go negative (assuming water supply cannot be negative)
        future_predictions_adjusted = np.clip(future_predictions_adjusted, 0, None)

        # Add the adjusted predictions to the future dataframe
        future_df['PredictedWaterSupplied_MLD'] = future_predictions_adjusted

        future_df = future_df.drop(columns=['Lag1'])

        # Show the future data
        st.write(future_df)

        # Plot the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['ActualWaterSupplied_MLD'], label='Historical Data', color='blue')
        plt.plot(future_dates, future_predictions_adjusted, label='Future Predictions (With Random Fluctuations)', color='red')
        plt.title('Future Water Supply Predictions for the Next 5 Years')
        plt.xlabel('Date')
        plt.ylabel('Water Supplied (MLD)')
        plt.legend()
        st.pyplot(plt)

