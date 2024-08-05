import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('lstm_model.h5')

# Load and preprocess the dataset
df = pd.read_csv('KLBF.JK.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Scale the Close prices
ms = MinMaxScaler(feature_range=(0, 1))
df['Close_ms'] = ms.fit_transform(df[['Close']])

# Streamlit app
st.title("Stock Price Prediction: PT Kalbe Farma Tbk.")
st.markdown("""
    This app predicts the stock price of PT Kalbe Farma Tbk. using an LSTM model.
    Select a target date to predict the stock price.
""")

# User input for the target date
last_date = df.index[-1]
target_date = st.date_input("Select a target date for prediction", datetime(2024, 8, 5))
st.write(f"Selected target date: {target_date}")

# Calculate the number of days to predict
num_days = (target_date - last_date).days
if num_days <= 0:
    st.error("The selected date must be in the future.")
else:
    # Predict function
    def predict_future_prices(num_days):
        # Prepare the last data point
        last_data = df['Close_ms'].values[-1].reshape(-1, 1)
        last_data = last_data.reshape((1, 1, 1))

        future_prices = []
        future_dates = []
        current_date = last_date

        for _ in range(num_days):
            predicted_close_ms = model.predict(last_data)
            predicted_close = ms.inverse_transform(predicted_close_ms)
            future_prices.append(predicted_close[0][0])

            # Prepare for the next iteration
            last_data = predicted_close_ms.reshape((1, 1, 1))
            current_date += timedelta(days=1)
            future_dates.append(current_date)

        return future_dates, future_prices

    # Display the predicted prices
    if st.button("Predict"):
        future_dates, future_prices = predict_future_prices(num_days)
        st.write(f"Predicted stock prices up to {target_date}:")

        # Create a dataframe for the predicted prices
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices})
        future_df.set_index('Date', inplace=True)
        st.write(future_df)

        # Plot the actual and predicted prices
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Close'], color='blue', label='Actual')
        plt.plot(future_df.index, future_df['Predicted_Close'], color='red', linestyle='dotted', label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price (Rp)')
        plt.title(f'Stock Price Prediction\nPT Kalbe Farma Tbk.\nLSTM until {target_date}', fontsize=20)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        plt.legend()
        st.pyplot(plt)

    # Display the original data
    st.subheader("Historical Data")
    st.write(df[['Close']].tail())
