import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('lstm_model.h5')

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)

# Load the dataset
df = pd.read_csv('KLBF.JK.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# Streamlit app
st.title("Stock Price Prediction App")
st.write("Welcome to the Stock Price Prediction App for PT Kalbe Farma Tbk.")
st.write("This app allows you to view historical stock prices and predict future stock prices.")

st.subheader("Instructions")
st.write("""
1. View the historical stock price data from Yahoo Finance.
2. Select a date for prediction (maximum up to 2 years from the last available date).
3. Click 'Predict' to see the predicted stock price.
""")

# Show historical data
st.subheader("Historical Stock Price Data")
st.line_chart(df['Close'])

# Select a date for prediction
st.subheader("Select a Date for Prediction")
max_date = df.index[-1] + timedelta(days=730)
prediction_date = st.date_input("Prediction Date", min_value=df.index[-1], max_value=max_date)

# Predict the stock price for the selected date
if st.button('Predict'):
    # Calculate how many days into the future the prediction date is
    days_ahead = (prediction_date - df.index[-1]).days

    # Use the last available data for prediction
    last_data = df['Close_scaled'].values[-1].reshape(-1, 1)
    last_data = last_data.reshape((1, 1, 1))

    # Perform prediction
    predicted_close_scaled = last_data
    for _ in range(days_ahead):
        predicted_close_scaled = model.predict(predicted_close_scaled)

    # Inverse transform the predicted value
    predicted_close = scaler.inverse_transform(predicted_close_scaled)

    # Display the prediction
    st.subheader("Predicted Stock Price")
    st.write(f"The predicted stock price for {prediction_date} is: Rp {predicted_close[0][0]:,.2f}")

    # Visualize the prediction
    df_pred = df.copy()
    df_pred = pd.concat([df_pred, pd.DataFrame({'Close': predicted_close[0][0]}, index=[prediction_date])])

    st.subheader("Stock Price Prediction Visualization")
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'], color='blue', label='Actual')
    plt.plot(df_pred.index, df_pred['Close'], color='red', linestyle='dotted', label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction\nPT Kalbe Farma Tbk.\nLSTM', fontsize=20)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.xticks(rotation=30)
    plt.legend()
    st.pyplot(plt)

# To run this app, save it as app.py and execute `streamlit run app.py` in your terminal.
