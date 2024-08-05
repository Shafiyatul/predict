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
    Select the number of days to predict up to 2 years (730 days).
""")

# User input for the number of days to predict
num_days = st.slider("Select the number of days to predict", 1, 730)
st.write(f"Selected number of days: {num_days}")

# Predict function
def predict_future_prices(num_days):
    # Prepare the last data point
    last_data = df['Close_ms'].values[-1].reshape(-1, 1)
    last_data = last_data.reshape((1, 1, 1))
    
    future_prices = []
    future_dates = []
    last_date = df.index[-1]

    for _ in range(num_days):
        predicted_close_ms = model.predict(last_data)
        predicted_close = ms.inverse_transform(predicted_close_ms)
        future_prices.append(predicted_close[0][0])

        # Prepare for the next iteration
        last_data = predicted_close_ms.reshape((1, 1, 1))
        last_date += timedelta(days=1)
        future_dates.append(last_date)
    
    return future_dates, future_prices

# Display the predicted prices
if st.button("Predict"):
    future_dates, future_prices = predict_future_prices(num_days)
    st.write(f"Predicted stock prices for the next {num_days} days:")

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
    plt.title(f'Stock Price Prediction\nPT Kalbe Farma Tbk.\nLSTM ({num_days} days)', fontsize=20)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.legend()
    st.pyplot(plt)

# Display the original data
st.subheader("Historical Data")
st.write(df[['Close']].tail())
