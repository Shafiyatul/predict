import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model('lstm_model.h5')

# Function to get historical stock prices
def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Function to make prediction
def predict_stock_price(model, data, scaler):
    data_scaled = scaler.transform(data.values.reshape(-1, 1))
    last_data = data_scaled[-1].reshape((1, 1, 1))
    predicted_scaled = model.predict(last_data)
    predicted = scaler.inverse_transform(predicted_scaled)
    return predicted[0][0]

# Streamlit App
st.title('Stock Price Prediction App')
st.write("Predict the future stock prices of PT Kalbe Farma Tbk. using LSTM.")

# Date input
start_date = st.date_input("Start Date", datetime.date(2019, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

if start_date >= end_date:
    st.error("End date must be after start date.")
else:
    # Get historical data
    df = get_stock_data('KLBF.JK', start=start_date, end=end_date)
    st.subheader('Historical Stock Prices')
    st.line_chart(df['Close'])

    # Prepare data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Close_scaled'] = scaler.fit_transform(df[['Close']])

    # Predict future prices
    future_years = st.slider('Years to Predict', 1, 2, 1)
    prediction_date = end_date + pd.DateOffset(years=future_years)

    if prediction_date.weekday() >= 5:  # Skip weekends
        prediction_date += pd.DateOffset(days=(7 - prediction_date.weekday()))

    predicted_price = predict_stock_price(model, df['Close_scaled'], scaler)
    st.subheader(f'Predicted Stock Price for {prediction_date.strftime("%Y-%m-%d")}')
    st.write(f"Rp {predicted_price:.2f}")

    # Visualization
    df_pred = df.copy()
    df_pred.loc[prediction_date] = [None] * (len(df.columns) - 1) + [predicted_price]

    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'], color='blue', label='Actual')
    plt.plot(df_pred.index, df_pred['Close_scaled'], color='red', linestyle='dotted', label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction\nPT Kalbe Farma Tbk.\nLSTM', fontsize=20)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.legend()
    st.pyplot(plt)
