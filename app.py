import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Load the model
model = tf.keras.models.load_model('lstm_model.h5')

# Load the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
df = pd.read_csv('KLBF.JK.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
scaler.fit(df[['Close']])

# Streamlit App
st.title('Prediksi Harga Saham PT Kalbe Farma Tbk.')

# Input date from user
input_date = st.date_input("Pilih Tanggal untuk Prediksi", min_value=datetime.today())
input_date_str = input_date.strftime('%Y-%m-%d')

# Prepare the last known data for prediction
last_data = df['Close'].values[-1].reshape(-1, 1)
last_data = last_data.reshape((1, 1, 1))

# Predict the price for the input date
if st.button('Prediksi Harga Saham'):
    predicted_close_ms = model.predict(last_data)
    predicted_close = scaler.inverse_transform(predicted_close_ms)
    
    st.write(f"Prediksi harga saham PT Kalbe Farma Tbk. untuk {input_date_str} adalah: Rp {predicted_close[0][0]:.2f}")

    # Add predicted value to DataFrame for visualization
    df_pred = df.copy()
    predicted_date = pd.Timestamp(input_date)
    df_pred = pd.concat([df_pred, pd.DataFrame({'Close': predicted_close[0][0]}, index=[predicted_date])])

    # Plot actual and predicted prices
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'], color='blue', label='Actual')
    plt.plot(df_pred.index, df_pred['Close'], color='red', linestyle='dotted', label='Predicted')
    plt.xlabel('Waktu')
    plt.ylabel('Harga Saham')
    plt.title('Prediksi Harga Saham PT Kalbe Farma Tbk. dengan LSTM', fontsize=20)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.legend()
    plt.xticks(rotation=30)
    st.pyplot(plt)

