import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model('lstm_model.h5')

# Load and preprocess data
@st.cache
def load_data():
    df = pd.read_csv('KLBF.JK.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    ms = MinMaxScaler(feature_range=(0, 1))
    df['Close_ms'] = ms.fit_transform(df[['Close']])
    return df, ms

df, ms = load_data()

def prepare_input(df, look_back=1):
    # Use last data to prepare the input for prediction
    last_data = df['Close_ms'].values[-look_back:].reshape(1, look_back, 1)
    return last_data

# Streamlit app
st.title('Stock Price Prediction with LSTM')

date_input = st.date_input("Select the date to predict", pd.to_datetime(df.index[-1]).date())

if st.button('Predict'):
    try:
        # Prepare data for prediction
        look_back = 1
        last_data = prepare_input(df, look_back)

        future_dates = pd.date_range(start=date_input, periods=365*2, freq='D')  # Predict for 2 years
        predictions = []

        for _ in future_dates:
            pred = model.predict(last_data, batch_size=1)
            predictions.append(pred[0, 0])
            # Update last_data to include the most recent prediction
            last_data = np.append(last_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
        
        # Inverse transform predictions
        predictions_original = ms.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create a DataFrame for visualization
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close': predictions_original.flatten()
        })
        
        # Display results
        st.write(pred_df)
        
        # Plot results
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Close'], color='blue', label='Historical')
        plt.plot(pred_df['Date'], pred_df['Predicted Close'], color='red', label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction for PT Kalbe Farma Tbk.')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        plt.legend()
        st.pyplot(plt)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
