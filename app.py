import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# Mengubah warna latar belakang dan menambahkan efek goyang
st.markdown(
    """
    <style>
    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        50% { transform: translateX(10px); }
        75% { transform: translateX(-10px); }
        100% { transform: translateX(0); }
    }

    .reportview-container {
        background-color: #e0f7fa; /* Hijau muda */
        animation: shake 5s infinite;
    }
    .css-1n6g4vv {
        display: flex;
        justify-content: flex-end;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = load_model('stock_price_lstm.h5')  # Ganti dengan path model Anda

# Load and preprocess data
df = pd.read_csv('KLBF.JK.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Filter data from 2019 onwards
df = df[df.index >= '2019-01-01']

ms = MinMaxScaler(feature_range=(0, 1))
df['Close_ms'] = ms.fit_transform(df[['Close']])

# Streamlit app
st.title('Stock Price Prediction')

st.markdown(
    """
    **Welcome to the Stock Price Prediction App!**
    This application predicts the stock price of PT Kalbe Farma Tbk based on historical data from 2019 onwards.
    Use the calendar to select a date to see the predicted stock price on that date (up to 2 years ahead).
    """
)

# Input date using date picker, allow up to 2 years ahead
two_years_ahead = datetime.date.today() + datetime.timedelta(days=2*365)
selected_date = st.date_input("Select the date:", value=None, min_value=datetime.date.today(), max_value=two_years_ahead, key='date_picker')

if selected_date:
    try:
        target_date = datetime.datetime.combine(selected_date, datetime.datetime.min.time())

        # Prediction logic
        def get_next_weekday(date):
            next_day = date + pd.Timedelta(days=1)
            while next_day.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                next_day += pd.Timedelta(days=1)
            return next_day

        current_date = df.index[-1]
        future_dates = []
        while current_date < target_date:
            current_date = get_next_weekday(current_date)
            future_dates.append(current_date)

        predicted_closes_ms = []
        last_data = df['Close_ms'].values[-1].reshape(-1, 1)
        last_data = last_data.reshape((1, 1, 1))

        for _ in range(len(future_dates)):
            predicted_close_ms = model.predict(last_data)
            predicted_closes_ms.append(predicted_close_ms[0, 0])
            last_data = predicted_close_ms.reshape((1, 1, 1))

        predicted_closes = ms.inverse_transform(np.array(predicted_closes_ms).reshape(-1, 1))
        df_future = pd.DataFrame(predicted_closes, index=future_dates, columns=['Close'])
        df_combined = pd.concat([df, df_future])

        st.write(f"Predicted price on {selected_date.strftime('%Y-%m-%d')}: **Rp {predicted_closes[-1][0]:.2f}**")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['Close'], label='Historical Price', color='blue')
        ax.plot(df_combined.index, df_combined['Close'], label='Predicted Price', color='red', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price (Rp)')
        ax.set_title('Stock Price Prediction')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except ValueError:
        st.error("Invalid date format. Please enter the date in YYYY-MM-DD format.")
