import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf

# Fungsi untuk mengunduh data saham dari Yahoo Finance
start = '2019-04-01'
end = date.today()

# Fungsi untuk memprediksi harga saham
def predict_stock_price(model, last_data, scaler, days_to_predict):
    predictions = []
    current_data = last_data.copy()

    for _ in range(days_to_predict):
        # Melakukan prediksi
        predicted_scaled = model.predict(current_data)
        # Simpan hasil prediksi
        predictions.append(predicted_scaled[0][0])

        # Perbarui data dengan memasukkan prediksi terakhir
        current_data = np.append(current_data[:, 1:, :], [[predicted_scaled]], axis=1)

    # Inverse transformasi prediksi ke skala aslinya
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predicted_prices

# Mengatur halaman
st.set_page_config(page_title="Prediksi Harga Saham PT Kalbe Farma Tbk", layout="wide")

# Menampilkan judul
st.title("Prediksi Harga Saham PT Kalbe Farma Tbk.")

# Input ticker
ticker = "KLBF.JK"

# Memuat data saham
st.sidebar.header("Parameter")
stock_data = load_stock_data(ticker)

# Menampilkan opsi tanggal prediksi
min_date = stock_data.index.max() + pd.Timedelta(days=1)
max_date = min_date + pd.DateOffset(years=2)
selected_date = st.sidebar.date_input(
    "Pilih tanggal prediksi:", min_value=min_date, max_value=max_date, value=min_date
)

# Memuat model dan scaler
model = load_model('lstm_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

# Mengambil data harga penutupan dan menskalakan data
stock_data['Close_scaled'] = scaler.fit_transform(stock_data[['Close']])

# Mengambil data terbaru untuk prediksi
last_data = stock_data['Close_scaled'].values[-1].reshape((1, 1, 1))

# Hitung hari prediksi (tidak termasuk akhir pekan)
days_to_predict = 0
date = min_date
while date <= selected_date:
    if date.weekday() < 5:  # 0-4 adalah hari kerja
        days_to_predict += 1
    date += pd.Timedelta(days=1)

# Memprediksi harga saham
predicted_prices = predict_stock_price(model, last_data, scaler, days_to_predict)

# Menampilkan grafik riwayat dan prediksi harga saham
st.subheader("Riwayat Harga Saham dan Prediksi")
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(stock_data.index, stock_data['Close'], label='Riwayat Harga Saham', color='blue')

# Menambahkan prediksi ke grafik
predicted_dates = pd.date_range(start=min_date, periods=days_to_predict, freq='B')
ax.plot(predicted_dates, predicted_prices, linestyle='dotted', color='red', label='Prediksi Harga Saham')

ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Saham (Rp)')
ax.set_title('Prediksi Harga Saham PT Kalbe Farma Tbk.')
ax.legend()

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=30)

st.pyplot(fig)

# Menampilkan harga prediksi terakhir
st.subheader(f"Prediksi Harga Saham untuk {selected_date}:")
predicted_price = predicted_prices[-1][0]
st.write(f"Rp {predicted_price:,.2f}")
