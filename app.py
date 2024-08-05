import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

# Fungsi untuk memplot grafik
def plot_graph(df, y_pred_original, predicted_date, predicted_close):
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'], color='blue', label='Actual')
    plt.plot(df.index[-len(y_pred_original):], y_pred_original, color='red', label='Predicted')
    plt.axvline(x=predicted_date, color='green', linestyle='--', label='Predicted Date')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction\nPT Kalbe Farma Tbk.\nLSTM', fontsize=20)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.legend()
    plt.xticks(rotation=30)
    st.pyplot(plt)

# Muat dataset dan model
df = pd.read_csv('KLBF.JK.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
ms = MinMaxScaler(feature_range=(0, 1))
df['Close_ms'] = ms.fit_transform(df[['Close']])

model = load_model('lstm_model.h5')

# Buat prediksi harga satu hari ke depan
last_data = df['Close_ms'].values[-1].reshape(-1, 1).reshape((1, 1, 1))
predicted_close_ms = model.predict(last_data)
predicted_close = ms.inverse_transform(predicted_close_ms)

predicted_date = pd.Timestamp('2024-04-02')
df_pred = df.copy()
df_pred = pd.concat([df_pred, pd.DataFrame({'Close': predicted_close[0][0]}, index=[predicted_date])])

y_pred_original = ms.inverse_transform(np.array(model.predict(df['Close_ms'].values.reshape(-1, 1))).reshape(-1, 1)).reshape(-1)

# Tampilan Streamlit
st.title('Prediksi Harga Saham PT Kalbe Farma Tbk.')

st.write(f"Prediksi harga saham PT Kalbe Farma Tbk. untuk {predicted_date.strftime('%d %B %Y')} adalah: Rp {predicted_close[0][0]:.2f}")

st.subheader('Grafik Harga Saham')
plot_graph(df, y_pred_original, predicted_date, predicted_close)

st.subheader('Tabel Perbandingan')
comparison_df = pd.DataFrame({
    'Date': df.index[-len(y_pred_original):],
    'Actual': df['Close'].values[-len(y_pred_original):],
    'Predicted': y_pred_original
})
st.write(comparison_df)
