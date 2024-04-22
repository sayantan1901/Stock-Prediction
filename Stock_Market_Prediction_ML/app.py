import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('Stock Predictions Model.keras')

# Streamlit app header
st.header('Stock Market Predictor')

# Input fields for company name, stock symbol, and date range
compname = st.text_input('Enter Company Name')
stock = st.text_input('Enter Stock Symbol')
start = '2012-01-01'
end = '2022-12-31'

# Retrieve stock data from Yahoo Finance based on the input symbol and date range
if stock:
    data = yf.download(stock, start, end)

    # Display stock data
    st.subheader('Stock Data')
    st.write(data)

    # Prepare data for prediction
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    # Preprocessing using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.fit_transform(data_test)

    # Plot MA50
    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(ma_50_days, color='red', label='MA50')
    ax1.plot(data.Close, color='green', label='Close Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_facecolor('#f0f0f0')  # Light gray background
    ax1.grid(color='gray', linestyle='--')  # Customize grid color and linestyle
    st.pyplot(fig1)

    # Plot MA50 and MA100
    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(ma_50_days, color='red', label='MA50')
    ax2.plot(ma_100_days, color='blue', label='MA100')
    ax2.plot(data.Close, color='green', label='Close Price')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.set_facecolor('#f0f0f0')  # Light gray background
    ax2.grid(color='gray', linestyle='--')  # Customize grid color and linestyle
    st.pyplot(fig2)

    # Plot MA100 and MA200
    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(ma_100_days, color='red', label='MA100')
    ax3.plot(ma_200_days, color='blue', label='MA200')
    ax3.plot(data.Close, color='green', label='Close Price')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Price')
    ax3.legend()
    ax3.set_facecolor('#f0f0f0')  # Light gray background
    ax3.grid(color='gray', linestyle='--')  # Customize grid color and linestyle
    st.pyplot(fig3)

    # Prepare data for prediction
    x = []
    y = []

    for i in range(100, data_test_scaled.shape[0]):
        x.append(data_test_scaled[i-100:i])
        y.append(data_test_scaled[i,0])

    x, y = np.array(x), np.array(y)

    # Make predictions using the pre-trained model
    predict = model.predict(x)

    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale

    # Plot Original Price vs Predicted Price
    st.subheader('Original Price vs Predicted Price')
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(predict, color='red', label='Predicted Price')
    ax4.plot(y, color='green', label='Original Price')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.set_facecolor('#f0f0f0')  # Light gray background
    ax4.grid(color='gray', linestyle='--')  # Customize grid color and linestyle
    st.pyplot(fig4)

    # Define threshold for buy and sell signals
    threshold = 0.05

    # Determine buy and sell signals based on threshold
    buy_signals = []
    sell_signals = []
    prev_pred = predict[0]
    for i in range(1, len(predict)):
        if predict[i] > prev_pred * (1 + threshold):
            buy_signals.append(i)
            prev_pred = predict[i]
        elif predict[i] < prev_pred * (1 - threshold):
            sell_signals.append(i)
            prev_pred = predict[i]

    # Plot Original Price, Predicted Price, Buy and Sell Signals
    st.subheader('Original Price vs Predicted Price with Buy/Sell Signals')
    fig5, ax5 = plt.subplots(figsize=(20, 6))
    ax5.plot(predict, color='red', label='Predicted Price')
    ax5.plot(y, color='green', label='Original Price')
    ax5.scatter(buy_signals, [predict[i] for i in buy_signals], color='blue', marker='^', label='Buy Signal')
    ax5.scatter(sell_signals, [predict[i] for i in sell_signals], color='black', marker='v', label='Sell Signal')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Price')
    ax5.legend()
    ax5.set_facecolor('#f0f0f0')  # Light gray background
    ax5.grid(color='gray', linestyle='--')  # Customize grid color and linestyle
    st.pyplot(fig5)
