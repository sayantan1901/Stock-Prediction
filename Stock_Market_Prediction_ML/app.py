import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('Stock Predictions Model.keras')

st.header('Stock Market Predictor')


compname = st.text_input('Enter Company Name')
stock =st.text_input('Enter Stock Symbol')
start = '2012-01-01'
end = '2022-12-31'

if stock:
    data = yf.download(stock, start ,end)

    st.subheader('Stock Data')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, color='red', label='MA50')
    plt.plot(data.Close, color='green', label='Close Price')
    plt.legend()  # Add legend for clarity
    plt.show()
    st.pyplot(fig1)


    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, color='red', label='MA50')
    plt.plot(ma_100_days, color='blue', label='MA100')
    plt.plot(data.Close, color='green', label='Close Price')
    plt.legend()  # Add legend for clarity
    plt.show()
    st.pyplot(fig2)


    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, color='red', label='MA100')
    plt.plot(ma_200_days, color='blue', label='MA200')
    plt.plot(data.Close, color='green', label='Close Price')
    plt.legend()  # Add legend for clarity
    plt.show()
    st.pyplot(fig3)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, color='red', label='Predicted Price')
    plt.plot(y, color='green', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()  # Add legend for clarity
    plt.show()
    st.pyplot(fig4)

    # Define threshold for buy and sell signals
    threshold = 0.05  # Adjust as needed

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
    fig5 = plt.figure(figsize=(20,6))
    plt.plot(predict, color='red', label='Predicted Price')
    plt.plot(y, color='green', label='Original Price')
    plt.scatter(buy_signals, [predict[i] for i in buy_signals], color='blue', marker='^', label='Buy Signal')
    plt.scatter(sell_signals, [predict[i] for i in sell_signals], color='black', marker='v', label='Sell Signal')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig5)
   # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, predict)

# Calculate Mean Absolute Percentage Error (MAPE)
    denominator = np.where(y == 0, 1, y)  # Replace zero values in y with 1
    mape = np.mean(np.abs((y - predict) / denominator)) * 100

# Handle cases where denominator is zero (to avoid division by zero)
    mape = np.where(denominator == 0, 0, mape)

# Replace zero predicted values with a small non-zero value
    predict_nonzero = np.where(predict == 0, 1e-8, predict)

# Recalculate MAPE with non-zero predicted values
    mape = np.mean(np.abs((y - predict_nonzero) / denominator)) * 100

# Display the accuracy metrics
    st.subheader('Accuracy Metrics')

# Define CSS style for the box
    box_style = """
    <style>
        .box {
            border: 2px solid #105716;
            padding: 10px;
            border-radius: 5px;
            background-color: rgb(167, 241, 167);
        }
    </style>
"""

# Write the HTML for the box and insert accuracy metrics inside
    box_content = f"""
    <div class="box">
        <p><strong>Mean Absolute Error (MAE):</strong> {mae}</p>
        <p><strong>Mean Absolute Percentage Error (MAPE):</strong> {mape.mean():.2f}%</p>
    </div>
"""

# Render the box using Markdown
    st.markdown(box_style, unsafe_allow_html=True)
    st.markdown(box_content, unsafe_allow_html=True)
