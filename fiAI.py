import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from ta.utils import dropna
import requests
import streamlit as st
import websocket
import json
import os

# Fetch stock data with technical indicators
def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = dropna(df)
    df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume')
    df = df[['Close', 'volume_adi', 'momentum_rsi', 'trend_macd', 'volatility_bbp']].fillna(0)
    return df

# Prepare data for training
def prepare_data(df, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    X, y = [], []
    
    # Ensure that the range and index are correctly aligned
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])  # Last 60 days of data as input features
        y.append(scaled_data[i, 0])  # Predict the 'Close' price (first column)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# Build optimized LSTM model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_model(ticker):
    df = get_stock_data(ticker)
    
    # Ensure we pass only the relevant columns for training
    X, y, scaler = prepare_data(df[['Close', 'volume_adi', 'momentum_rsi', 'trend_macd', 'volatility_bbp']])
    
    model = build_model((X.shape[1], X.shape[2]))  # Correct input shape
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    return model, scaler

# Predict next price
def predict_next_price(model, scaler, ticker):
    df = get_stock_data(ticker, period='70d')
    
    # Prepare the data for prediction
    X, _, _ = prepare_data(df[['Close', 'volume_adi', 'momentum_rsi', 'trend_macd', 'volatility_bbp']])
    
    # Predict the last available time step
    prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))  # Predict the next closing price
    return scaler.inverse_transform(np.concatenate((prediction, np.zeros((1, X.shape[2] - 1))), axis=1))[0][0]

# Real-time stock prediction using WebSockets
def on_message(ws, message):
    data = json.loads(message)
    if 'p' in data:  # Extract price if available
        latest_price = data['p']
        st.session_state['latest_price'] = latest_price

# Fetch real-time sentiment analysis
def get_sentiment_score(ticker):
    api_key = os.getenv("FINNHUB_API_KEY")
    url = f'https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('sentiment_score', 0)
    return 0

# Streamlit Web Dashboard
st.title("AI Stock Adviser")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

if st.button("Predict Next Price"):
    model, scaler = train_model(ticker)
    prediction = predict_next_price(model, scaler, ticker)
    sentiment = get_sentiment_score(ticker)
    st.write(f"Predicted Next Price: ${prediction:.2f}")
    st.write(f"Market Sentiment Score: {sentiment}")

# Start WebSocket for real-time data
if st.button("Start Real-Time Streaming"):
    ws_url = f'wss://ws.finnhub.io?token={os.getenv("FINNHUB_API_KEY")}'
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    ws.run_forever()

# Deployment instructions for Google Cloud
if __name__ == "__main__":
    st.write("App is running. Deploy using Google Cloud App Engine.")

