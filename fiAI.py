import yfinance as yf
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import ta  # For technical analysis indicators


# Function to fetch stock data
def fetch_stock_data(ticker, interval="1d", period="1mo"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return stock, data


# Function to calculate trend
def calculate_trend(data):
    recent_close = data["Close"].values[-10:]
    trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
    return trend


# Function for linear regression
def linear_regression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    m = numerator / denominator
    b = y_mean - m * X_mean
    return m, b


# Function to predict stock movement
def predict_stock_movement(ticker, prediction_days=5):
    stock, data = fetch_stock_data(ticker)
    if data.empty:
        return None
    stock_name = stock.info.get("shortName", ticker)
    current_price = data["Close"].iloc[-1]
    past_high = data["High"].max()
    past_low = data["Low"].min()

    # Feature engineering for linear regression
    data["Day"] = range(len(data))
    X = data["Day"].values
    y = data["Close"].values

    # Calculate regression coefficients
    m, b = linear_regression(X, y)

    # Predict future prices
    future_days = np.array([len(data) + i for i in range(1, prediction_days + 1)])
    predicted_prices = m * future_days + b
    predicted_high = max(predicted_prices)
    predicted_low = min(predicted_prices)

    trend = calculate_trend(data)
    prediction = "Up" if trend > 0 else "Down"
    return stock_name, current_price, predicted_high, predicted_low, prediction, data, predicted_prices


# Add technical indicators to data
def add_technical_indicators(data):
    # Add Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    # Add Bollinger Bands
    data['Bollinger_High'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Bollinger_Low'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    # Add RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    return data


# Plot interactive chart with Plotly
def plot_stock_chart(data, predicted_prices, future_dates, stock_name, ticker):
    fig = go.Figure()

    # Plot historical close prices
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price', line=dict(color='blue', width=3)))

    # Plot predicted prices
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Predicted Prices',
                             line=dict(color='red', dash='dash', width=3)))

    # Add Moving Averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='green', width=2)))
    fig.add_trace(
        go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='orange', width=2)))

    # Update layout
    fig.update_layout(
        title=f"Stock Price Trend for {stock_name} ({ticker})",
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template="plotly_dark",
        showlegend=True
    )

    return fig


# Streamlit UI
st.set_page_config(page_title="FiAI-n1 Stock Prediction", layout="wide")

# Add logo and title
st.title("📈 FiAI-n1 - Stock Price Predictor")

# Sidebar for user input
st.sidebar.header("Stock Prediction Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL").strip().upper()
prediction_days = st.sidebar.slider('Select Prediction Days', min_value=1, max_value=30, value=5)

# Display Stock Prediction Button
if st.sidebar.button("Predict Stock"):
    result = predict_stock_movement(ticker, prediction_days)
    if result:
        stock_name, price, high, low, movement, data, predicted_prices = result
        data = add_technical_indicators(data)

        # Display current price and predictions
        st.markdown(f"### <span style='color:#1f77b4'>{stock_name} ({ticker})</span> - Current Price: **${price:.2f}**",
                    unsafe_allow_html=True)
        st.write(
            f"**Predicted High (Next {prediction_days} Days):** ${high:.2f}   |   **Predicted Low:** ${low:.2f}   |   **Movement:** {movement.upper()}")

        # Display stock chart
        future_dates = pd.date_range(start=data.index[-1], periods=prediction_days + 1, freq='D')[1:]
        st.plotly_chart(plot_stock_chart(data, predicted_prices, future_dates, stock_name, ticker),
                        use_container_width=True)

    else:
        st.error("❌ Invalid ticker or no data available.")

# Portfolio Management (For simplicity, just showing a table)
st.sidebar.header("Portfolio Management")
portfolio_tickers = st.sidebar.text_area("Enter multiple tickers (comma separated)", "AAPL, TSLA")
if st.sidebar.button("Show Portfolio"):
    tickers = portfolio_tickers.split(",")
    portfolio_data = pd.DataFrame()
    for ticker in tickers:
        stock, data = fetch_stock_data(ticker.strip())
        data["Ticker"] = ticker.strip()
        portfolio_data = pd.concat([portfolio_data, data])
    st.write(portfolio_data)

# Footer with styling
st.markdown("""
<footer style="text-align: center; padding: 10px;">
    <p style="color: #888888; font-size: 14px;">Made by FiAI-n1. Powered by Streamlit and Yahoo Finance.</p>
</footer>
""", unsafe_allow_html=True)
