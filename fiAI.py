import yfinance as yf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def fetch_stock_data(ticker, interval="1d", period="1mo"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return stock, data


def calculate_trend(data):
    recent_close = data["Close"].values[-10:]
    trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
    return trend


def linear_regression(X, y):
    # Simple linear regression calculation using numpy
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    # Calculate slope (m) and intercept (b)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    m = numerator / denominator
    b = y_mean - m * X_mean

    return m, b


def predict_stock_movement(ticker):
    stock, data = fetch_stock_data(ticker)
    if data.empty:
        return None
    stock_name = stock.info.get("shortName", ticker)
    current_price = data["Close"].iloc[-1]
    past_high = data["High"].max()
    past_low = data["Low"].min()
    volatility_weight = (past_high - past_low) / past_high

    # Feature engineering for manual linear regression
    data["Day"] = range(len(data))
    X = data["Day"].values
    y = data["Close"].values

    # Calculate linear regression coefficients
    m, b = linear_regression(X, y)

    # Predict future prices
    future_days = np.array([len(data) + i for i in range(1, 6)])
    predicted_prices = m * future_days + b
    predicted_high = max(predicted_prices)
    predicted_low = min(predicted_prices)

    trend = calculate_trend(data)
    prediction = "Up" if trend > 0 else "Down"
    return stock_name, current_price, predicted_high, predicted_low, prediction, data, predicted_prices


# Streamlit UI
st.title("FiAI-n1")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL").strip().upper()

if st.button("Predict"):
    result = predict_stock_movement(ticker)
    if result:
        stock_name, price, high, low, movement, data, predicted_prices = result
        st.markdown(f"### <span style='color:blue'>{stock_name} ({ticker}) - Current Price:</span> **${price:.2f}**",
                    unsafe_allow_html=True)
        st.write(
            f"**Predicted High (Next 5 Days):** ${high:.2f}   |    Predicted Low: ${low:.2f}   |   **Movement:** {movement.upper()}")

        # Stock Price Trend & Prediction graph
        st.subheader("Stock Price Trend & Prediction")
        fig, ax = plt.subplots()
        ax.plot(data.index, data["Close"], label="Close Price", color='blue')
        future_dates = pd.date_range(start=data.index[-1], periods=6, freq='D')[1:]
        ax.plot(future_dates, predicted_prices, label="Predicted Prices", linestyle="dashed", color='red')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.set_title(f"Stock Price Trend for {stock_name} ({ticker})")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Invalid ticker or no data available.")
