import yfinance as yf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5d", interval="1h")
    return data

def calculate_trend(data):
    recent_close = data["Close"].iloc[-5:].values
    trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
    return trend

def predict_stock_movement(ticker):
    data = fetch_stock_data(ticker)
    if data.empty:
        return None
    today_data = data.loc[data.index.date == data.index[-1].date()]
    current_price = today_data["Close"].iloc[-1]
    open_price = today_data["Open"].iloc[0]
    price_change = current_price - open_price
    percentage_change = price_change / open_price
    past_high = data["High"].max()
    past_low = data["Low"].min()
    volatility_weight = (past_high - past_low) / past_high
    np.random.seed(42)
    simulated_prices = np.random.normal(
        loc=current_price,
        scale=current_price * abs(percentage_change) * volatility_weight,
        size=1000
    )
    predicted_high = np.percentile(simulated_prices, 95)
    predicted_low = np.percentile(simulated_prices, 5)
    trend = calculate_trend(data)
    prediction = "Up" if trend > 0 else "Down"
    return current_price, predicted_high, predicted_low, prediction, data

# Streamlit UI
st.title("FiAI-n1")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL").strip().upper()

if st.button("Predict"):
    result = predict_stock_movement(ticker)
    if result:
        price, high, low, movement, data = result
        st.write(f"**Current Price:** ${price:.2f}")
        st.write(f"**Predicted High:** ${high:.2f}   |    Predicted Low: ${low:.2f}   |   **Movement:** {movement.upper()}")
        st.subheader("Stock Price Trend")
        fig, ax = plt.subplots()
        ax.plot(data.index, data["Close"], label="Close Price", color='blue')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Invalid ticker or no data available.")
