import sys
import yfinance as yf
import numpy as np
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit
from PyQt6.QtCore import QTimer
import pyqtgraph as pg


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


class StockPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Stock Predictor")
        self.setGeometry(100, 100, 400, 400)
        layout = QVBoxLayout()

        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter stock ticker")
        self.stock_input.returnPressed.connect(self.update_prediction)

        self.price_label = QLabel("Loading...")
        self.prediction_label = QLabel("...")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.update_prediction)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.stock_input)
        layout.addWidget(self.price_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.refresh_button)

        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_prediction)
        self.timer.start(1000)
        self.update_prediction()

    def update_prediction(self):
        ticker = self.stock_input.text().strip().upper()
        if not ticker:
            return
        price, high, low, movement, data = predict_stock_movement(ticker)
        self.price_label.setText(f"Current Price: ${price:.2f}")
        self.prediction_label.setText(f"Predicted High: ${high:.2f} | Low: ${low:.2f} | Movement: {movement}")
        self.plot_widget.clear()
        self.plot_widget.plot(data["Close"].values, pen='b')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictorApp()
    window.show()
    sys.exit(app.exec())

