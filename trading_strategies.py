import yfinance as yf
import matplotlib.pyplot as plt
# import mplfinance as mpf
import pandas as pd
import streamlit as st
from datetime import date, datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly as px
import requests
import re
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
import numpy as np

def price_mean_reverse_strategy(prices):
    # Compute the rolling mean and standard deviation
    rolling_mean = prices['Close'].rolling(window=20).mean()
    rolling_std = prices['Close'].rolling(window=20).std()
    # Compute the z-score
    z_score = (prices['Close'] - rolling_mean) / rolling_std

    # Define the trading signals
    threshold = 1.5
    long_signal = (z_score < -threshold).astype(int)
    short_signal = (z_score > threshold).astype(int)

    # Combine the signals
    signal = long_signal - short_signal

    # Compute the returns
    prices["mean_reverse_returns"] = np.log(prices['Close']).diff()
    # Apply the signals to the returns to get the strategy returns
    strategy_returns = signal.shift(1) * prices["mean_reverse_returns"]
    # Compute the cumulative returns
    cumulative_returns = strategy_returns.cumsum()
    
    prices['Mean_reverse_Strategy_Returns'] = strategy_returns;

    st.write("Cumulative Returns")
    # st.write(cumulative_returns)
    # Plot the results
    st.line_chart(prices,x="Date",y="Mean_reverse_Strategy_Returns")

    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(cumulative_returns)
    # ax.set_title('Cumulative Returns')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Returns')
    # plt.show()

def price_momentum_strategy(data):
    # Calculate the 10-day momentum
    data['Momentum'] = data['Adj Close'].pct_change(periods=10)

    # Define a buy signal if the momentum is positive
    data['Momentum_strategy_Signal'] = (data['Momentum'] > 0).astype(int)
    
    # Calculate the daily returns
    data['returns'] = data['Adj Close'].pct_change()

    # Calculate the returns for the momentum strategy
    data['Momentum_Strategy_Returns'] = data['Momentum_strategy_Signal'].shift(1) * data['returns']
    
    # st.write(data['Momentum_Strategy_Returns'])
    # Calculate the total returns for the strategy
    # total_returns = (data['Momentum_Strategy_Returns'] + 1).cumprod()[-1] - 1
    # print when momentum strategy return is more than 0
    st.write(data[data['Momentum_Strategy_Returns'] > 0.0])

    # st.write("Total returns price action " + total_returns)

def priceactionstrategy(data):
    #simple price action strategy in Python
    # Calculate the 20-day and 50-day moving averages
    data['ma20'] = data['Adj Close'].rolling(window=20).mean()

    data['ma50'] = data['Adj Close'].rolling(window=50).mean()
    # Calculate the "signal" column, which is 1 if the 20-day moving average is above the 50-day moving average,
    # and 0 otherwise
    data['pa_stragy_signal'] = (data['ma20'] > data['ma50']).astype(int)

    # Calculate the "position" column, which is 1 if the current day's signal is 1 and the previous day's signal was 0,
    # and -1 if the current day's signal is 0 and the previous day's signal was 1. Otherwise, the position is the same as
    # the previous day's position.
    data['position'] = data['pa_stragy_signal'].diff()
    data['position'].fillna(method='ffill', inplace=True)

    # Calculate the daily returns
    data['returns'] = data['Adj Close'].pct_change()

    # Calculate the strategy returns
    data['pa_strategy_returns'] = data['position'] * data['returns']

    # Calculate the cumulative returns
    data['pa_cum_returns'] = (1 + data['pa_strategy_returns']).cumprod() - 1
    # Plot the cumulative returns
    # st.line_chart(data['pa_cum_returns'])



# Define a function to calculate the RSI
def calculate_rsi(data, window):
    delta = data["Adj Close"].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up/ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi
