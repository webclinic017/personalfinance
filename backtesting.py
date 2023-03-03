import backtrader as bt
import pandas as pd
import streamlit as st

# Define the trading strategy
#
class MyStrategy(bt.Strategy):
    params = (
        ('ma_period_short', 10),
        ('ma_period_long', 30)
    )

    def __init__(self):
        # Initialize the short-term and long-term moving averages
        self.ma_short = bt.indicators.SMA(self.data, period=self.params.ma_period_short)
        self.ma_long = bt.indicators.SMA(self.data, period=self.params.ma_period_long)

    def next(self):
        # Buy when the short-term MA crosses above the long-term MA
        if not self.position and self.ma_short > self.ma_long:
            self.buy()

        # Sell when the short-term MA crosses below the long-term MA
        elif self.position and self.ma_short < self.ma_long:
            self.sell()


