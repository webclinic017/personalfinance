import yfinance as yf
import matplotlib.pyplot as plt
# import mplfinance as mpf
import pandas as pd
import streamlit as st
from datetime import date, datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
# import plotly as px
import requests
import re
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
import numpy as np
from trading_strategies import price_mean_reverse_strategy
from trading_strategies import price_momentum_strategy, priceactionstrategy,calculate_rsi
import datetime



# @st.cache
def get_historical_data(symbol, start_date = None, enddate = None):
    data = yf.download(symbol, start_date, end_date)
    data.reset_index(inplace=True)
    return data

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def getdata(url):
  r = requests.get(url,headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
  data = pd.read_html(r.text)
  return data


today = datetime.date.today()
before = today - datetime.timedelta(days=200)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')


st.title('Stock Forecast App')
stocks = ('SPY','TSLA','QQQ','GOOG', 'AAPL', 'MSFT', 'GME','SONATSOFTW.NS','YESBANK.NS')

selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
url = f'https://finance.yahoo.com/quote/{selected_stock}?p={selected_stock}'
summary_data = getdata(url)
tickerdata = [summary_data[0], summary_data[1]]
tickerdata = pd.concat(summary_data)
tickerdata.reset_index(drop=True, inplace=True)
df = tickerdata.transpose()
df.columns = df.iloc[0]
df = df.drop(0)
df = df.reset_index(drop=True)

st.write("52 Week Range:" + df["52 Week Range"].__str__())

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365



# Retrieve the historical stock data for the index
data_load_state = st.text('Loading data...')
data = get_historical_data(selected_stock,start_date,end_date)
tesla_pct_change = data['Adj Close'].pct_change()
st.header(selected_stock + ' Price Movement')
st.line_chart(data,x="Date",y="Adj Close")


st.subheader("Price action strategy")
#analyze pricae action strategy
priceactionstrategy(data)
st.subheader("Price Momentum strategy")
#analyze moment strategy
price_momentum_strategy(data)

st.subheader("Price Reverse Mean strategy")
st.info("Mean Reversion Strategy: This strategy involves identifying stocks or ETFs that have deviated from their long-term average and betting on their return to the mean. For example, if a stock has had a string of bad news that has caused it to decline sharply, you might bet on its recovery by buying it when it is trading below its historical average price.")
price_mean_reverse_strategy(data)

# data['MA50'] = data['Close'].rolling(window=50).mean()
data_load_state.text('Loading data... done!')
plot_raw_data()

# Predict forecast with Prophet.
def predict_prices(data):
    df_train = data[['Date','Adj Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Adj Close": "y"})


    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)



# Calculate the RSI for a window of 14 days
data["RSI"] = calculate_rsi(data, 14)

benchmarkRSI:int = 30

# Calculate the RSI for the last 2 periods (default period is 14 days)
# rsi = yf.TaIndicator(data["Close"]).rsi()

# Find the dates when RSI for the last 2 periods is below 30
# low_rsi_dates = data[data["RSI"] < 30].index

# st.write(low_rsi_dates)
# Print the dates in a human-readable format
# for date in low_rsi_dates:
#    st.write(data.iloc[date])

# # Identify the dates when the RSI drops below 10 for 2 consecutive days
below_10 = data[data["RSI"] < benchmarkRSI]
below_10_2day = below_10["RSI"].rolling(window=2).apply(lambda x: all(x < 30))

st.subheader("Dates when "+selected_stock + " RSI goes below " + benchmarkRSI.__str__())
st.write(below_10_2day)
# for date in below_10_2day.index:
#    st.write(data.iloc[date])


# Bollinger Bands
indicator_bb = BollingerBands(data['Adj Close'])
bb = data
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['Adj Close','bb_h','bb_l']]

#MACD
macd = MACD(data['Adj Close']).macd()

chartcol1,chartcol2 = st.columns(2)
# Plot the prices and the bolinger bands
chartcol1.write('Stock Bollinger Bands')
chartcol1.info("""Bollinger Bands consist of three lines on a price chart. The first line is a moving average, usually a 20-day simple moving average (SMA), which serves as the middle band. The upper and lower bands are plotted at a distance of two standard deviations from the middle band. The upper band is calculated by adding two standard deviations to the middle band, while the lower band is calculated by subtracting two standard deviations from the middle band. Traders use Bollinger Bands to identify potential buy and sell signals. When the price of a security touches the lower Bollinger Band, it is considered oversold, which could be a buy signal. Conversely, when the price touches the upper Bollinger Band, it is considered overbought, which could be a sell signal.
"Bolling er Bands can be used in conjunction with other technical indicators, such as the Relative Strength Index (RSI), to confirm buy and sell signals.""")

chartcol1.line_chart(bb)

progress_bar = st.progress(0)

# Plot MACD
chartcol2.info("Moving Average Crossover Strategy: This strategy involves using two moving averages (e.g. a 50-day moving average and a 200-day moving average) to identify when to buy or sell a stock. When the shorter-term moving average crosses above the longer-term moving average, it is a buy signal, and when the shorter-term moving average crosses below the longer-term moving average, it is a sell signal.")
chartcol2.write('Stock Moving Average Convergence Divergence (MACD)')
chartcol2.area_chart(macd)

# chartcol21,chartcol22 = st.columns(2)

# Plot RSI
st.write('Stock RSI ')
st.header("RSI Chart:"+ selected_stock)

# create traces for each line
trace1 = go.Scatter(x=data["Date"], y=data["Adj Close"], name='Close')
trace2 = go.Scatter(x=data["Date"], y=data["RSI"], name='RSI')
trace3 = go.Scatter(x=data["Date"], y=data["ma20"], name='MA20')
trace4 = go.Scatter(x=data["Date"], y=data["ma50"], name='MA50')


# create a figure with multiple traces
fig = go.Figure()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.add_trace(trace3)
fig.add_trace(trace4)


# add a title and axis labels
fig.update_layout(title='Prices and RSI', xaxis_title='X Axis', yaxis_title='Y Axis')

st.plotly_chart(fig)
# display the chart
# fig.show()


# fig1 = plot_plotly(m, data["Adj Close"])

# fig = px.sca
# st.line_chart(data["Adj Close"])
# st.line_chart.(data["RSI"])



# chartcol21.line_chart(rsi)

#chartcol21.write('Stock RSI ')
# st.write('Prices')
# st.line_chart(data['Adj Close'])

# chartcol31,chartcol32 = st.columns(2)
# Data of recent days
st.subheader('Last 10 days')
st.dataframe(data.tail(10))

# st.write(below_10_2day)


# dates_below_10_2day = below_10[below_10_2day.index] #.index

# # Print the dates when the RSI dropped below 10 for 2 days
# # print(dates_below_10_2day)
# st.header("Dates Below "+benchmarkRSI.__str__()+" RSI")

# st.write(dates_below_10_2day)


st.info("Momentum Strategy: This strategy involves identifying stocks or ETFs that are trending strongly in a particular direction and betting on their continued momentum. For example, if a stock has had a string of positive earnings surprises and is trading at all-time highs, you might bet on its continued rise by buying it when it is trading above its historical average price.")

st.info("Breakout Strategy: This strategy involves identifying stocks or ETFs that have broken through a key resistance level and betting on their continued upward momentum. For example, if a stock has been trading in a narrow range for an extended period of time and then breaks out to the upside, you might bet on its continued rise by buying it when it is trading above the breakout level.")


# 
# Strategy
# buy 52 week high 
# 5% Retrace 
# stop loss: 5%
# > PE Ratio