import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
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


# Define a function to send an email with the results
def send_email(dates):
    # Set the email parameters
    sender = "your_email@example.com"
    password = "your_email_password"
    recipient = "recipient_email@example.com"
    subject = "RSI below 10"
    body = f"The RSI for {ticker} dropped below 10 on the following dates: {dates}"
    # Create the email message
    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    # Send the email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, message.as_string())

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

# @st.cache
def get_historical_data(symbol, start_date = None,enddate=None):
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

import datetime

today = datetime.date.today()
before = today - datetime.timedelta(days=200)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')


# start_date = st.date_input("Start Date",value= date.today()-datetime.timedelta(days=365))
# end_date = st.date_input("End Date",value= date.today())
# start_date = "2021-01-29"
# end_date = date.today().strftime("%Y-%m-%d")

# Define the stock ticker symbol
# ticker = "SONATSOFTW.NS"

st.title('Stock Forecast App')
stocks = ('SPY','QQQ','GOOG', 'AAPL', 'MSFT', 'GME','SONATSOFTW.NS','YESBANK.NS')

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


portfolio = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'SONATSOFTW.NS']
quantity = [100, 50, 25,216]
purchase_price = [2000, 1500, 3000,142]
portfolio_data = pd.DataFrame()

for stock in portfolio:
    stock_data = yf.download(stock, start=start_date, end=end_date)
    portfolio_data[stock] = stock_data['Adj Close']


daily_returns = portfolio_data.pct_change()
weights = [1/len(portfolio)]*len(portfolio)
portfolio_returns = daily_returns.dot(weights)

current_price = portfolio_data.iloc[-1]
portfolio_value = sum(current_price*quantity)
total_investment = 0
for px in purchase_price:
    total_investment = total_investment+ sum(px*quantity)

# 
data = {'Stock': portfolio, 'Quantity': quantity,
        'Purchase Price': purchase_price,
        'Current Price': current_price,
        'Portfolio Value': current_price*quantity,
        'Portfolio Returns': ((current_price-purchase_price)/purchase_price) *100
         }


df = pd.DataFrame(data)
st.header("Portfolio")
st.write(df)
st.line_chart(portfolio_returns)


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

# data['MA50'] = data['Close'].rolling(window=50).mean()
data_load_state.text('Loading data... done!')
plot_raw_data()

# Predict forecast with Prophet.
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
for date in below_10_2day.index:
   st.write(data.iloc[date])


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
chartcol1.markdown("""Bollinger Bands consist of three lines on a price chart. The first line is a moving average, usually a 20-day simple moving average (SMA), which serves as the middle band. The upper and lower bands are plotted at a distance of two standard deviations from the middle band. The upper band is calculated by adding two standard deviations to the middle band, while the lower band is calculated by subtracting two standard deviations from the middle band. Traders use Bollinger Bands to identify potential buy and sell signals. When the price of a security touches the lower Bollinger Band, it is considered oversold, which could be a buy signal. Conversely, when the price touches the upper Bollinger Band, it is considered overbought, which could be a sell signal.
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

st.info("Mean Reversion Strategy: This strategy involves identifying stocks or ETFs that have deviated from their long-term average and betting on their return to the mean. For example, if a stock has had a string of bad news that has caused it to decline sharply, you might bet on its recovery by buying it when it is trading below its historical average price.")

st.info("Momentum Strategy: This strategy involves identifying stocks or ETFs that are trending strongly in a particular direction and betting on their continued momentum. For example, if a stock has had a string of positive earnings surprises and is trading at all-time highs, you might bet on its continued rise by buying it when it is trading above its historical average price.")

st.info("Breakout Strategy: This strategy involves identifying stocks or ETFs that have broken through a key resistance level and betting on their continued upward momentum. For example, if a stock has been trading in a narrow range for an extended period of time and then breaks out to the upside, you might bet on its continued rise by buying it when it is trading above the breakout level.")

