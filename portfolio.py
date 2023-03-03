import pandas as pd
import streamlit as st

def portfolio_return(start_date,end_date):
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

