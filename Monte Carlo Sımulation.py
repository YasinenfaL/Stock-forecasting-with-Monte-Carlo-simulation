####################################
# Monte-Carlo Simutaltion
####################################

############################
# Import Libraries
############################

# Data prep.
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

# Financial Library
import yfinance as yf

# Visualazition
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp

############################
# Import Data
############################

start_date = '2020-01-27'
end_date = '2023-01-30'
symbols = ["SISE.IS"]
shares = yf.download(symbols, start=start_date, end=end_date)
shares = pd.DataFrame(shares)
shares.reset_index(inplace=True)
shares.head()


############################
# Overview
############################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(shares)


############################
# Past Prıce Movements And Volume Of The Share
############################

def plot_share(dataframe):
    fig1 = go.Figure(data=[go.Candlestick(x=dataframe["Date"],
                                          open=dataframe["Open"],
                                          high=dataframe["High"],
                                          low=dataframe["Low"],
                                          close=dataframe["Close"])])

    fig2 = px.line(dataframe, x='Date', y='Volume', labels={'Date': 'Date', 'Volume': 'Volume'})

    fig = sp.make_subplots(rows=1, cols=2)
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=1, col=2)

    fig.update_layout(width=1000, height=500, title_text="Şişecam Price and Volume")

    fig.show()


plot_share(shares)

############################
# Stock return calculation
############################

# Stock return
prices = shares['Adj Close']
returns = prices.pct_change().dropna()

# Visualization
fig = px.histogram(returns, x=returns, nbins=50, labels={'value': 'Getiriler'}, )
fig.update_xaxes(title_text='Returns')
fig.show()


############################
# Monte Carlo Simulation
############################

def simulate_stock_price(prices, num_simulations=1000, num_days=252):
    """
   Simulates stock prices using Monte Carlo simulation.

    Parameters:
    num_simulations (int): Number of simulations to run.
    num_days (int): Number of days to simulate for.
    prices (pd.Series): Series containing historical stock prices.
    returns (pd.Series): Series containing historical stock returns.

    Returns:
    simulated_prices (np.ndarray): 2D array containing simulated stock prices for each simulation and each day.
    """
    returns = prices.pct_change().dropna()
    last_price = prices.iloc[-1]
    mean = np.mean(returns)
    var = np.var(returns)
    drift = mean - 0.5 * var
    z = norm.ppf(np.random.rand(10, 2))
    vol = np.std(returns)
    r = drift + vol * norm.ppf(np.random.rand(num_days, num_simulations))
    simulated_prices = np.zeros_like(r)
    simulated_prices[0] = last_price
    for i in range(1, num_days):
        simulated_prices[i] = simulated_prices[i - 1] * (1 + r[i])
    return simulated_prices


simulated_prices = simulate_stock_price(prices)

############################
# Visualization of Monte Carlo Simulation
############################

num_simulations = 1000
traces = []
for i in range(num_simulations):
    trace = go.Scatter(x=prices.index, y=simulated_prices[:, i], mode='lines', name=f"Simulation {i + 1}")
    traces.append(trace)

layout = go.Layout(title='Monte Carlo Simulation', xaxis=dict(title='Days'), yaxis=dict(title='Price'))
fig = go.Figure(data=traces, layout=layout)
fig.show()

############################
# Conclusion
############################

mean_price = round(np.mean(simulated_prices), 2)
low_price = np.percentile(simulated_prices, 5)
high_price = np.percentile(simulated_prices, 95)
print(f"Mean price: {mean_price}")
print(f"Low price: {low_price}")
print(f"High price: {high_price}")

# Mean price: 61.51
# Low price: 36.9295016235503
# High price: 109.59407392403743
