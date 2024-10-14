#########################################
# Monte Carlo Simulation - Production Ready
#########################################

# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.subplots as sp

############################
# Helper Functions
############################

def check_df(dataframe, head=5):
    """
    Provides general information about the DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to inspect.
    head (int): Number of rows to display from the top and bottom.

    Returns:
    None
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### Missing Values #####################")
    print(dataframe.isnull().sum())
    print("##################### Descriptive Statistics #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def plot_share(dataframe, title="Price and Volume Chart"):
    """
    Plots the stock price and volume charts.

    Parameters:
    dataframe (pd.DataFrame): Stock data.
    title (str): Title of the chart.

    Returns:
    None
    """
    fig1 = go.Figure(data=[go.Candlestick(x=dataframe["Date"],
                                          open=dataframe["Open"],
                                          high=dataframe["High"],
                                          low=dataframe["Low"],
                                          close=dataframe["Close"])])
    fig2 = go.Figure(data=[go.Scatter(x=dataframe['Date'], y=dataframe['Volume'], mode='lines', name='Volume')])

    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Price Chart", "Volume Chart"))
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=1, col=2)

    fig.update_layout(width=1000, height=500, title_text=title)
    fig.show()

def simulate_stock_price(prices, num_simulations=1000, num_days=252):
    """
    Simulates stock prices using Monte Carlo simulation.

    Parameters:
    prices (pd.Series): Historical stock prices.
    num_simulations (int): Number of simulations to run.
    num_days (int): Number of days to simulate.

    Returns:
    simulated_prices (np.ndarray): Simulated stock prices.
    """
    returns = prices.pct_change().dropna()
    last_price = prices.iloc[-1]
    mean = returns.mean()
    var = returns.var()
    drift = mean - 0.5 * var
    vol = returns.std()
    daily_returns = np.exp(drift + vol * norm.ppf(np.random.rand(num_days, num_simulations)))
    simulated_prices = np.zeros_like(daily_returns)
    simulated_prices[0] = last_price

    for t in range(1, num_days):
        simulated_prices[t] = simulated_prices[t - 1] * daily_returns[t]

    return simulated_prices

def plot_simulation(prices, simulated_prices, num_simulations=100):
    """
    Visualizes the results of the Monte Carlo simulation.

    Parameters:
    prices (pd.Series): Historical stock prices.
    simulated_prices (np.ndarray): Simulated stock prices.
    num_simulations (int): Number of simulations to display.

    Returns:
    None
    """
    fig = go.Figure()
    days = np.arange(simulated_prices.shape[0])

    for i in range(min(num_simulations, simulated_prices.shape[1])):
        fig.add_trace(go.Scatter(x=days, y=simulated_prices[:, i],
                                 mode='lines',
                                 name=f"Simulation {i + 1}",
                                 line=dict(width=1)))

    fig.update_layout(title='Monte Carlo Simulation Results',
                      xaxis_title='Days',
                      yaxis_title='Price')
    fig.show()

def calculate_statistics(simulated_prices):
    """
    Calculates statistical values from the simulation results.

    Parameters:
    simulated_prices (np.ndarray): Simulated stock prices.

    Returns:
    stats (dict): Dictionary containing mean, low, and high price values.
    """
    mean_price = round(np.mean(simulated_prices[-1]), 2)
    low_price = round(np.percentile(simulated_prices[-1], 5), 2)
    high_price = round(np.percentile(simulated_prices[-1], 95), 2)

    stats = {
        'mean_price': mean_price,
        'low_price': low_price,
        'high_price': high_price
    }

    return stats

############################
# Main Flow
############################

def main():
    """
    Main function that orchestrates the entire process.

    Returns:
    None
    """
    
    start_date = '2020-01-27'
    end_date = '2024-01-30'
    symbol = "SISE.IS"

    try:
        shares = yf.download(symbol, start=start_date, end=end_date)
        shares.reset_index(inplace=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    
    check_df(shares)


    plot_share(shares, title=f"{symbol} Price and Volume Chart")

   
    prices = shares['Adj Close']
    returns = prices.pct_change().dropna()

    fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=50)])
    fig.update_layout(title='Return Distribution', xaxis_title='Return', yaxis_title='Frequency')
    fig.show()

   
    num_simulations = 1000
    num_days = 252
    simulated_prices = simulate_stock_price(prices, num_simulations=num_simulations, num_days=num_days)

    
    plot_simulation(prices, simulated_prices, num_simulations=100)

    
    stats = calculate_statistics(simulated_prices)
    print(f"Mean Price: {stats['mean_price']}")
    print(f"5% Low Price: {stats['low_price']}")
    print(f"95% High Price: {stats['high_price']}")

if __name__ == "__main__":
    main()
