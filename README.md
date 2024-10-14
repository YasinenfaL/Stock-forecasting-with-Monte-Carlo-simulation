# Monte Carlo Simulation

![monte-carlo](https://user-images.githubusercontent.com/111612847/228836535-11324c4b-049d-42eb-bf4d-6c7f24125c6e.png)

## Business Problem
An investment firm wants to predict the future price movements of Şişecam stocks. For this purpose, they have conducted fundamental analysis based on Şişecam’s financial performance over the past 3 years and decided to use Monte Carlo simulation to simulate possible scenarios. The firm aims to provide its clients with a price range for Şişecam stocks within the next 9 months, using the data obtained from the Monte Carlo simulation. The possible price range for Şişecam stocks will be determined based on the results of the Monte Carlo simulation and presented to the clients.

## Data Quality
This data set includes historical price data for Şişecam stock.

## Variables:

* Date: The date corresponding to a specific trading day.
* Open: The opening price of the stock on a given trading day.
* High: The highest price of the stock on a given trading day.
* Low: The lowest price of the stock on a given trading day.
* Close: The closing price of the stock on a given trading day.
* Adj Close: The closing price of the stock is the price calculated by adjusting for splits in the stock price and other factors.
* Volume: The number of stocks realized on a given trading day.

## Install: 
- docker build -t monte-carlo-simulation .

- docker run -it --rm monte-carlo-simulation

