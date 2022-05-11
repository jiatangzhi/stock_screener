import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose


def get_close(symbol):
    """
    Return the close value for stock indicated by symbol.

    Note: Data for a stock is stored in file: data/<symbol>.csv
    """

    df = pd.read_csv("data/{}".format(symbol))  # read in data
    df1 = df[['Close']]
    result = seasonal_decompose(df1, model='multiplicative',       # multiplicative because the magnitude of the seasonal component changes with time.
                                period=12)                         # data aggregated by month; period is set to 12 because the period wanted to be analyzed is by year.
    solution = result.observed - result.trend  # standardised data obtained from subtracting the raw data minus the trend.
    return solution


def get_list():
    """
       Return the list of all files from the data directory.
    """

    path = "/Users/mcarmentz/Desktop/stock_screener/data"
    dir_list = os.listdir(path)

    data_list = []                         # get a list of lists of stocks

    for symbol in dir_list:
        data = get_close(symbol)
        data = data.dropna()               # drop null values
        if len(data.values) == 492:        # check length value == 492 as some stocks do not have enough data and/or disappeared
            data_list.append(data.values)

    return data_list


def test_run():
    """Function called by Test Run."""
    get_list()


if __name__ == "__main__":
    test_run()

