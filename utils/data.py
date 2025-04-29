'''
Data extraction, preprocessing, and loading module.
'''


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
# from sklearn.model_selection import train_test_split

# Skiping the technical indicators and risk turbulence for now - come back later if required


def download_data(ticker, start_date, end_date, save=False):
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters:
    ticker (str): Stock ticker symbol: '^DJI' for DOW, 'SPY' for S&P 500
    start_date (str): Start date for historical data (YYYY-MM-DD).
    end_date (str): End date for historical data (YYYY-MM-DD).
    
    Returns:
    pd.DataFrame: Historical stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    print(data.head())

    if save:
        data.to_csv(f"data/{ticker}_historical_data.csv")


def load_viz_data(ticker, start_date, end_date, feature='Close', plot=False):
    """
    Parameters:
    feature (str): Feature to visualize (e.g. 'Close', 'Open', 'High', 'Low', 'Volume').
    """

    data = pd.read_csv(f"data/{ticker}_historical_data.csv")
    data = data.dropna() # Drop rows with NaN
    #print(data.head())
    data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']

    data = data.drop(index=0).reset_index(drop=True) # Dropping filler heading row

    
    # Set 'Date' as datetime and sorting it
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)


    if plot:

        plt.figure(figsize=(12, 6))
        plt.plot(data['close'], label=f'{ticker} {feature} Price', color='blue')
        plt.title(f'{ticker} Historical {feature} Price ({start_date} to {end_date})')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return data


def stock_env_technical_indicators(data, ticker: str, tech_ind = False):
    """
    data: DataFrame with Data columns ['date', 'close', 'high', 'low', 'open', 'volume']
    ticker: e.g. '^DJI' or 'SPY'
    """
    df = data.copy()
    df = df.dropna(subset=['close'])

    df['tic'] = ticker

    df['date'] = pd.to_datetime(df['date'])
    df['timestamp'] = pd.to_datetime(df['date'])
    df["turbulence"] = 0


    for col in data.columns:
        if col not in ['date', 'tic', 'timestamp']:
            for i in range(len(data[col])):
                df.loc[i, col] = values = float(data[col][i])

    #Add technical indicators
    #  RSI (14), CCI (20), ADX (14), MACD (12,26,9)
    if tech_ind:
        df['rsi_30'] = ta.rsi(df['close'], length=30)
        df['cci_30'] = ta.cci(df['high'], df['low'], df['close'], length=30)
        df['adx_30'] = ta.adx(df['high'], df['low'], df['close'], length=30)['ADX_30']
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']

    df = df.dropna().reset_index(drop=True)

    return df


def train_test_split(data, train_size=0.8):
    split_idx = int(len(data) * train_size)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data

    






