"""
This script contains the performance evaluation functions for the DRL agents. We will use the pyfolio library
    to do Backtesting and to evaluate the performace of the trading strategy
    We will calculate various performance metrics such as:

    Sharpe ratio, ...
"""

import numpy as np
# import pyfolio as pf
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('TkAgg')

def compute_sharpe(returns, risk_free_rate=0.0):
    """
    Annualized Sharpe ratio, assumes daily returns.
    """
    excess_ret = returns - risk_free_rate/252
    return (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)

# def compute_cum_returns(data, stock, date_data):
#     djia = data.dropna()
#     vals = djia['close'].to_numpy()

#     dates = pd.date_range(start=str(date_data['date'].iloc[0]), end = str(date_data['date'].iloc[-1]), periods=len(vals))
    
#     cumulative_return = (vals / vals[0]) - 1

#     df_plot = pd.DataFrame({f'{stock}': cumulative_return}, index=dates)
#     return df_plot


def analyze_performance(portfolio_values, stock, start_date=None, plotting = True):
    """
    Plot cumulative returns of portfolio returns.
    
    Args:
        portfolio_values (list): Portfolio value over time.
        stock_tickers (list): List of stock tickers traded.
        baseline_values (dict): Optional, dictionary {ticker: list of values}.
        start_date (str): Optional, for x-axis labeling.
    """
    # Convert portfolio values to cumulative returns
    portfolio_values = np.array(portfolio_values)
    cumulative_return = (portfolio_values / portfolio_values[0]) - 1

    risk_free_rate = 0.0
    pv = pd.Series(portfolio_values)
    returns = pv.pct_change().dropna()
    portfolio_cum_returns = (1 + returns).cumprod() - 1 # Is this or cumulative_return better?

    sharpe = compute_sharpe(returns, risk_free_rate)

    dates = pd.date_range(start=start_date, periods=len(portfolio_values)) if start_date else range(len(portfolio_values))
    df_plot = pd.DataFrame({f'{stock}': cumulative_return}, index=dates)

    if plotting:
        # Plot
        plt.figure(figsize=(12, 6))
       

        plt.plot(df_plot.index, df_plot, label=stock)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.suptitle(f"Sharpe: {sharpe:.2f}")
        plt.savefig(f"utils/results/{stock}_cumulative_returns.png")

    return df_plot


#





