import pandas as pd
import yfinance as yf
from pathlib import Path
from . import config


def get_stock_data(ticker=None, start_date=None, end_date=None):
    """
    Download stock data from Yahoo Finance, clean and return a DataFrame

    Args:
        ticker (str): Ticker symbol of the stock
        start_date (datetime): Start date of the data range
        end_date (datetime): End date of the data range

    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
    if ticker is None:
        ticker = config.TICKER
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = config.END_DATE
    
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.columns = stock_data.columns.get_level_values(-2)
    stock_data.columns = stock_data.columns.str.lower()
      
    print(f"{ticker} exitoso: {stock_data.shape[0]} filas, {stock_data.shape[1]} columnas")
    return stock_data


def calculate_daily_returns(df, price_col='close'):
    """
    Calculate daily returns from price data.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        price_col (str): Name of the column containing price data
        
    Returns:
        tuple: (df with daily_returns added, stats dict)
    """
    df = df.copy()
    df['daily_returns'] = df[price_col].pct_change() * 100
    
    daily_returns = df['daily_returns'].dropna()
    stats = {
        'mean': daily_returns.mean(),
        'std': daily_returns.std(),
        'max': daily_returns.max(),
        'min': daily_returns.min(),
        'skewness': daily_returns.skew(),
        'kurtosis': daily_returns.kurtosis()
    }
    
    df = df.dropna()
    
    return df, stats


def save_dataframe(df, filename=None, output_dir=None):
    """Save DataFrame to CSV"""
    if output_dir is None:
        output_dir = config.PROCESSED_DIR
    if filename is None:
        filename = config.OUTPUT_FILENAME
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    output_path = path / filename
    df.to_csv(output_path)
    
    return str(output_path)
