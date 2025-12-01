import pandas as pd
import numpy as np
import yfinance as yf


def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data from Yahoo Finance, clean and return a DataFrame

    Args:
        ticker (str): Ticker symbol of the stock
        start_date (datetime): Start date of the data range
        end_date (datetime): End date of the data range

    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.columns = stock_data.columns.get_level_values(-2)
    stock_data.columns = stock_data.columns.str.lower()
      
    print(f"{ticker} exitoso: {stock_data.shape[0]} filas, {stock_data.shape[1]} columnas")
    return stock_data


def canonicalize_index(df, tz="UTC"):
    """
    Make index timezone-aware and deduplicated
    
    Args:
        df (pd.DataFrame): Input DataFrame
        tz (str): Timezone to localize/convert to
        
    Returns:
        pd.DataFrame: DataFrame with canonicalized index
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    return df


def ensure_numeric(df, cols=None):
    """
    Ensure specified columns are numeric
    
    Args:
        df (pd.DataFrame): Input DataFrame
        cols (list): List of columns to convert. If None, convert all columns.

    Returns:
        pd.DataFrame: DataFrame with numeric columns
    """
    df = df.copy()
    if cols is None:
        cols = df.columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return df


def fill_trade_gaps(df, price_cols=['open','high','low','close'], method='ffill', max_gap=3):
    """
    Conservative fill for small intra-week gaps only
    
    Args:
        df (pd.DataFrame): Input DataFrame
        price_cols (list): List of price columns to fill
        method (str): 'ffill' or 'bfill' method
        max_gap (int): only forward-fill sequences <= max_gap length (in rows)
        
    Returns:
        pd.DataFrame: DataFrame with filled price columns
    """
    df = df.copy()
    for col in price_cols:
        if col in df:
            isna = df[col].isna()
            grp = (~isna).cumsum()
            sizes = isna.groupby(grp).transform('sum')
            mask = (isna) & (sizes <= max_gap)
            df.loc[mask, col] = df.loc[mask, col].ffill() if method=='ffill' else df.loc[mask, col].bfill()
    return df


def compute_log_returns(df, price_col='close'):
    """
    Compute log returns from a stock price DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        price_col (str): Name of the column containing stock prices
    
    Returns:
        pd.Series: Series containing the log returns
    """
    s = df[price_col].sort_index()
    return np.log(s / s.shift(1)).rename(df.name if hasattr(df,'name') else price_col)


def build_returns_df(names, returns_dict, how='inner'):
    """
    Build a DataFrame of returns from a dictionary of return series

    Args:
        names (list): List of company names
        returns_dict (dict): Dictionary of return series
        how (str): Join method ('inner' or 'outer')
    
    Returns:
        pd.DataFrame: DataFrame containing the returns
    """
    series = []
    for n in names:
        if n in returns_dict:
            s = returns_dict[n].rename(n)
            series.append(s)
    if not series: 
        return pd.DataFrame()
    return pd.concat(series, axis=1, join=how)


def missingness_report(df):
    """Generate a report of missing values in DataFrame"""
    m = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'n_missing': df.isna().sum(),
        'pct_missing': 100*df.isna().mean()
    })
    return m.sort_values('pct_missing', ascending=False)


def load_all_stocks(tickers_dict, start_date, end_date):
    """
    Load all stocks from tickers dictionary
    
    Args:
        tickers_dict (dict): Dictionary mapping company names to tickers
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        dict: Dictionary mapping company names to DataFrames
    """
    stock_data = {}
    
    print("\n=== Descargando datos de acciones ===")
    for name, ticker in tickers_dict.items():
        try:
            df = get_stock_data(ticker, start_date, end_date)
            df = canonicalize_index(df)
            df = ensure_numeric(df)
            df = fill_trade_gaps(df)
            stock_data[name] = df
        except Exception as e:
            print(f"Error descargando {name} ({ticker}): {e}")
    
    return stock_data


def compute_all_returns(stock_data):
    """
    Compute returns for all stocks
    
    Args:
        stock_data (dict): Dictionary of stock DataFrames
        
    Returns:
        dict: Dictionary of return series
    """
    returns_dict = {}
    
    print("\n=== Calculando retornos logarítmicos ===")
    for name, df in stock_data.items():
        try:
            returns_dict[name] = compute_log_returns(df)
            print(f"{name}: {returns_dict[name].shape[0]} observaciones")
        except Exception as e:
            print(f"Error calculando retornos para {name}: {e}")
    
    return returns_dict
