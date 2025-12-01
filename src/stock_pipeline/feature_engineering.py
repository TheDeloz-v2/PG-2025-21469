import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from . import config


def create_price_features(df):
    """
    Create return-based features
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with return features added
    """
    features = df.copy()
    
    # Returns at different timeframes
    features['daily_return'] = df['close'].pct_change()
    features['weekly_return'] = df['close'].pct_change(5)  # 5 trading days
    features['monthly_return'] = df['close'].pct_change(21)  # 21 trading days
    
    return features


def add_technical_indicators(df):
    """
    Add technical indicators (RSI, MACD, Bollinger Bands)

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    features = df.copy()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config.RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config.RSI_WINDOW).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=config.MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=config.MACD_SLOW, adjust=False).mean()
    features['MACD'] = exp1 - exp2
    features['MACD_signal'] = features['MACD'].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    
    # Bollinger Bands
    features['BB_middle'] = df['close'].rolling(window=config.BOLLINGER_WINDOW).mean()
    bb_std = df['close'].rolling(window=config.BOLLINGER_WINDOW).std()
    features['BB_upper'] = features['BB_middle'] + 2 * bb_std
    features['BB_lower'] = features['BB_middle'] - 2 * bb_std
    
    return features


def add_volatility_features(df):
    """
    Add volatility-based features
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with volatility features added
    """
    features = df.copy()
    
    # Volatility measures
    if 'daily_return' in df.columns:
        features['daily_volatility'] = df['daily_return'].rolling(window=config.VOLATILITY_WINDOW).std()
    
    features['high_low_ratio'] = df['high'] / df['low']
    
    # True Range and ATR
    features['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift(1))
    )
    features['true_range'] = np.maximum(
        features['true_range'],
        np.abs(df['low'] - df['close'].shift(1))
    )
    features['ATR'] = features['true_range'].rolling(window=config.ATR_WINDOW).mean()
    
    return features


def prepare_final_dataset(df, normalize=True):
    """
    Prepare final dataset by cleaning and optionally normalizing
    
    Args:
        df (pd.DataFrame): Input DataFrame
        normalize (bool): Whether to normalize features
        
    Returns:
        pd.DataFrame: Clean and optionally normalized DataFrame
    """
    # Drop NaN values
    df_clean = df.dropna()
    
    if normalize:
        # Normalize features
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(df_clean)
        df_final = pd.DataFrame(
            features_normalized, 
            columns=df_clean.columns, 
            index=df_clean.index
        )
    else:
        df_final = df_clean
    
    print(f"\n✓ Final dataset prepared: {df_final.shape}")
    print(f"  Date range: {df_final.index.min()} to {df_final.index.max()}")
    print(f"  Features: {df_final.shape[1]}")
    
    return df_final
