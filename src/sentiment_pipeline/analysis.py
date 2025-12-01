import pandas as pd


def basic_stats(df):
    """
    Compute basic statistics for the DataFrame
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Dictionary with basic statistics
    """
    out = {
        'n_rows': df.shape[0],
        'sentiment_mean': df['sentiment_score'].mean() if 'sentiment_score' in df.columns else None,
        'acceptance_mean': df['acceptance'].mean() if 'acceptance' in df.columns else None
    }
    return out


def save_daily_csv(df_daily, path):
    """
    Save daily DataFrame to CSV
    
    Args:
        df_daily (pd.DataFrame): Daily aggregated DataFrame
        path (str): Output file path

    Returns:
        str: Path where the CSV was saved
    """
    df_daily.to_csv(path, index=False)
    return path
