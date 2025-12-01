import pandas as pd
from . import config


def load_raw_posts(path=None):
    """
    Load raw posts CSV and parse dates
    
    Args:
        path (str): Path to the raw posts CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the raw posts
    """
    if path is None:
        path = config.RAW_PATH
    df = pd.read_csv(path, parse_dates=['createdAt'])
    df.sort_values(by='createdAt', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_df(df, filename, index=False):
    """
    Save DataFrame to CSV in processed directory
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Filename to save as
        index (bool): Whether to save the index
        
    Returns:
        str: Path where the CSV was saved
    """
    out = f"{config.PROCESSED_DIR}/{filename}"
    Path = __import__('pathlib').Path
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    return out
