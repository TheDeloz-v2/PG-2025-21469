import pandas as pd
from pathlib import Path
from . import config


def load_market_data(path=None):
    """
    Load market pipeline output with target
    
    Returns:
        pd.DataFrame: Market features with target_close_t+1
    """
    if path is None:
        path = config.INPUT_MARKET
    
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    
    # Remove timezone if present to ensure consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    print(f"✓ Market data loaded: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Target column: {'target_close_t+1' if 'target_close_t+1' in df.columns else 'MISSING'}")
    
    return df


def load_sentiment_data(path=None):
    """
    Load sentiment pipeline output
    
    Returns:
        pd.DataFrame: Sentiment scores (sentiment_score, acceptance)
    """
    if path is None:
        path = config.INPUT_SENTIMENT
    
    df = pd.read_csv(path, parse_dates=['day'], index_col='day')
    
    # Remove timezone if present to ensure consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    print(f"✓ Sentiment data loaded: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Features: {list(df.columns)}")
    
    return df


def load_stock_data(path=None):
    """
    Load stock pipeline output (decompositions + technical indicators)
    
    Returns:
        pd.DataFrame: Stock features (normalized)
    """
    if path is None:
        path = config.INPUT_STOCK
    
    # Stock pipeline saves with unnamed index, need to parse it
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = 'Date'
    
    # Remove timezone if present to ensure consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    print(f"✓ Stock data loaded: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Features: {df.shape[1]} (technical + decomposition)")
    
    return df


def save_dataset(df, filename, output_dir=None):
    """Save fused dataset to CSV"""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    output_path = path / filename
    df.to_csv(output_path)
    
    return str(output_path)


def validate_dataframe(df, name="Dataset"):
    """
    Validate DataFrame structure and coverage
    
    Returns:
        dict: Validation results
    """
    validation = {
        'name': name,
        'shape': df.shape,
        'date_range': (df.index.min(), df.index.max()),
        'has_target': config.TARGET_COLUMN in df.columns,
        'n_features': df.shape[1] - (1 if config.TARGET_COLUMN in df.columns else 0),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'samples_with_target': df[config.TARGET_COLUMN].notna().sum() if config.TARGET_COLUMN in df.columns else 0
    }
    
    return validation


def print_validation_report(validation):
    """Print formatted validation report"""
    print(f"\n{'='*60}")
    print(f"Validation Report: {validation['name']}")
    print(f"{'='*60}")
    print(f"Shape: {validation['shape']}")
    print(f"Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
    print(f"Has target: {validation['has_target']}")
    print(f"Features: {validation['n_features']}")
    print(f"Samples with target: {validation['samples_with_target']}")
    
    # Report features with high missing %
    high_missing = {k: v for k, v in validation['missing_pct'].items() if v > 10}
    if high_missing:
        print(f"\nFeatures with >10% missing:")
        for feat, pct in sorted(high_missing.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat}: {pct:.1f}%")
