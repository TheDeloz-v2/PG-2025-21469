from . import config


def extract_target(df, target_col=None):
    """
    Extract target column as separate series
    
    Args:
        df (pd.DataFrame): DataFrame with target column
        target_col (str): Name of the target column (default from config)
    
    Returns:
        pd.Series: Target values
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    return df[target_col].copy()


def get_all_target_columns(df):
    """
    Get all available target columns (for multi-horizon prediction)
    
    Returns:
        dict: Dictionary mapping horizon names to column names
    """
    available_targets = {}
    
    for horizon_name, horizon_days in config.TARGET_HORIZONS.items():
        target_col = f"target_return_t+{horizon_days}"
        if target_col in df.columns:
            available_targets[horizon_name] = target_col
        else:
            print(f"Warning: Target column '{target_col}' not found for {horizon_name} horizon")
    
    return available_targets


def create_market_only_dataset(full_df):
    """
    Dataset A: Market features + target
    
    Returns:
        pd.DataFrame: Market features with target
    """
    print("\n" + "="*60)
    print("Creating Dataset A: Market Features Only")
    print("="*60)
    
    # Get sentiment columns to exclude
    sentiment_cols = config.EXCLUDE_COLUMNS.get('sentiment', [])
    
    # Stock-specific keywords to exclude
    stock_keywords = ['rsi', 'macd', 'bb_', 'atr', 'trend', 'seasonal', 
                     'residual', 'wavelet', 'close', 'volume', 'high', 'low', 
                     'open', 'daily_return', 'weekly_return', 'monthly_return',
                     'volatility', 'true_range']
    
    # Keep market features: lagged returns, PCA, correlations, Granger, indices
    market_cols = [c for c in full_df.columns 
                  if c not in sentiment_cols 
                  and not any(keyword in c.lower() for keyword in stock_keywords)
                  or c == config.TARGET_COLUMN]
    
    dataset_a = full_df[market_cols].copy()
    
    # Drop any remaining NaN (should be minimal after alignment)
    dataset_a = dataset_a.dropna()
    
    print(f"✓ Shape: {dataset_a.shape}")
    print(f"  Features: {dataset_a.shape[1] - 1} (+ target)")
    print(f"  Samples: {len(dataset_a)}")
    print(f"  Date range: {dataset_a.index.min()} to {dataset_a.index.max()}")
    
    return dataset_a


def create_sentiment_only_dataset(full_df):
    """
    Dataset B: Sentiment features + target
    
    Returns:
        pd.DataFrame: Sentiment features with target
    """
    print("\n" + "="*60)
    print("Creating Dataset B: Sentiment Features Only")
    print("="*60)
    
    # Get sentiment columns from config
    sentiment_cols = config.EXCLUDE_COLUMNS.get('sentiment', [])
    
    if not sentiment_cols:
        # Fallback: detect by name
        sentiment_cols = [c for c in full_df.columns 
                         if 'sentiment' in c.lower() or 'acceptance' in c.lower()]
        print(f"⚠ Auto-detected sentiment columns: {sentiment_cols}")
    
    # Add target
    keep_cols = sentiment_cols + [config.TARGET_COLUMN]
    keep_cols = [c for c in keep_cols if c in full_df.columns]
    
    dataset_b = full_df[keep_cols].copy()
    
    # Drop rows with missing sentiment OR target
    dataset_b = dataset_b.dropna()
    
    print(f"✓ Shape: {dataset_b.shape}")
    print(f"  Features: {dataset_b.shape[1] - 1} (+ target)")
    print(f"  Samples: {len(dataset_b)}")
    print(f"  Date range: {dataset_b.index.min()} to {dataset_b.index.max()}")
    
    if len(dataset_b) < config.MIN_SAMPLES:
        print(f"⚠ WARNING: Only {len(dataset_b)} samples (min: {config.MIN_SAMPLES})")
    
    return dataset_b


def create_stock_only_dataset(full_df):
    """
    Dataset C: Stock features + target
    
    Returns:
        pd.DataFrame: Stock technical features with target
    """
    print("\n" + "="*60)
    print("Creating Dataset C: Stock Features Only")
    print("="*60)
    
    # Get sentiment columns to exclude
    sentiment_cols = config.EXCLUDE_COLUMNS.get('sentiment', [])
    
    # Stock columns are all columns that are NOT:
    # - target
    # - sentiment features
    # - market-specific features (lagged returns, PCA, granger, etc)
    market_keywords = ['_return_lag', '_pca_', 'granger_', 'corr_', 'supplier', 
                      'tech', 'comp', 'index', 'market', 'sp500', 'nasdaq', 'dow']
    
    stock_cols = [c for c in full_df.columns 
                 if c != config.TARGET_COLUMN 
                 and c not in sentiment_cols
                 and not any(keyword in c.lower() for keyword in market_keywords)]
    
    # Add target
    keep_cols = stock_cols + [config.TARGET_COLUMN]
    
    dataset_c = full_df[keep_cols].copy()
    
    # Drop rows with missing stock OR target
    dataset_c = dataset_c.dropna()
    
    print(f"✓ Shape: {dataset_c.shape}")
    print(f"  Features: {dataset_c.shape[1] - 1} (+ target)")
    print(f"  Samples: {len(dataset_c)}")
    print(f"  Date range: {dataset_c.index.min()} to {dataset_c.index.max()}")
    
    return dataset_c


def create_full_fusion_dataset(full_df):
    """
    Dataset D: All features (market + sentiment + stock) + target
    
    Returns:
        pd.DataFrame: Complete feature set with target
    """
    print("\n" + "="*60)
    print("Creating Dataset D: Full Fusion (All Features)")
    print("="*60)
    
    # Use complete aligned dataset, drop rows with missing target
    dataset_d = full_df[full_df[config.TARGET_COLUMN].notna()].copy()
    
    # Option: Drop rows with >30% missing features
    feature_cols = [c for c in dataset_d.columns if c != config.TARGET_COLUMN]
    missing_pct = dataset_d[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    dataset_d = dataset_d[missing_pct < 0.3].copy()
    
    print(f"✓ Shape: {dataset_d.shape}")
    print(f"  Features: {dataset_d.shape[1] - 1} (+ target)")
    print(f"  Samples: {len(dataset_d)}")
    print(f"  Date range: {dataset_d.index.min()} to {dataset_d.index.max()}")
    
    # Report missing values per feature
    missing_counts = dataset_d.isnull().sum()
    high_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        print(f"\n  Features with missing values (top 10):")
        for feat, count in high_missing.head(10).items():
            pct = count / len(dataset_d) * 100
            print(f"    {feat}: {count} ({pct:.1f}%)")
    else:
        print("\n  ✓ No missing values!")
    
    return dataset_d


def generate_all_datasets(aligned_df):
    """
    Generate all 4 datasets (A, B, C, D) from aligned DataFrame
    
    Returns:
        dict: Dictionary with keys 'market', 'sentiment', 'stock', 'fusion'
    """
    print("\n" + "="*70)
    print("GENERATING 4 DATASETS FOR MODEL COMPARISON")
    print("="*70)
    
    datasets = {}
    
    # Dataset A: Market only
    datasets['market'] = create_market_only_dataset(aligned_df)
    
    # Dataset B: Sentiment only
    datasets['sentiment'] = create_sentiment_only_dataset(aligned_df)
    
    # Dataset C: Stock only
    datasets['stock'] = create_stock_only_dataset(aligned_df)
    
    # Dataset D: Full fusion
    datasets['fusion'] = create_full_fusion_dataset(aligned_df)
    
    # Summary
    print("\n" + "="*70)
    print("DATASET GENERATION SUMMARY")
    print("="*70)
    for name, df in datasets.items():
        print(f"{name.upper():12} : {df.shape[0]:4} samples × {df.shape[1]:3} features")
    
    return datasets


def validate_targets_consistency(datasets):
    """
    Ensure all datasets have valid targets with same target column
    
    Returns:
        dict: Validation report
    """
    print("\n" + "="*60)
    print("Target Consistency Check")
    print("="*60)
    
    report = {}
    
    for name, df in datasets.items():
        # Check target exists
        has_target = config.TARGET_COLUMN in df.columns
        
        # Check no missing targets
        if has_target:
            target_missing = df[config.TARGET_COLUMN].isnull().sum()
            target_valid = target_missing == 0
        else:
            target_valid = False
        
        report[name] = {
            'has_target': has_target,
            'target_column': config.TARGET_COLUMN if has_target else None,
            'n_samples': len(df),
            'target_valid': target_valid,
            'status': '✓ PASS' if (has_target and target_valid) else '✗ FAIL'
        }
        
        print(f"{name.upper():12}: {report[name]['status']} ({report[name]['n_samples']} samples)")
    
    all_pass = all(r['status'] == '✓ PASS' for r in report.values())
    
    print(f"\n{'='*60}")
    print(f"Overall: {'✓ ALL PASS' if all_pass else '✗ SOME FAILURES'}")
    print(f"{'='*60}")
    
    return report


def generate_datasets_for_horizon(aligned_df, horizon_name, target_col):
    """
    Generate all 4 datasets (market, sentiment, stock, fusion) for a specific prediction horizon
    
    Args:
        aligned_df: Aligned DataFrame with all features
        horizon_name: Name of horizon ('short', 'medium', 'long')
        target_col: Target column name (e.g., 'target_return_t+1')
    
    Returns:
        dict: Dictionary with keys 'market', 'sentiment', 'stock', 'fusion'
    """
    print("\n" + "="*70)
    print(f"GENERATING DATASETS FOR {horizon_name.upper()} HORIZON ({target_col})")
    print("="*70)
    
    datasets = {}
    
    # Get sentiment columns to exclude
    sentiment_cols = config.EXCLUDE_COLUMNS.get('sentiment', [])
    
    # Stock-specific keywords to exclude for market dataset
    stock_keywords = ['rsi', 'macd', 'bb_', 'atr', 'trend', 'seasonal', 
                     'residual', 'wavelet', 'close', 'volume', 'high', 'low', 
                     'open', 'daily_return', 'weekly_return', 'monthly_return',
                     'volatility', 'true_range']
    
    # Market keywords to exclude for stock dataset
    market_keywords = ['_return_lag', '_pca_', 'granger_', 'corr_', 'supplier', 
                      'tech', 'comp', 'index', 'market', 'sp500', 'nasdaq', 'dow']
    
    # Dataset A: Market only
    market_cols = [c for c in aligned_df.columns 
                  if c not in sentiment_cols 
                  and not any(keyword in c.lower() for keyword in stock_keywords)
                  or c == target_col]
    datasets['market'] = aligned_df[market_cols].dropna(subset=[target_col])
    print(f"  MARKET      : {datasets['market'].shape[0]:4} samples × {datasets['market'].shape[1]:3} features")
    
    # Dataset B: Sentiment only
    keep_cols = sentiment_cols + [target_col]
    keep_cols = [c for c in keep_cols if c in aligned_df.columns]
    datasets['sentiment'] = aligned_df[keep_cols].dropna(subset=[target_col])
    print(f"  SENTIMENT   : {datasets['sentiment'].shape[0]:4} samples × {datasets['sentiment'].shape[1]:3} features")
    
    # Dataset C: Stock only
    stock_cols = [c for c in aligned_df.columns 
                 if c != target_col 
                 and c not in sentiment_cols
                 and not any(keyword in c.lower() for keyword in market_keywords)]
    keep_cols = stock_cols + [target_col]
    datasets['stock'] = aligned_df[keep_cols].dropna(subset=[target_col])
    print(f"  STOCK       : {datasets['stock'].shape[0]:4} samples × {datasets['stock'].shape[1]:3} features")
    
    # Dataset D: Full fusion
    datasets['fusion'] = aligned_df[aligned_df[target_col].notna()].copy()
    feature_cols = [c for c in datasets['fusion'].columns if c != target_col]
    missing_pct = datasets['fusion'][feature_cols].isnull().sum(axis=1) / len(feature_cols)
    datasets['fusion'] = datasets['fusion'][missing_pct < 0.3].copy()
    print(f"  FUSION      : {datasets['fusion'].shape[0]:4} samples × {datasets['fusion'].shape[1]:3} features")
    
    return datasets


def generate_all_multi_horizon_datasets(aligned_df):
    """
    Generate datasets for all prediction horizons
    
    Returns:
        dict: Nested dictionary {horizon_name: {dataset_type: dataframe}}
    """
    print("\n" + "="*70)
    print("GENERATING MULTI-HORIZON DATASETS")
    print("="*70)
    
    all_datasets = {}
    available_targets = get_all_target_columns(aligned_df)
    
    if not available_targets:
        print("ERROR: No target columns found in DataFrame!")
        print(f"Available columns: {list(aligned_df.columns)}")
        return {}
    
    for horizon_name, target_col in available_targets.items():
        all_datasets[horizon_name] = generate_datasets_for_horizon(
            aligned_df, horizon_name, target_col
        )
    
    print("\n" + "="*70)
    print("MULTI-HORIZON DATASET GENERATION SUMMARY")
    print("="*70)
    for horizon_name, datasets in all_datasets.items():
        horizon_days = config.TARGET_HORIZONS[horizon_name]
        print(f"\n{horizon_name.upper()} ({horizon_days} days):")
        for dataset_name, df in datasets.items():
            print(f"  {dataset_name:12} : {df.shape[0]:4} samples × {df.shape[1]:3} features")
    
    return all_datasets
