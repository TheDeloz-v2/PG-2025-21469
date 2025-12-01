from . import config


def align_by_date(market_df, sentiment_df, stock_df):
    """
    Align all datasets using market dates as base (left join)
    
    Args:
        market_df: Market features with target
        sentiment_df: Sentiment scores
        stock_df: Stock technical features
    
    Returns:
        pd.DataFrame: Aligned dataset with market dates as index
    """
    print("\n" + "="*60)
    print("Aligning datasets by date...")
    print("="*60)
    
    # Market is the base (has the target)
    aligned = market_df.copy()
    print(f"✓ Base (market): {aligned.shape[0]} days")
    
    # Left join sentiment (will have NaN gaps)
    aligned = aligned.join(sentiment_df, how='left', rsuffix='_sent')
    print(f"✓ After sentiment join: {aligned.shape}")
    sentiment_missing = aligned[sentiment_df.columns].isnull().any(axis=1).sum()
    print(f"  → {sentiment_missing} days without sentiment ({sentiment_missing/len(aligned)*100:.1f}%)")
    
    # Left join stock (should align well - trading days)
    aligned = aligned.join(stock_df, how='left', rsuffix='_stock')
    print(f"✓ After stock join: {aligned.shape}")
    stock_missing = aligned[stock_df.columns].isnull().any(axis=1).sum()
    print(f"  → {stock_missing} days without stock data ({stock_missing/len(aligned)*100:.1f}%)")
    
    return aligned


def forward_fill_sentiment(df, sentiment_columns, max_days=None):
    """
    Forward-fill sentiment scores up to max_days
    
    Args:
        df: Aligned DataFrame
        sentiment_columns: List of sentiment column names
        max_days: Maximum days to forward-fill (default from config)
    
    Returns:
        pd.DataFrame: DataFrame with filled sentiment
    """
    if max_days is None:
        max_days = config.MAX_FILL_DAYS
    
    print(f"\nForward-filling sentiment (max {max_days} days)...")
    
    filled = df.copy()
    
    for col in sentiment_columns:
        if col in filled.columns:
            before_missing = filled[col].isnull().sum()
            
            # Forward fill with limit
            filled[col] = filled[col].fillna(method='ffill', limit=max_days)
            
            after_missing = filled[col].isnull().sum()
            filled_count = before_missing - after_missing
            
            print(f"  {col}: filled {filled_count} gaps, {after_missing} still missing")
    
    return filled


def handle_stock_gaps(df, stock_columns, method='drop'):
    """
    Handle missing stock data (non-trading days)
    
    Args:
        df: Aligned DataFrame
        stock_columns: List of stock column names
        method: 'drop' or 'ffill'
    
    Returns:
        pd.DataFrame: DataFrame with handled gaps
    """
    print(f"\nHandling stock data gaps (method: {method})...")
    
    if method == 'drop':
        # Drop rows where ALL stock features are missing
        before = len(df)
        stock_mask = df[stock_columns].isnull().all(axis=1)
        df_clean = df[~stock_mask].copy()
        dropped = before - len(df_clean)
        print(f"✓ Dropped {dropped} rows with no stock data")
        return df_clean
    
    elif method == 'ffill':
        # Forward fill stock features (typically 1-2 days for weekends/holidays)
        df_filled = df.copy()
        for col in stock_columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(method='ffill', limit=3)
        return df_filled
    
    return df


def validate_alignment(df):
    """
    Validate alignment quality
    
    Returns:
        dict: Validation metrics
    """
    print("\n" + "="*60)
    print("Alignment Validation")
    print("="*60)
    
    # Check target column
    target_missing = df[config.TARGET_COLUMN].isnull().sum()
    print(f"Target ({config.TARGET_COLUMN}):")
    print(f"  Valid samples: {len(df) - target_missing}")
    print(f"  Missing: {target_missing} ({target_missing/len(df)*100:.1f}%)")
    
    # Overall feature coverage
    feature_cols = [c for c in df.columns if c != config.TARGET_COLUMN]
    coverage = (~df[feature_cols].isnull()).mean()
    print(f"\nFeature coverage:")
    print(f"  Average: {coverage.mean()*100:.1f}%")
    print(f"  Min: {coverage.min()*100:.1f}%")
    print(f"  Features with <50% coverage: {(coverage < 0.5).sum()}")
    
    # Check minimum sample requirement
    valid_samples = df[config.TARGET_COLUMN].notna().sum()
    min_samples_ok = valid_samples >= config.MIN_SAMPLES
    print(f"\nSample count check:")
    print(f"  Valid samples: {valid_samples}")
    print(f"  Minimum required: {config.MIN_SAMPLES}")
    print(f"  Status: {'✓ PASS' if min_samples_ok else '✗ FAIL'}")
    
    return {
        'total_samples': len(df),
        'valid_target_samples': valid_samples,
        'target_missing_pct': target_missing/len(df)*100,
        'avg_feature_coverage': coverage.mean()*100,
        'min_feature_coverage': coverage.min()*100,
        'meets_min_samples': min_samples_ok,
        'features_below_threshold': (coverage < config.MIN_FEATURE_COVERAGE).sum()
    }


def drop_low_coverage_features(df, min_coverage=None):
    """
    Drop features with coverage below threshold
    
    Args:
        df: Aligned DataFrame
        min_coverage: Minimum coverage threshold (default from config)
    
    Returns:
        pd.DataFrame: DataFrame with low-coverage features removed
    """
    if min_coverage is None:
        min_coverage = config.MIN_FEATURE_COVERAGE
    
    print(f"\nChecking feature coverage (min: {min_coverage*100:.0f}%)...")
    
    feature_cols = [c for c in df.columns if c != config.TARGET_COLUMN]
    coverage = (~df[feature_cols].isnull()).sum() / len(df)
    
    low_coverage = coverage[coverage < min_coverage]
    
    if len(low_coverage) > 0:
        print(f"⚠ Dropping {len(low_coverage)} features with low coverage:")
        for feat, cov in low_coverage.items():
            print(f"  {feat}: {cov*100:.1f}%")
        
        df_clean = df.drop(columns=low_coverage.index)
        return df_clean
    else:
        print("✓ All features meet coverage threshold")
        return df


def drop_incomplete_samples(df):
    """
    Drop samples with missing target or excessive missing features
    
    Returns:
        pd.DataFrame: Clean dataset
    """
    print("\nRemoving incomplete samples...")
    
    before = len(df)
    
    # Drop rows with missing target
    df_clean = df[df[config.TARGET_COLUMN].notna()].copy()
    target_drops = before - len(df_clean)
    print(f"  Dropped {target_drops} samples with missing target")
    
    # Drop rows with >50% missing features
    feature_cols = [c for c in df_clean.columns if c != config.TARGET_COLUMN]
    missing_pct = df_clean[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    df_clean = df_clean[missing_pct < 0.5].copy()
    feature_drops = before - target_drops - len(df_clean)
    print(f"  Dropped {feature_drops} samples with >50% missing features")
    
    print(f"✓ Final dataset: {len(df_clean)} samples ({len(df_clean)/before*100:.1f}% retained)")
    
    return df_clean


def align_datasets_to_common_dates(datasets_dict):
    """
    Align all datasets to have the same date range (intersection)
    
    This ensures fair comparison between models by using the exact same
    time period and number of samples across all datasets.
    
    Args:
        datasets_dict: Dictionary of DataFrames with date indices
                      e.g., {'market': df1, 'sentiment': df2, 'stock': df3, 'fusion': df4}
    
    Returns:
        dict: Dictionary with aligned datasets (same date range)
    """
    print("\n" + "="*70)
    print("ALIGNING ALL DATASETS TO COMMON DATE RANGE")
    print("="*70)
    
    if not datasets_dict:
        print("No datasets provided")
        return datasets_dict
    
    # Step 1: Find date range for each dataset
    print("\nOriginal date ranges:")
    date_ranges = {}
    for name, df in datasets_dict.items():
        if len(df) == 0:
            print(f"  {name.upper():12}: EMPTY DATASET")
            continue
        start = df.index.min()
        end = df.index.max()
        date_ranges[name] = (start, end)
        print(f"  {name.upper():12}: {start.date()} to {end.date()} ({len(df)} samples)")
    
    if not date_ranges:
        print("All datasets are empty")
        return datasets_dict
    
    # Step 2: Find common date range (intersection)
    all_starts = [start for start, _ in date_ranges.values()]
    all_ends = [end for _, end in date_ranges.values()]
    
    common_start = max(all_starts)
    common_end = min(all_ends)
    
    print("\n" + "="*70)
    print(f"Common date range: {common_start.date()} to {common_end.date()}")
    print("="*70)
    
    if common_start > common_end:
        print("ERROR: No overlapping dates between datasets!")
        for name, (start, end) in date_ranges.items():
            print(f"  {name}: {start.date()} to {end.date()}")
        raise ValueError("Datasets have no overlapping date range")
    
    # Step 3: Trim all datasets to common range
    aligned_datasets = {}
    print("\nAligned datasets:")
    
    for name, df in datasets_dict.items():
        if len(df) == 0:
            aligned_datasets[name] = df
            continue
        
        # Filter to common date range
        mask = (df.index >= common_start) & (df.index <= common_end)
        df_aligned = df.loc[mask].copy()
        
        aligned_datasets[name] = df_aligned
        
        original_samples = len(df)
        aligned_samples = len(df_aligned)
        removed = original_samples - aligned_samples
        
        print(f"  {name.upper():12}: {aligned_samples:4} samples (removed {removed})")
    
    # Step 4: Verify all have same dates
    print("\n" + "="*70)
    print("Verification:")
    print("="*70)
    
    all_same_count = len(set(len(df) for df in aligned_datasets.values() if len(df) > 0))
    
    for name, df in aligned_datasets.items():
        if len(df) == 0:
            continue
        print(f"  {name.upper():12}: {len(df):4} samples | {df.index.min().date()} to {df.index.max().date()}")
    
    if all_same_count == 1:
        print("\nSUCCESS: All datasets have the same number of samples!")
    else:
        print(f"\nWARNING: Datasets have different sample counts")
        print("This may be due to NaN handling. Consider using dropna() consistently.")
    
    return aligned_datasets


def create_aligned_dataset(market_df, sentiment_df, stock_df, 
                          fill_sentiment=True, handle_stock='drop',
                          drop_low_coverage=True, clean_samples=True):
    """
    Full alignment pipeline
    
    Returns:
        pd.DataFrame: Fully aligned and cleaned dataset
    """
    print("\n" + "="*70)
    print("FUSION PIPELINE: DATASET ALIGNMENT")
    print("="*70)
    
    # Step 1: Temporal alignment
    aligned = align_by_date(market_df, sentiment_df, stock_df)
    
    # Step 2: Identify column groups
    sentiment_cols = [c for c in sentiment_df.columns if c in aligned.columns]
    stock_cols = [c for c in stock_df.columns if c in aligned.columns]
    
    # Step 3: Fill sentiment gaps
    if fill_sentiment:
        aligned = forward_fill_sentiment(aligned, sentiment_cols)
    
    # Step 4: Handle stock gaps
    aligned = handle_stock_gaps(aligned, stock_cols, method=handle_stock)
    
    # Step 5: Validate
    validation = validate_alignment(aligned)
    
    # Step 6: Drop low-coverage features
    if drop_low_coverage:
        aligned = drop_low_coverage_features(aligned)
    
    # Step 7: Clean incomplete samples
    if clean_samples:
        aligned = drop_incomplete_samples(aligned)
    
    print("\n" + "="*70)
    print("ALIGNMENT COMPLETE")
    print("="*70)
    print(f"Final shape: {aligned.shape}")
    print(f"Date range: {aligned.index.min()} to {aligned.index.max()}")
    print(f"Valid samples with target: {aligned[config.TARGET_COLUMN].notna().sum()}")
    
    return aligned
