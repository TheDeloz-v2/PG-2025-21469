import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def causal_rolling(series, func='mean', window=10, min_periods=1):
    """
    Causal rolling with explicit closed='right' and no centering
    
    Args:
        series (pd.Series): Input time series
        func (str or callable): Aggregation function ('mean', 'std', 'sum', or custom)
        
    Returns:
        pd.Series: Series aligned with original index where value at t uses data <= t
    """
    if func == 'mean':
        return series.rolling(window=window, min_periods=min_periods, closed='right').mean()
    if func == 'std':
        return series.rolling(window=window, min_periods=min_periods, closed='right').std()
    if func == 'sum':
        return series.rolling(window=window, min_periods=min_periods, closed='right').sum()
    return series.rolling(window=window, min_periods=min_periods, closed='right').apply(func)


def create_targets(df, target_col='close', horizons=[1], use_returns=True):
    """
    Create target variables for prediction
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Column to create targets from
        horizons (list): List of prediction horizons
        use_returns (bool): If True, create return-based targets; if False, use raw prices
        
    Returns:
        pd.DataFrame: DataFrame with target columns added
    """
    df = df.copy()
    
    for h in horizons:
        if use_returns:
            # Create return-based target: (price_t+h - price_t) / price_t
            cname = f"target_return_t+{h}"
            df[cname] = df[target_col].pct_change(h).shift(-h)
        else:
            # Create price-based target
            cname = f"target_{target_col}_t+{h}"
            df[cname] = df[target_col].shift(-h)
    
    if use_returns:
        first_target = f"target_return_t+{horizons[0]}"
    else:
        first_target = f"target_{target_col}_t+{horizons[0]}"
    
    df = df.dropna(subset=[first_target])
    return df


def build_lagged_features(returns_df, feature_plan, prefix=''):
    """
    Build lagged return features based on feature_plan
    
    Args:
        returns_df: DataFrame with returns (columns = tickers)
        feature_plan: dict {ticker: [lags]} specifying which lags to create
        prefix: optional prefix for column names (e.g., 'supplier_', 'tech_', 'comp_')
    
    Returns:
        DataFrame with lagged features, aligned to returns_df.index
    """
    lagged_dfs = []
    
    for ticker, lags in feature_plan.items():
        if ticker not in returns_df.columns:
            continue
        
        for lag in lags:
            col_name = f"{prefix}{ticker}_ret_lag{lag}"
            lagged_dfs.append(
                returns_df[ticker].shift(lag).rename(col_name)
            )
    
    if lagged_dfs:
        return pd.concat(lagged_dfs, axis=1)
    else:
        return pd.DataFrame(index=returns_df.index)


def compute_pca_factor(returns_df, company_list, n_components=1):
    """
    Compute PCA factor from a list of companies
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        company_list (list): List of company names
        n_components (int): Number of PCA components
        
    Returns:
        pd.Series: PCA factor series
    """
    available = [c for c in company_list if c in returns_df.columns]
    if not available:
        return pd.Series(index=returns_df.index, dtype=float)
    
    subset = returns_df[available].dropna()
    if subset.shape[0] < 50:
        return pd.Series(index=returns_df.index, dtype=float)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset)
    
    pca = PCA(n_components=n_components)
    factor = pca.fit_transform(scaled)
    
    return pd.Series(factor[:, 0], index=subset.index, name='pca_factor')


def build_final_dataset(
    df_tesla,
    returns_df,
    suppliers_plan,
    tech_plan,
    comp_plan,
    include_pca_factors=True,
    suppliers_pca=None,
    tech_pca=None,
    comp_pca=None,
    market_indices=['NASDAQ', 'S&P 500', 'VIX'],
    missing_threshold=0.6
):
    """
    Construct final dataset combining Tesla features, exogenous lagged returns,
    and optional PCA factors
    
    Returns:
        final_df: DataFrame ready for train/test split and modeling
    """
    
    # Start with Tesla's own features
    final_df = df_tesla.copy()
    
    # 1. Add lagged returns from suppliers
    suppliers_features = build_lagged_features(
        returns_df, suppliers_plan, prefix='supplier_'
    )
    final_df = final_df.join(suppliers_features, how='left')
    
    # 2. Add lagged returns from tech peers
    tech_features = build_lagged_features(
        returns_df, tech_plan, prefix='tech_'
    )
    final_df = final_df.join(tech_features, how='left')
    
    # 3. Add lagged returns from competitors
    comp_features = build_lagged_features(
        returns_df, comp_plan, prefix='comp_'
    )
    final_df = final_df.join(comp_features, how='left')
    
    # 4. Optional: Add PCA factors (already lagged if desired)
    if include_pca_factors:
        if suppliers_pca is not None and not suppliers_pca.empty:
            final_df['supplier_pca_lag1'] = suppliers_pca.shift(1)
            final_df['supplier_pca_lag2'] = suppliers_pca.shift(2)
        
        if tech_pca is not None and not tech_pca.empty:
            final_df['tech_pca_lag1'] = tech_pca.shift(1)
            final_df['tech_pca_lag2'] = tech_pca.shift(2)
        
        if comp_pca is not None and not comp_pca.empty:
            final_df['comp_pca_lag1'] = comp_pca.shift(1)
            final_df['comp_pca_lag2'] = comp_pca.shift(2)
    
    # 5. Add market indices as control features (lagged)
    for idx_name in market_indices:
        if idx_name in returns_df.columns:
            final_df[f'market_{idx_name}_lag1'] = returns_df[idx_name].shift(1)
            final_df[f'market_{idx_name}_lag2'] = returns_df[idx_name].shift(2)
    
    # 6. Handle missing values
    feature_cols = [c for c in final_df.columns 
                   if not c.startswith('target_') and c != 'split']
    
    # Report missingness
    miss_pct = final_df[feature_cols].isna().mean() * 100
    print("\n=== Feature missingness (top 10) ===")
    print(miss_pct.sort_values(ascending=False).head(10))
    
    # Drop rows with excessive NaNs
    thresh = int(missing_threshold * len(feature_cols))
    final_df = final_df.dropna(subset=feature_cols, thresh=thresh)
    
    # Forward-fill conservatively (max 3 steps)
    for col in feature_cols:
        final_df[col] = final_df[col].fillna(method='ffill', limit=3)
    
    # Final drop: any remaining NaNs
    final_df = final_df.dropna(subset=feature_cols)
    
    print(f"\n=== Final dataset ===")
    print(f"Shape: {final_df.shape}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
    
    return final_df
