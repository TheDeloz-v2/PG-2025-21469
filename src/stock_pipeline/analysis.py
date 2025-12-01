import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.feature_selection import mutual_info_regression


def perform_stationarity_tests(series, title="Series"):
    """
    Perform both ADF and KPSS tests to check for stationarity
    
    Args:
        series (pd.Series): Time series data
        title (str): Title for the series being tested
    
    Returns:
        dict: Test results with statistics and conclusions
    """
    results = {'title': title}
    
    # ADF Test
    adf_result = adfuller(series)
    results['adf'] = {
        'statistic': adf_result[0],
        'pvalue': adf_result[1],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05
    }
    
    # KPSS Test
    kpss_result = kpss(series)
    results['kpss'] = {
        'statistic': kpss_result[0],
        'pvalue': kpss_result[1],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] >= 0.05
    }
    
    return results


def calculate_correlation_significance(df, target_col='close'):
    """
    Calculate correlations and their statistical significance

    Args:
        df (pd.DataFrame): DataFrame containing features and target variable
        target_col (str): Name of the target column

    Returns:
        pd.DataFrame: Correlations and p-values sorted by correlation strength
    """
    correlations = {}
    p_values = {}
    
    for column in df.columns:
        if column != target_col:
            try:
                correlation, p_value = stats.pearsonr(df[column], df[target_col])
                correlations[column] = correlation
                p_values[column] = p_value
            except:
                pass
    
    results = pd.DataFrame({
        'Correlation': correlations,
        'P-value': p_values
    })
    
    return results.sort_values('Correlation', ascending=False, key=abs)


def assess_feature_importance(X, y):
    """
    Assess feature importance using mutual information
    
    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.Series): Target variable series
    
    Returns:
        pd.DataFrame: Feature importance scores
    """
    mi_scores = mutual_info_regression(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': mi_scores
    })
    
    return importance_df.sort_values('Importance', ascending=False)


def print_stationarity_report(results):
    """Print formatted stationarity test results"""
    print(f"\n{'='*60}")
    print(f"Stationarity Tests for {results['title']}")
    print(f"{'='*60}")
    
    print('\nAugmented Dickey-Fuller Test:')
    print(f"  ADF Statistic: {results['adf']['statistic']:.4f}")
    print(f"  p-value: {results['adf']['pvalue']:.4f}")
    print(f"  Conclusion: {'Stationary' if results['adf']['is_stationary'] else 'Non-stationary'}")
    
    print('\nKwiatkowski-Phillips-Schmidt-Shin Test:')
    print(f"  KPSS Statistic: {results['kpss']['statistic']:.4f}")
    print(f"  p-value: {results['kpss']['pvalue']:.4f}")
    print(f"  Conclusion: {'Stationary' if results['kpss']['is_stationary'] else 'Non-stationary'}")


def print_daily_returns_stats(stats):
    """Print formatted daily returns statistics"""
    print("\n" + "="*60)
    print("DAILY RETURN STATISTICS")
    print("="*60)
    print(f"Average daily return: {stats['mean']:.4f}%")
    print(f"Volatility (std dev): {stats['std']:.4f}%")
    print(f"Maximum daily return: {stats['max']:.4f}%")
    print(f"Minimum daily return: {stats['min']:.4f}%")
    print(f"Skewness: {stats['skewness']:.4f}")
    print(f"Kurtosis: {stats['kurtosis']:.4f}")
