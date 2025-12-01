import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.seasonal import seasonal_decompose
from . import config


def build_classical_decomposition(df, price_col='close', model=None, period=None):
    """
    Build classical time series decomposition
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        price_col (str): Column name for price data
        model (str): 'additive' or 'multiplicative'
        period (int): Seasonal period
        
    Returns:
        pd.DataFrame: DataFrame with trend, seasonal, and residual components
    """
    if model is None:
        model = config.CLASSICAL_DECOMPOSITION['model']
    if period is None:
        period = config.CLASSICAL_DECOMPOSITION['period']
    
    df = df.copy()
    df = df.asfreq("B")  # Business day frequency
    
    ts = df[price_col].interpolate()
    
    decomp = seasonal_decompose(ts, model=model, period=period)
    
    df['trend'] = decomp.trend
    df['seasonal'] = decomp.seasonal
    df['resid'] = decomp.resid
    
    print(f"\n✓ Classical decomposition completed (model={model}, period={period})")
    
    return df


def build_wavelet_decomposition(df, price_col='close', wavelet=None, level=None):
    """
    Build wavelet decomposition with proper reconstruction
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        price_col (str): Column name for price data
        wavelet (str): Wavelet type
        level (int): Decomposition level
        
    Returns:
        pd.DataFrame: DataFrame with wavelet components
    """
    if wavelet is None:
        wavelet = config.WAVELET_DECOMPOSITION['wavelet']
    if level is None:
        level = config.WAVELET_DECOMPOSITION['level']
    
    df = df.copy()
    df = df.asfreq("B")
    ts = df[price_col].interpolate()
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(ts.values, wavelet, level=level)
    
    # Initialize reconstructed signals
    reconstructed = pd.DataFrame(index=ts.index)
    
    # Reconstruct approximation (lowest frequency)
    coeff_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    reconstructed['wavelet_approx'] = pywt.waverec(coeff_list, wavelet)[:len(ts)]
    
    # Reconstruct details (higher frequencies)
    for i in range(1, len(coeffs)):
        coeff_list = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
        coeff_list[i] = coeffs[i]
        detail = pywt.waverec(coeff_list, wavelet)[:len(ts)]
        reconstructed[f'wavelet_detail_{level-i+1}'] = detail
    
    # Verify reconstruction
    reconstruction = reconstructed['wavelet_approx'].copy()
    for i in range(level):
        reconstruction += reconstructed[f'wavelet_detail_{i+1}']
    
    reconstruction_error = np.mean(np.abs(ts.values - reconstruction))
    print(f"\n✓ Wavelet decomposition completed (wavelet={wavelet}, level={level})")
    print(f"  Mean reconstruction error: {reconstruction_error:.10f}")
    
    return reconstructed


def combine_decompositions(df_original, df_classical, df_wavelet):
    """
    Combine original data with decomposition components
    
    Args:
        df_original (pd.DataFrame): Original DataFrame
        df_classical (pd.DataFrame): DataFrame with classical decomposition components
        df_wavelet (pd.DataFrame): DataFrame with wavelet decomposition components
    
    Returns:
        pd.DataFrame: Combined DataFrame with all components
    """
    features = df_original.copy()
    
    # Classical decomposition
    if 'trend' in df_classical.columns:
        features['trend'] = df_classical['trend']
    if 'seasonal' in df_classical.columns:
        features['seasonal'] = df_classical['seasonal']
    if 'resid' in df_classical.columns:
        features['residual'] = df_classical['resid']
    
    # Wavelet decomposition
    for col in df_wavelet.columns:
        if col != 'original':
            features[col] = df_wavelet[col]
    
    return features
