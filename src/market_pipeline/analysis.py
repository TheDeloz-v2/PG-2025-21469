import pandas as pd
import numpy as np
import warnings
import io
import contextlib
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


def _run_granger_simple(y, x, maxlag=5, min_obs=60):
    """
    Simple Granger causality test wrapper
    
    Args:
        y (pd.Series): Target time series
        x (pd.Series): Predictor time series
        maxlag (int): Maximum lag to test
        min_obs (int): Minimum observations required to run test
        
    Returns:
        dict: p-values for each lag tested
    """
    df = (x.to_frame().join(y.to_frame(), how='inner')).dropna()
    if df.shape[0] < max(min_obs, maxlag + 10):
        return {}
    arr = df[[y.name, x.name]].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = grangercausalitytests(arr, maxlag=maxlag, verbose=False)
    pvals = {f'lag_{l}': float(res[l][0]['ssr_ftest'][1]) for l in range(1, maxlag+1)}
    return pvals


def analyze_and_select_from_summary(
    suppliers_summary,
    returns_df=None,
    target_col='Tesla',
    pearson_thresh=0.08,
    rolling_mean_thresh=0.06,
    rolling_std_max=0.17,
    min_obs_required=100,
    granger_maxlag=5,
    granger_pval_thresh=0.05,
    require_granger=False,
    default_lags=[1,2]
):
    """
    Select suppliers using precomputed suppliers_summary
    
    Returns:
      summary_out: suppliers_summary copy extended with decision columns
      selected: list of tickers to keep
      feature_plan: dict ticker -> list of lags to include
    """
    df = suppliers_summary.copy()
    
    # ensure required columns exist
    for c in ['n','pearson','rolling_mean','rolling_std']:
        if c not in df.columns:
            df[c] = np.nan

    df['meets_strength'] = df['pearson'].abs() >= pearson_thresh
    df['meets_rolling'] = (df['rolling_mean'] >= rolling_mean_thresh) & (df['rolling_std'] <= rolling_std_max)
    df['enough_obs'] = df['n'] >= min_obs_required

    df['preselect'] = df['enough_obs'] & (df['meets_strength'] | df['meets_rolling'])

    # If returns_df provided and refinement requested, run granger for candidates
    df['granger_sig'] = False
    granger_dict = {}
    if returns_df is not None:
        for comp in df.index:
            if comp not in returns_df.columns or target_col not in returns_df.columns:
                continue
            pvals = _run_granger_simple(returns_df[target_col], returns_df[comp], maxlag=granger_maxlag, min_obs=60)
            if pvals:
                sig_lags = [int(k.split('_')[1]) for k,v in pvals.items() if v < granger_pval_thresh]
                if sig_lags:
                    df.at[comp, 'granger_sig'] = True
                granger_dict[comp] = pvals

    # Final decision
    selected = []
    feature_plan = {}
    for comp in df.index:
        if not df.at[comp, 'enough_obs']:
            df.at[comp, 'selected'] = False
            continue

        if require_granger:
           keep = bool(df.at[comp, 'granger_sig'])
        else:
            keep = bool(df.at[comp, 'preselect']) or bool(df.at[comp, 'granger_sig'])

        df.at[comp, 'selected'] = bool(keep)

        if keep:
            lags = []
            if comp in granger_dict:
                lags = sorted([int(k.split('_')[1]) for k,v in granger_dict[comp].items() if v < granger_pval_thresh])
            if not lags:
                lags = default_lags.copy()
            feature_plan[comp] = lags
            selected.append(comp)

    df.attrs['granger_pvals'] = granger_dict

    return df, selected, feature_plan


def analyze_and_select_tech_peers(
    summary_correlations,
    summary_ols,
    returns_df=None,
    target_col='Tesla',
    pearson_thresh=0.3,
    partial_thresh=0.2,
    rolling_partial_mean_thresh=0.2,
    rolling_partial_std_max=0.15,
    pval_company_thresh=0.05,
    granger_maxlag=5,
    granger_pval_thresh=0.05,
    require_granger=False,
    default_lags=[1,2]
):
    """
    Analyze tech peers vs Tesla using Section B summaries and optional Granger test

    Returns:
      summary_out: combined DataFrame with decision columns
      selected: list of tickers kept as exogenous features
      feature_plan: dict {ticker: [lags]} of lagged-return features
    """
    df_corr = summary_correlations.copy()
    df_ols = summary_ols.copy()
    merged = df_corr.join(df_ols, how='outer', lsuffix='_corr', rsuffix='_ols')

    # basic criteria
    merged['meets_corr'] = merged['pearson'].abs() >= pearson_thresh
    merged['meets_partial'] = merged['partial_nasdaq'].abs() >= partial_thresh
    merged['meets_rolling'] = (merged['rolling_partial_mean'] >= rolling_partial_mean_thresh) & \
                              (merged['rolling_partial_std'] <= rolling_partial_std_max)
    merged['ols_significant'] = merged['pval_company'] < pval_company_thresh

    # optional granger refinement
    granger_dict = {}
    merged['granger_sig'] = False
    if returns_df is not None:
        for comp in merged.index:
            if comp not in returns_df.columns or target_col not in returns_df.columns:
                continue
            pvals = _run_granger_simple(returns_df[target_col], returns_df[comp],
                                        maxlag=granger_maxlag, min_obs=60)
            if pvals:
                sig_lags = [int(k.split('_')[1]) for k,v in pvals.items() if v < granger_pval_thresh]
                if sig_lags:
                    merged.at[comp, 'granger_sig'] = True
                granger_dict[comp] = pvals

    # decision logic
    selected, feature_plan = [], {}
    for comp in merged.index:
        cond_corr = merged.at[comp,'meets_corr'] or merged.at[comp,'meets_partial'] or merged.at[comp,'meets_rolling']
        cond_ols = merged.at[comp,'ols_significant']
        cond_granger = merged.at[comp,'granger_sig']

        if require_granger:
            keep = bool(cond_granger)
        else:
            keep = bool(cond_corr or cond_ols or cond_granger)

        merged.at[comp,'selected'] = keep
        if keep:
            selected.append(comp)
            sig_lags = []
            if comp in granger_dict:
                sig_lags = [int(k.split('_')[1]) for k,v in granger_dict[comp].items() if v < granger_pval_thresh]
            feature_plan[comp] = sorted(sig_lags) if sig_lags else default_lags

    merged.attrs['granger_pvals'] = granger_dict
    return merged, selected, feature_plan


def analyze_and_select_competitors(
    summary_df,
    returns_df=None,
    target_col='Tesla',
    pearson_thresh=0.10,
    rolling_mean_thresh=0.07,
    rolling_std_max=0.15,
    min_obs_required=100,
    granger_maxlag=5,
    granger_pval_thresh=0.05,
    require_granger=False,
    default_lags=[1, 2]
):
    """
    Evaluate competitors summary and select which to include as features
    
    Returns:
        summary_out: competitors_summary copy extended with decision columns
        selected: list of tickers to keep
        feature_plan: dict ticker -> list of lags to include
    """
    df = summary_df.copy()

    # Guard against missing expected columns
    for col in ['n_obs','pearson','rolling_corr_mean','rolling_corr_std']:
        if col not in df.columns:
            df[col] = np.nan

    df['enough_obs'] = df['n_obs'] >= min_obs_required
    df['meets_corr'] = df['pearson'].abs() >= pearson_thresh
    df['meets_rolling'] = (df['rolling_corr_mean'] >= rolling_mean_thresh) & (df['rolling_corr_std'] <= rolling_std_max)
    df['preselect'] = df['enough_obs'] & (df['meets_corr'] | df['meets_rolling'])

    df['granger_sig'] = False
    granger_dict = {}

    # Optionally refine with Granger causality
    if returns_df is not None:
        for comp in df.index:
            if comp not in returns_df.columns or target_col not in returns_df.columns:
                continue
            pair = returns_df[[target_col, comp]].dropna()
            if pair.shape[0] < granger_maxlag + 10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = grangercausalitytests(pair.values, maxlag=granger_maxlag, verbose=False)
            pvals = {f'lag_{l}': float(res[l][0]['ssr_ftest'][1]) for l in range(1, granger_maxlag + 1)}
            granger_dict[comp] = pvals
            sig_lags = [int(k.split('_')[1]) for k, v in pvals.items() if v < granger_pval_thresh]
            if sig_lags:
                df.at[comp, 'granger_sig'] = True

    # Selection logic
    selected = []
    feature_plan = {}

    for comp in df.index:
        if not df.at[comp, 'enough_obs']:
            df.at[comp, 'selected'] = False
            continue

        if require_granger:
            keep = df.at[comp, 'granger_sig']
        else:
            keep = df.at[comp, 'preselect'] or df.at[comp, 'granger_sig']

        df.at[comp, 'selected'] = bool(keep)
        if keep:
            lags = []
            if comp in granger_dict:
                lags = sorted([int(k.split('_')[1]) for k,v in granger_dict[comp].items() if v < granger_pval_thresh])
            if not lags:
                lags = default_lags.copy()
            feature_plan[comp] = lags
            selected.append(comp)

    df.attrs['granger_pvals'] = granger_dict
    return df, selected, feature_plan
