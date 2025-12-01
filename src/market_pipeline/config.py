import datetime

# ==================== DATE RANGES ====================
START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime(2025, 7, 1)


# ==================== TICKERS ====================
TICKERS = {
    # Target
    "Tesla": "TSLA",
    
    # Suppliers
    "CATL": "300750.SZ",
    "Panasonic": "6752.T",
    #"LG Energy Solutions": "373220.KS",
    "Ganfeng Lithium": "1772.HK",
    "Glencore": "GLEN.L",
    "AGC Inc.": "5201.T",
    "Fuyao Glass": "3606.HK",
    
    # Competitors
    "Ford": "F",
    "General Motors": "GM",
    "BYD": "1211.HK",
    "NIO": "NIO",
    #"Rivian": "RIVN",
    
    # Tech Companies
    "NVIDIA": "NVDA",
    "Apple": "AAPL",
    "Meta": "META",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    
    # Market Indices
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "VIX": "^VIX"
}


# ==================== COMPANY CATEGORIES ====================
SUPPLIERS = ["CATL", "Panasonic", "Ganfeng Lithium", 
             "Glencore", "AGC Inc.", "Fuyao Glass"]

COMPETITORS = ["Ford", "General Motors", "BYD", "NIO"]

TECH_PEERS = ["NVIDIA", "Apple", "Meta", "Microsoft", "Google"]

MARKET_INDICES = ["S&P 500", "NASDAQ", "Dow Jones", "VIX"]


# ==================== ANALYSIS PARAMETERS ====================
SUPPLIERS_PARAMS = {
    'pearson_thresh': 0.08,
    'rolling_mean_thresh': 0.06,
    'rolling_std_max': 0.17,
    'min_obs_required': 100,
    'granger_maxlag': 5,
    'granger_pval_thresh': 0.05,
    'require_granger': False,
    'default_lags': [1, 2]
}

TECH_PEERS_PARAMS = {
    'pearson_thresh': 0.3,
    'partial_thresh': 0.2,
    'rolling_partial_mean_thresh': 0.2,
    'rolling_partial_std_max': 0.15,
    'pval_company_thresh': 0.05,
    'granger_maxlag': 5,
    'granger_pval_thresh': 0.05,
    'require_granger': False,
    'default_lags': [1, 2]
}

COMPETITORS_PARAMS = {
    'pearson_thresh': 0.10,
    'rolling_mean_thresh': 0.07,
    'rolling_std_max': 0.15,
    'min_obs_required': 100,
    'granger_maxlag': 5,
    'granger_pval_thresh': 0.05,
    'require_granger': False,
    'default_lags': [1, 2]
}


# ==================== FEATURE ENGINEERING PARAMS ====================
FEATURE_PARAMS = {
    'target_horizons': [1, 5, 21],  # Predict 1 day, 1 week (5 trading days), 1 month (21 trading days) ahead
    'include_pca_factors': True,
    'rolling_window': 10,
    'rolling_min_periods': 1,
    'max_fill_gap': 3,
    'missing_threshold': 0.6  # Keep rows with <40% features missing
}


# ==================== OUTPUT PATHS ====================
OUTPUT_DIR = "data/processed"
DATASET_FILENAME = "tesla_market_features.csv"