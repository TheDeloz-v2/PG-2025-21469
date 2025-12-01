# ==================== INPUT PATHS ====================
INPUT_MARKET = "data/processed/tesla_market_features.csv"
INPUT_SENTIMENT = "data/processed/sentiment-scores.csv"
INPUT_STOCK = "data/processed/tesla_features.csv"

# ==================== OUTPUT PATHS ====================
OUTPUT_DIR = "data/processed/fusion"
OUTPUT_MARKET_ONLY = "dataset_A_market.csv"
OUTPUT_SENTIMENT_ONLY = "dataset_B_sentiment.csv"
OUTPUT_STOCK_ONLY = "dataset_C_stock.csv"
OUTPUT_FULL_FUSION = "dataset_D_full.csv"

# ==================== TARGET CONFIGURATION ====================
TARGET_HORIZONS = {
    'short': 1,    # 1 day ahead
    'medium': 5,   # 1 week (5 trading days) ahead
    'long': 21     # 1 month (21 trading days) ahead
}

# Primary target for initial training
TARGET_COLUMN = "target_return_t+1"  # Return-based target for trend prediction
TARGET_HORIZON = 1  # Predict 1 day ahead (backward compatibility)

# ==================== ALIGNMENT PARAMETERS ====================
MERGE_METHOD = "left"  # Use market dates as base (most complete)
FILL_METHOD = "ffill"  # Forward-fill for sentiment gaps
MAX_FILL_DAYS = 3  # Max days to forward-fill sentiment
MIN_DATE = None  # Auto-detect from market data
MAX_DATE = None  # Auto-detect from market data

# ==================== DATASET PARAMETERS ====================
EXCLUDE_COLUMNS = {
    # Sentiment features (for creating datasets A, C, D)
    'sentiment': ['sentiment_score', 'acceptance'],
    # Stock features (for creating datasets A, B, D)
    # auto-detected as columns from stock pipeline that aren't in market
    'stock': []
}

# ==================== VALIDATION PARAMETERS ====================
MIN_SAMPLES = 100 
MIN_FEATURE_COVERAGE = 0.7
