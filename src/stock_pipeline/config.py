import datetime

# ==================== DATE RANGES ====================
START_DATE = datetime.datetime(2019, 12, 31)
END_DATE = datetime.datetime(2025, 7, 1)

# ==================== TICKER ====================
TICKER = "TSLA"
COMPANY_NAME = "Tesla"

# ==================== DECOMPOSITION PARAMETERS ====================
CLASSICAL_DECOMPOSITION = {
    'model': 'additive',
    'period': 21
}

# ==================== WAVELET PARAMETERS ====================
WAVELET_DECOMPOSITION = {
    'wavelet': 'db4',
    'level': 3
}

# ==================== INDICATORS WINDOWS ====================
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
VOLATILITY_WINDOW = 21
ATR_WINDOW = 14

# ==================== OUTPUT PATHS ====================
PROCESSED_DIR = "data/processed"
OUTPUT_FILENAME = "tesla_features.csv"
