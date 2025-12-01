import datetime

# ==================== DATE RANGES ====================
START_DATE = datetime.date(2020, 1, 1)
END_DATE = datetime.date(2025, 7, 1)


# ==================== INPUT/OUTPUT PATHS ====================
RAW_PATH = "data/raw/all-musk-posts.csv"
PROCESSED_DIR = "data/processed"
FILTERED_POSTS = "filtered-posts.csv"
CLEANED_POSTS = "cleaned-posts.csv"
SENTIMENT_SCORES = "sentiment-scores.csv"


# ==================== ANALYSIS PARAMETERS ====================
MIN_VIEWS_FOR_RATIO = 1
EMOJI_ONLY_REMOVE = True
NLTK_DOWNLOADS = ['stopwords','punkt','averaged_perceptron_tagger','wordnet']


# ==================== SAVING PARAMETERS ====================
SAVE_CLEANED = True
SAVE_DAILY_SENTIMENT = True