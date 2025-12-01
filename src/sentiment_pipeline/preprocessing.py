import re
import html
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import io
import sys
from . import config

lemmatizer = WordNetLemmatizer()


def ensure_nltk():
    # Redirect stdout to suppress download messages
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    for pkg in config.NLTK_DOWNLOADS:
        try:
            nltk.data.find(pkg)
        except Exception:
            nltk.download(pkg, quiet=True)
    
    # Restore stdout
    sys.stdout = old_stdout


def filter_posts(df):
    """Filter out replies/retweets and posts before START_DATE"""
    df = df.copy()
    df['day'] = pd.to_datetime(df['createdAt']).dt.date
    # Exclude replies and retweets if columns exist
    for col in ['isReply', 'isRetweet', 'isQuote']:
        if col in df.columns:
            df = df[df[col] == False]
    # Keep date range
    df = df[df['day'] >= config.START_DATE]
    df = df.drop(columns=[c for c in ['isRetweet', 'isReply', 'isQuote'] if c in df.columns])
    df = df.reset_index(drop=True)
    return df


def remove_emoji_only(df, text_col='fullText'):
    """Remove rows that are only emojis"""
    pattern = re.compile(r'^[\U0001F600-\U0001F64F' r'\U0001F300-\U0001F5FF' r'\U0001F680-\U0001F6FF'
                         r'\U0001F1E0-\U0001F1FF' r'\U00002700-\U000027BF' r'\U000024C2-\U0001F251'
                         r']+$', flags=re.UNICODE)
    mask = df[text_col].astype(str).apply(lambda x: bool(pattern.fullmatch(str(x).strip())))
    return df[~mask].reset_index(drop=True)


def impute_viewcounts(df):
    """Impute missing viewCount using likeCount and global ratio"""
    df = df.copy()
    if 'viewCount' not in df.columns or 'likeCount' not in df.columns:
        return df
    valid_views = df[df['viewCount'].notna() & df['likeCount'].notna()]
    if valid_views.shape[0] == 0:
        return df
    avg_view = valid_views['viewCount'].mean()
    avg_like = valid_views['likeCount'].mean()
    ratio = avg_view / avg_like if avg_like > 0 else 0

    def impute(row):
        if pd.isna(row['viewCount']) and not pd.isna(row.get('likeCount')):
            return row['likeCount'] * ratio
        return row['viewCount']

    df['viewCount'] = df.apply(impute, axis=1)
    return df


def clean_text(df, text_col='fullText'):
    """Clean and tokenize text, keep hashtags, remove urls, html, and non-alpha chars"""
    df = df.copy()
    df['cleaned_tweet'] = df[text_col].astype(str).map(lambda x: x + ' ')
    df['cleaned_tweet'] = df['cleaned_tweet'].map(lambda x: re.sub(r'http\S+', '', x))
    df['cleaned_tweet'] = df['cleaned_tweet'].map(lambda x: html.unescape(x))
    df['cleaned_tweet'] = df['cleaned_tweet'].map(lambda x: re.sub(r'[^a-zA-Z#]', ' ', x))
    df['cleaned_tweet'] = df['cleaned_tweet'].map(lambda x: x.lower())

    stopword_list = set(stopwords.words('english')) if 'stopwords' in nltk.corpus.__dict__ else set()

    def tokenize_remove_stopwords(text):
        tokens = word_tokenize(text)
        clean_tokens = [w for w in tokens if w.lower() not in stopword_list]
        return clean_tokens

    df['cleaned_tweet'] = df['cleaned_tweet'].apply(tokenize_remove_stopwords)
    return df


def lemmatize_series(token_series):
    """Lemmatize a pandas Series of token lists and return Series of strings"""
    out = []
    for tokens in token_series:
        if not isinstance(tokens, list):
            out.append('')
            continue
        pos_tag_list = nltk.pos_tag(tokens)
        wordnet_tags = []
        for _, tag in pos_tag_list:
            if tag.startswith('J'):
                wordnet_tags.append(wordnet.ADJ)
            elif tag.startswith('N'):
                wordnet_tags.append(wordnet.NOUN)
            elif tag.startswith('R'):
                wordnet_tags.append(wordnet.ADV)
            elif tag.startswith('V'):
                wordnet_tags.append(wordnet.VERB)
            else:
                wordnet_tags.append(wordnet.NOUN)
        lem_words = [lemmatizer.lemmatize(w, pos=t) for w, t in zip(tokens, wordnet_tags)]
        out.append(' '.join(lem_words))
    return pd.Series(out)


def compute_acceptance_score(df):
    """Compute acceptance score based on interaction ratios (like/view etc.)"""
    df = df.copy()
    interaction_cols = ["likeCount", "retweetCount", "quoteCount", "bookmarkCount"]
    for col in interaction_cols + ['viewCount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    view_safe = df['viewCount'].clip(lower=1)
    for col in interaction_cols:
        if col in df.columns:
            df[f'ratio_{col}'] = np.log1p(df[col]) - np.log1p(view_safe)
    median_ratios = {col: df[f'ratio_{col}'].median() for col in interaction_cols if f'ratio_{col}' in df.columns}
    for col in interaction_cols:
        if f'ratio_{col}' in df.columns:
            df[f'norm_{col}'] = df[f'ratio_{col}'] - median_ratios[col]
    df['acceptance_raw'] = df[[f'norm_{c}' for c in interaction_cols if f'norm_{c}' in df.columns]].mean(axis=1)
    min_raw = df['acceptance_raw'].min() if 'acceptance_raw' in df.columns else 0
    shift = -min_raw if pd.notna(min_raw) and min_raw < 0 else 0.0
    df['acceptance_raw_shifted'] = df.get('acceptance_raw', 0) + shift
    df['acceptance'] = df['acceptance_raw_shifted'] / (df['acceptance_raw_shifted'] + 1.0)
    df['acceptance'] = df['acceptance'].fillna(0.0).clip(0.0, 1.0)
    # drop intermediate columns
    drop_cols = [c for c in df.columns if c.startswith('ratio_') or c.startswith('norm_') or c=='acceptance_raw_shifted']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def compute_vader_scores(df):
    """Compute VADER sentiment scores and attach to DataFrame"""
    sia = SentimentIntensityAnalyzer()
    df['sentiment_vader_raw'] = df['fullText'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_vader_light'] = df['fullText'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_score'] = df['sentiment_vader_raw']
    return df


def aggregate_daily(df):
    """Aggregate sentiment and acceptance by day"""
    df_daily = df[['day', 'sentiment_score', 'acceptance']].copy()
    df_daily['day'] = pd.to_datetime(df_daily['day'])
    daily_avg = df_daily.groupby('day')[['sentiment_score', 'acceptance']].mean().reset_index()
    daily_avg = daily_avg.set_index('day').sort_index()
    # Reindex daily and fill gaps conservatively
    full_idx = pd.date_range(daily_avg.index.min(), daily_avg.index.max(), freq='D')
    df_full = daily_avg.reindex(full_idx)
    prev = df_full.ffill()
    nxt  = df_full.bfill()
    for col in ['sentiment_score','acceptance']:
        avg = (prev[col] + nxt[col]) / 2.0
        filled = avg.fillna(prev[col]).fillna(nxt[col])
        df_full[col] = df_full[col].fillna(filled)
    df_filled = df_full.reset_index().rename(columns={'index':'day'})
    return df_filled
