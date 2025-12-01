from pathlib import Path
from . import config
from .data_loader import load_raw_posts, save_df
from .preprocessing import (
    ensure_nltk, filter_posts, remove_emoji_only, impute_viewcounts,
    clean_text, lemmatize_series, compute_acceptance_score, compute_vader_scores, aggregate_daily
)
from .analysis import basic_stats, save_daily_csv

class SentimentPipeline:
    """
    Pipeline for sentiment analysis of social media posts
    """
    def __init__(self, custom_config=None):
        """
        Initialize the SentimentPipeline
        
        Args:
            custom_config (dict): Optional dictionary to override default config values
        """
        self.config = config
        if custom_config:
            self._update_config(custom_config)
        self.raw = None
        self.filtered = None
        self.cleaned = None
        self.daily = None

    def _update_config(self, custom):
        """Update configuration with custom values"""
        for k,v in custom.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

    def load_raw(self, path=None):
        """Load raw posts from CSV file"""
        print("\n" + "="*60)
        print("PASO 1: CARGA DE DATOS EN BRUTO")
        print("="*60)
        
        self.raw = load_raw_posts(path)
        print(f"\n✓ Cargados {self.raw.shape[0]} posts en bruto desde {path or self.config.RAW_PATH}")
        return self

    def preprocess(self):
        """Preprocess the raw posts"""
        print("\n" + "="*60)
        print("PASO 2: PREPROCESAMIENTO DE DATOS")
        print("="*60)
        
        ensure_nltk()
        df = self.raw.copy()
        df = filter_posts(df)
        if self.config.EMOJI_ONLY_REMOVE:
            df = remove_emoji_only(df)
        df = impute_viewcounts(df)
        df = clean_text(df)
        # Lemmatize
        df['cleaned_tweet'] = lemmatize_series(df['cleaned_tweet'])
        df = df[df['cleaned_tweet'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        self.cleaned = df
        if self.config.SAVE_CLEANED:
            save_df(df, self.config.CLEANED_POSTS, index=False)
            
        print(f"\n✓ Preprocesamiento completado: {self.cleaned.shape[0]} posts limpios")
        return self

    def analyze(self):
        """Analyze the cleaned posts to compute sentiment scores"""
        print("\n" + "="*60)
        print("PASO 3: ANÁLISIS DE DATOS")
        print("="*60)
        df = self.cleaned.copy()
        df = compute_acceptance_score(df)
        df = compute_vader_scores(df)
        self.cleaned = df
        daily = aggregate_daily(df)
        self.daily = daily
        if self.config.SAVE_DAILY_SENTIMENT:
            out = f"{self.config.PROCESSED_DIR}/{self.config.SENTIMENT_SCORES}"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(out, index=False)
            print(f"\n✓ Guardados los puntajes diarios de sentimiento en {out}")
        return self

    def run(self, save=True, output_path=None):
        """
        Run the sentiment analysis pipeline
    
        Args:
            save (bool, optional): Whether to save the output. Defaults to True.
            output_path (str, optional): Path to save the output. Defaults to None.

        Returns:
            str: Path to the saved output file or the processed DataFrame.
        """
        print("\n" + "="*60)
        print("INICIANDO EL PIPELINE DE ANÁLISIS DE SENTIMIENTO")
        print("="*60)
        
        self.load_raw()
        self.preprocess()
        self.analyze()
        
        if save:
            print("\n" + "="*60)
            print("PIPELINE COMPLETADO ✓")
            print("="*60)
            if output_path is None:
                output_path = f"{self.config.PROCESSED_DIR}/{self.config.SENTIMENT_SCORES}"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.daily.to_csv(output_path, index=False)
            return output_path
        else:
            return self.daily


def run_pipeline(custom_config=None, save=True, output_path=None):
    """
    Convenience function to run the sentiment analysis pipeline
    
    Args:
        custom_config (dict): Custom configuration overrides
        save (bool): Whether to save the output
        output_path (str): Path to save the output
        
    Returns:
        str: Path to the saved output file or the processed DataFrame
    """
    p = SentimentPipeline(custom_config)
    return p.run(save=save, output_path=output_path)
