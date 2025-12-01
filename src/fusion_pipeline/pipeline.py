from pathlib import Path
from . import config
from .data_loader import (
    load_market_data, 
    load_sentiment_data, 
    load_stock_data,
    save_dataset,
    validate_dataframe,
    print_validation_report
)
from .alignment import create_aligned_dataset, align_datasets_to_common_dates
from .target_engineering import (
    generate_all_datasets,
    validate_targets_consistency
)


def clean_dataset_for_leakage(df, dataset_name='dataset', verbose=True):
    """
    Remove features with data leakage from fusion datasets and drop rows with NaNs.
    
    Args:
        df: DataFrame with all features
        dataset_name: Name of dataset for reporting
        verbose: Print cleaning summary
        
    Returns:
        Cleaned DataFrame without NaNs
    """
    if verbose:
        print(f"\n→ Limpiando data leakage en {dataset_name}...")
    
    original_shape = df.shape
    
    # Features to remove (same logic as stock pipeline)
    price_leakage = ['close', 'high', 'low', 'open', 
                     'close_stock', 'high_stock', 'low_stock', 'open_stock']
    derived_leakage = ['wavelet_approx', 'BB_middle', 'BB_upper', 'BB_lower']
    temporal_leakage = ['trend', 'seasonal', 'residual']
    duplicates = ['daily_returns', 'volume_stock']
    
    all_to_remove = price_leakage + derived_leakage + temporal_leakage + duplicates
    
    # Remove features that exist
    features_removed = [f for f in all_to_remove if f in df.columns]
    df_clean = df.drop(columns=features_removed, errors='ignore')
    
    # **NEW: Drop rows with NaN values**
    nans_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.dropna()
    nans_removed = nans_before - df_clean.isnull().sum().sum()
    
    if verbose:
        if features_removed:
            print(f"  Eliminados {len(features_removed)} features con leakage")
        if nans_removed > 0:
            print(f"  Eliminadas {original_shape[0] - df_clean.shape[0]} filas con NaN ({nans_removed} valores NaN)")
        print(f"  Shape: {original_shape} → {df_clean.shape}")
    
    return df_clean


class FusionPipeline:
    """
    Fusion pipeline to generate 4 aligned datasets:
    - A (market): Market features + target
    - B (sentiment): Sentiment features + target  
    - C (stock): Stock technical features + target
    - D (fusion): All features + target
    """
    
    def __init__(self, market_path=None, sentiment_path=None, stock_path=None):
        """Initialize with input paths"""
        self.market_path = market_path or config.INPUT_MARKET
        self.sentiment_path = sentiment_path or config.INPUT_SENTIMENT
        self.stock_path = stock_path or config.INPUT_STOCK
        
        self.market_df = None
        self.sentiment_df = None
        self.stock_df = None
        self.aligned_df = None
        self.datasets = {}
    
    def load_data(self):
        """Load all 3 input datasets"""
        print("\n" + "="*70)
        print("STEP 1: LOADING INPUT DATASETS")
        print("="*70)
        
        self.market_df = load_market_data(self.market_path)
        self.sentiment_df = load_sentiment_data(self.sentiment_path)
        self.stock_df = load_stock_data(self.stock_path)
        
        print(f"\n✓ All datasets loaded successfully")
        return self
    
    def align_datasets(self, fill_sentiment=True, handle_stock='drop'):
        """Temporally align datasets using market as base"""
        print("\n" + "="*70)
        print("STEP 2: TEMPORAL ALIGNMENT")
        print("="*70)
        
        self.aligned_df = create_aligned_dataset(
            self.market_df,
            self.sentiment_df, 
            self.stock_df,
            fill_sentiment=fill_sentiment,
            handle_stock=handle_stock,
            drop_low_coverage=True,
            clean_samples=True
        )
        
        print(f"\n✓ Alignment complete")
        return self
    
    def generate_datasets(self):
        """Generate 4 datasets (A, B, C, D)"""
        print("\n" + "="*70)
        print("STEP 3: DATASET GENERATION")
        print("="*70)
        
        if self.aligned_df is None:
            raise ValueError("Must run align_datasets() before generate_datasets()")
        
        self.datasets = generate_all_datasets(self.aligned_df)
        
        print(f"\n✓ Generated {len(self.datasets)} datasets")
        return self
    
    def equalize_date_ranges(self):
        """
        Equalize all datasets to have the same date range (STEP 3.5)
        
        This ensures fair comparison by using the same time period across all datasets.
        """
        print("\n" + "="*70)
        print("STEP 3.5: EQUALIZING DATE RANGES")
        print("="*70)
        
        if not self.datasets:
            raise ValueError("No datasets generated. Run generate_datasets() first.")
        
        self.datasets = align_datasets_to_common_dates(self.datasets)
        
        print(f"\n✓ Date ranges equalized across {len(self.datasets)} datasets")
        return self
    
    def validate_datasets(self):
        """Validate all generated datasets"""
        print("\n" + "="*70)
        print("STEP 4: VALIDATION")
        print("="*70)
        
        if not self.datasets:
            raise ValueError("No datasets generated. Run generate_datasets() first.")
        
        # Check target consistency
        target_report = validate_targets_consistency(self.datasets)
        
        # Detailed validation for each dataset
        for name, df in self.datasets.items():
            validation = validate_dataframe(df, name=name.upper())
            print_validation_report(validation)
        
        return target_report
    
    def clean_leakage(self, verbose=True):
        """Clean data leakage from all generated datasets"""
        print("\n" + "="*70)
        print("STEP 5: CLEANING DATA LEAKAGE")
        print("="*70)
        
        if not self.datasets:
            raise ValueError("No datasets to clean. Run generate_datasets() first.")
        
        print("Eliminando features con data leakage de todos los datasets...")
        
        for name in ['market', 'sentiment', 'stock', 'fusion']:
            if name in self.datasets:
                self.datasets[name] = clean_dataset_for_leakage(
                    self.datasets[name], 
                    dataset_name=name.upper(),
                    verbose=verbose
                )
        
        print("\n✓ Limpieza completada en todos los datasets")
        return self
    
    def save_datasets(self, output_dir=None):
        """Save all datasets to CSV"""
        print("\n" + "="*70)
        print("STEP 6: SAVING DATASETS")
        print("="*70)
        
        if not self.datasets:
            raise ValueError("No datasets to save. Run generate_datasets() first.")
        
        output_paths = {}
        
        # Save each dataset
        for name in ['market', 'sentiment', 'stock', 'fusion']:
            if name in self.datasets:
                filename = f"dataset_{name}.csv"
                path = save_dataset(self.datasets[name], filename, output_dir)
                output_paths[name] = path
                print(f"✓ {name.upper():12}: {path}")
        
        return output_paths
    
    def run(self, save=True, validate=True, clean_leakage=True, equalize_dates=True):
        """
        Execute full fusion pipeline
        
        Args:
            save: Whether to save datasets to CSV
            validate: Whether to validate datasets
            clean_leakage: Whether to clean data leakage (recommended: True)
            equalize_dates: Whether to equalize date ranges across datasets (recommended: True)
        
        Returns:
            dict: Generated datasets
        """
        print("\n" + "="*70)
        print("FUSION PIPELINE: START")
        print("="*70)
        
        # Execute pipeline steps
        self.load_data()
        self.align_datasets()
        self.generate_datasets()
        
        if equalize_dates:
            self.equalize_date_ranges()
        else:
            print("\nWARNING: Date range equalization DISABLED - comparisons may be unfair")
        
        if validate:
            self.validate_datasets()
        
        # Clean data leakage (new step)
        if clean_leakage:
            self.clean_leakage(verbose=True)
        else:
            print("\nADVERTENCIA: Limpieza de data leakage DESACTIVADA")
        
        if save:
            output_paths = self.save_datasets()
            
            print("\n" + "="*70)
            print("FUSION PIPELINE: COMPLETE")
            print("="*70)
            print(f"Generated {len(self.datasets)} datasets:")
            for name, path in output_paths.items():
                shape = self.datasets[name].shape
                date_range = f"{self.datasets[name].index.min().date()} to {self.datasets[name].index.max().date()}"
                print(f"  {name.upper():12}: {shape[0]:4} × {shape[1]:3} | {date_range}")
            
            return self.datasets, output_paths
        
        return self.datasets


def run_pipeline(market_path=None, sentiment_path=None, stock_path=None, 
                save=True, validate=True, clean_leakage=True, equalize_dates=True):
    """
    Convenience function to run fusion pipeline
    
    Args:
        market_path: Path to market dataset
        sentiment_path: Path to sentiment dataset
        stock_path: Path to stock dataset
        save: Whether to save datasets
        validate: Whether to validate datasets
        clean_leakage: Whether to clean data leakage (recommended: True)
        equalize_dates: Whether to equalize date ranges across datasets (recommended: True)
    
    Returns:
        dict: Generated datasets (and output paths if save=True)
    """
    pipeline = FusionPipeline(
        market_path=market_path,
        sentiment_path=sentiment_path,
        stock_path=stock_path
    )
    
    return pipeline.run(
        save=save, 
        validate=validate, 
        clean_leakage=clean_leakage,
        equalize_dates=equalize_dates
    )


# Example usage
if __name__ == "__main__":
    # Run with defaults from config
    datasets, paths = run_pipeline()
    
    print("\n" + "="*70)
    print("Pipeline executed successfully!")
    print("="*70)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()} Dataset:")
        print(df.head())
        print(f"Shape: {df.shape}")
