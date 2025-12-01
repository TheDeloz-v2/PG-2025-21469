from . import config
from .data_loader import get_stock_data, calculate_daily_returns, save_dataframe
from .analysis import (
    perform_stationarity_tests, 
    calculate_correlation_significance,
    assess_feature_importance,
    print_stationarity_report,
    print_daily_returns_stats
)
from .decomposition import (
    build_classical_decomposition,
    build_wavelet_decomposition,
    combine_decompositions
)
from .feature_engineering import (
    create_price_features,
    add_technical_indicators,
    add_volatility_features,
    prepare_final_dataset
)


class StockPipeline:
    """
    Pipeline for Tesla stock analysis, decomposition, and feature engineering
    """
    
    def __init__(self, custom_config=None):
        self.config = config
        if custom_config:
            self._update_config(custom_config)
        
        self.raw_data = None
        self.daily_returns_stats = None
        self.classical_decomp = None
        self.wavelet_decomp = None
        self.final_dataset = None
    
    def _update_config(self, custom_config):
        """Update configuration with custom values"""
        for key, value in custom_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def load_data(self):
        """Load stock data from Yahoo Finance"""
        print("\n" + "="*60)
        print("PASO 1: CARGA DE DATOS")
        print("="*60)
        
        self.raw_data = get_stock_data()
        
        # Calculate daily returns
        self.raw_data, self.daily_returns_stats = calculate_daily_returns(self.raw_data)
        
        print_daily_returns_stats(self.daily_returns_stats)
        
        return self
    
    def analyze_stationarity(self, verbose=True):
        """Perform stationarity tests"""
        print("\n" + "="*60)
        print("PASO 2: ANÁLISIS DE ESTACIONARIEDAD")
        print("="*60)
        
        # Test close prices
        close_results = perform_stationarity_tests(
            self.raw_data['close'], 
            title='Close Prices'
        )
        
        # Test returns
        returns_results = perform_stationarity_tests(
            self.raw_data['daily_returns'], 
            title='Daily Returns'
        )
        
        if verbose:
            print_stationarity_report(close_results)
            print_stationarity_report(returns_results)
        
        return self
    
    def decompose(self):
        """Perform classical and wavelet decompositions"""
        print("\n" + "="*60)
        print("PASO 3: DESCOMPOSICIONES")
        print("="*60)
        
        # Classical decomposition
        print("\n→ Descomposición clásica...")
        self.classical_decomp = build_classical_decomposition(self.raw_data)
        
        # Wavelet decomposition
        print("\n→ Descomposición wavelet...")
        self.wavelet_decomp = build_wavelet_decomposition(self.raw_data)
        
        return self
    
    def engineer_features(self):
        """Create features from raw data and decompositions"""
        print("\n" + "="*60)
        print("PASO 4: INGENIERÍA DE CARACTERÍSTICAS")
        print("="*60)
        
        # Create base features
        df = create_price_features(self.raw_data)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Add volatility features
        df = add_volatility_features(df)
        
        # Combine with decompositions
        df = combine_decompositions(df, self.classical_decomp, self.wavelet_decomp)
        
        # Prepare final dataset
        self.final_dataset = prepare_final_dataset(df, normalize=True)
        
        return self
    
    def save_dataset(self, output_path=None):
        """Save final dataset to CSV"""
        print("\n" + "="*60)
        print("PASO 5: GUARDAR DATASET")
        print("="*60)
        
        if self.final_dataset is None:
            raise ValueError("No hay dataset para guardar. Ejecuta engineer_features() primero.")
        
        path = save_dataframe(self.final_dataset, filename=output_path)
        
        print(f"\n✓ Dataset guardado en: {path}")
        print(f"✓ Shape: {self.final_dataset.shape}")
        
        return path
    
    def run(self, save=True, output_path=None, verbose=True):
        """
        Run complete pipeline
        
        Args:
            save (bool): Whether to save the final dataset
            output_path (str): Optional custom output path
            verbose (bool): Whether to print detailed analysis
            
        Returns:
            pd.DataFrame or str: Final dataset or path to saved file
        """
        print("\n" + "="*60)
        print("INICIANDO PIPELINE DE DESCOMPOSICIÓN DE TESLA")
        print("="*60)
        
        self.load_data()
        self.analyze_stationarity(verbose=verbose)
        self.decompose()
        self.engineer_features()
        
        if save:
            path = self.save_dataset(output_path)
            print("\n" + "="*60)
            print("PIPELINE COMPLETADO ✓")
            print("="*60)
            return path
        else:
            print("\n" + "="*60)
            print("PIPELINE COMPLETADO ✓")
            print("="*60)
            return self.final_dataset


def run_pipeline(custom_config=None, save=True, output_path=None, verbose=True):
    """
    Convenience function to run the complete pipeline
    
    Args:
        custom_config (dict): Optional custom configuration
        save (bool): Whether to save the final dataset
        output_path (str): Optional custom output path
        verbose (bool): Whether to print detailed analysis
        
    Returns:
        pd.DataFrame or str: Final dataset or path to saved file
    """
    pipeline = StockPipeline(custom_config)
    return pipeline.run(save=save, output_path=output_path, verbose=verbose)
