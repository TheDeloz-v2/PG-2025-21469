import os
import pandas as pd
import numpy as np
from pathlib import Path

from . import config
from .data_loader import load_all_stocks, compute_all_returns, build_returns_df
from .feature_engineering import (
    create_targets, build_final_dataset, compute_pca_factor
)
from .analysis import (
    analyze_and_select_from_summary,
    analyze_and_select_tech_peers,
    analyze_and_select_competitors
)


class StockMarketPipeline:
    """
    Main pipeline for stock market feature engineering
    """
    
    def __init__(self, custom_config=None):
        """
        Initialize pipeline with configuration
        
        Args:
            custom_config (dict): Optional custom configuration to override defaults
        """
        self.config = config
        if custom_config:
            self._update_config(custom_config)
        
        self.stock_data = {}
        self.returns_dict = {}
        self.returns_df = None
        self.suppliers_plan = {}
        self.tech_plan = {}
        self.comp_plan = {}
        self.final_dataset = None
        
    def _update_config(self, custom_config):
        """Update configuration with custom values"""
        for key, value in custom_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def load_data(self):
        """Load all stock data"""
        print("\n" + "="*60)
        print("PASO 1: CARGA DE DATOS")
        print("="*60)
        
        self.stock_data = load_all_stocks(
            self.config.TICKERS,
            self.config.START_DATE,
            self.config.END_DATE
        )
        
        print(f"\n✓ Cargadas {len(self.stock_data)} acciones exitosamente")
        return self
    
    def compute_returns(self):
        """Compute returns for all stocks"""
        print("\n" + "="*60)
        print("PASO 2: CÁLCULO DE RETORNOS")
        print("="*60)
        
        self.returns_dict = compute_all_returns(self.stock_data)
        
        # Build unified returns DataFrame
        all_names = list(self.config.TICKERS.keys())
        self.returns_df = build_returns_df(all_names, self.returns_dict, how='outer')
        
        print(f"\n✓ Retornos calculados: {self.returns_df.shape}")
        return self
    
    def analyze_suppliers(self):
        """Analyze and select supplier features"""
        print("\n" + "="*60)
        print("PASO 3: ANÁLISIS DE PROVEEDORES")
        print("="*60)
        
        # Build simple correlation summary for suppliers
        suppliers_summary = self._build_simple_summary(
            self.config.SUPPLIERS, 
            'Tesla'
        )
        
        # Analyze and select
        summary_out, selected, feature_plan = analyze_and_select_from_summary(
            suppliers_summary,
            returns_df=self.returns_df,
            target_col='Tesla',
            **self.config.SUPPLIERS_PARAMS
        )
        
        self.suppliers_plan = feature_plan
        
        print(f"\n✓ Proveedores seleccionados: {selected}")
        print(f"✓ Features totales: {sum(len(v) for v in feature_plan.values())}")
        
        return self
    
    def analyze_tech_peers(self):
        """Analyze and select tech peer features"""
        print("\n" + "="*60)
        print("PASO 4: ANÁLISIS DE TECH PEERS")
        print("="*60)
        
        # Build simple summaries
        tech_corr_summary = self._build_simple_summary(
            self.config.TECH_PEERS,
            'Tesla'
        )
        
        # For OLS summary, we need partial correlations with NASDAQ
        tech_ols_summary = self._build_ols_summary(
            self.config.TECH_PEERS,
            'Tesla',
            'NASDAQ'
        )
        
        # Analyze and select
        summary_out, selected, feature_plan = analyze_and_select_tech_peers(
            tech_corr_summary,
            tech_ols_summary,
            returns_df=self.returns_df,
            target_col='Tesla',
            **self.config.TECH_PEERS_PARAMS
        )
        
        self.tech_plan = feature_plan
        
        print(f"\n✓ Tech peers seleccionados: {selected}")
        print(f"✓ Features totales: {sum(len(v) for v in feature_plan.values())}")
        
        return self
    
    def analyze_competitors(self):
        """Analyze and select competitor features"""
        print("\n" + "="*60)
        print("PASO 5: ANÁLISIS DE COMPETIDORES")
        print("="*60)
        
        # Build simple summary
        comp_summary = self._build_competitors_summary(
            self.config.COMPETITORS,
            'Tesla'
        )
        
        # Analyze and select
        summary_out, selected, feature_plan = analyze_and_select_competitors(
            comp_summary,
            returns_df=self.returns_df,
            target_col='Tesla',
            **self.config.COMPETITORS_PARAMS
        )
        
        self.comp_plan = feature_plan
        
        print(f"\n✓ Competidores seleccionados: {selected}")
        print(f"✓ Features totales: {sum(len(v) for v in feature_plan.values())}")
        
        return self
    
    def build_features(self):
        """Build final feature dataset"""
        print("\n" + "="*60)
        print("PASO 6: CONSTRUCCIÓN DE FEATURES")
        print("="*60)
        
        # Prepare Tesla data with targets
        df_tesla = self.stock_data['Tesla'].copy()
        df_tesla = create_targets(
            df_tesla,
            target_col='close',
            horizons=self.config.FEATURE_PARAMS['target_horizons'],
            use_returns=True
        )
        
        # Compute PCA factors for each group
        suppliers_pca = compute_pca_factor(
            self.returns_df,
            self.config.SUPPLIERS
        )
        
        tech_pca = compute_pca_factor(
            self.returns_df,
            self.config.TECH_PEERS
        )
        
        comp_pca = compute_pca_factor(
            self.returns_df,
            self.config.COMPETITORS
        )
        
        # Build final dataset
        self.final_dataset = build_final_dataset(
            df_tesla=df_tesla,
            returns_df=self.returns_df,
            suppliers_plan=self.suppliers_plan,
            tech_plan=self.tech_plan,
            comp_plan=self.comp_plan,
            include_pca_factors=self.config.FEATURE_PARAMS['include_pca_factors'],
            suppliers_pca=suppliers_pca,
            tech_pca=tech_pca,
            comp_pca=comp_pca,
            market_indices=self.config.MARKET_INDICES,
            missing_threshold=self.config.FEATURE_PARAMS['missing_threshold']
        )
        
        print(f"\n✓ Dataset final construido: {self.final_dataset.shape}")
        
        return self
    
    def save_dataset(self, output_path=None):
        """
        Save final dataset to CSV
        
        Args:
            output_path (str): Optional custom output path
        """
        print("\n" + "="*60)
        print("PASO 7: GUARDAR DATASET")
        print("="*60)
        
        if self.final_dataset is None:
            raise ValueError("No hay dataset para guardar. Ejecuta build_features() primero.")
        
        if output_path is None:
            output_dir = Path(self.config.OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / self.config.DATASET_FILENAME
        
        self.final_dataset.to_csv(output_path)
        
        print(f"\n✓ Dataset guardado en: {output_path}")
        print(f"✓ Tamaño del archivo: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return str(output_path)
    
    def run(self, save=True, output_path=None):
        """
        Run complete pipeline
        
        Args:
            save (bool): Whether to save the final dataset
            output_path (str): Optional custom output path
            
        Returns:
            pd.DataFrame or str: Final dataset or path to saved file
        """
        print("\n" + "="*60)
        print("INICIANDO PIPELINE DE STOCK MARKET")
        print("="*60)
        
        self.load_data()
        self.compute_returns()
        self.analyze_suppliers()
        self.analyze_tech_peers()
        self.analyze_competitors()
        self.build_features()
        
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
    
    # Helper methods for building summaries
    
    def _build_simple_summary(self, companies, target):
        """Build simple correlation summary"""
        summary_data = []
        
        for comp in companies:
            if comp not in self.returns_df.columns or target not in self.returns_df.columns:
                continue
            
            pair = self.returns_df[[target, comp]].dropna()
            if pair.shape[0] < 50:
                continue
            
            pearson = pair[target].corr(pair[comp])
            
            # Rolling correlation
            rolling_corr = pair[target].rolling(30, min_periods=15).corr(pair[comp])
            
            summary_data.append({
                'n': pair.shape[0],
                'pearson': pearson,
                'rolling_mean': rolling_corr.mean(),
                'rolling_std': rolling_corr.std()
            })
        
        if not summary_data:
            return pd.DataFrame()
        
        return pd.DataFrame(summary_data, index=companies[:len(summary_data)])
    
    def _build_ols_summary(self, companies, target, market_index):
        """Build OLS summary with partial correlations"""
        import statsmodels.api as sm
        
        summary_data = []
        
        for comp in companies:
            if comp not in self.returns_df.columns or target not in self.returns_df.columns:
                continue
            if market_index not in self.returns_df.columns:
                continue
            
            df = self.returns_df[[target, comp, market_index]].dropna()
            if df.shape[0] < 50:
                continue
            
            # Partial correlation via residuals
            # 1. Regress target on market
            X_market = sm.add_constant(df[market_index])
            model_target = sm.OLS(df[target], X_market).fit()
            resid_target = model_target.resid
            
            # 2. Regress company on market
            model_comp = sm.OLS(df[comp], X_market).fit()
            resid_comp = model_comp.resid
            
            # 3. Correlation of residuals
            partial_corr = resid_target.corr(resid_comp)
            
            # Rolling partial correlation
            rolling_partial = resid_target.rolling(30, min_periods=15).corr(resid_comp)
            
            # OLS with both market and company
            X_both = sm.add_constant(df[[market_index, comp]])
            model_both = sm.OLS(df[target], X_both).fit()
            pval_comp = model_both.pvalues[comp] if comp in model_both.pvalues else np.nan
            
            summary_data.append({
                'partial_nasdaq': partial_corr,
                'rolling_partial_mean': rolling_partial.mean(),
                'rolling_partial_std': rolling_partial.std(),
                'pval_company': pval_comp
            })
        
        if not summary_data:
            return pd.DataFrame()
        
        return pd.DataFrame(summary_data, index=companies[:len(summary_data)])
    
    def _build_competitors_summary(self, companies, target):
        """Build competitors summary"""
        summary_data = []
        
        for comp in companies:
            if comp not in self.returns_df.columns or target not in self.returns_df.columns:
                continue
            
            pair = self.returns_df[[target, comp]].dropna()
            if pair.shape[0] < 50:
                continue
            
            pearson = pair[target].corr(pair[comp])
            
            # Rolling correlation
            rolling_corr = pair[target].rolling(30, min_periods=15).corr(pair[comp])
            
            summary_data.append({
                'n_obs': pair.shape[0],
                'pearson': pearson,
                'rolling_corr_mean': rolling_corr.mean(),
                'rolling_corr_std': rolling_corr.std()
            })
        
        if not summary_data:
            return pd.DataFrame()
        
        return pd.DataFrame(summary_data, index=companies[:len(summary_data)])


def run_pipeline(custom_config=None, save=True, output_path=None):
    """
    Convenience function to run the complete pipeline
    
    Args:
        custom_config (dict): Optional custom configuration
        save (bool): Whether to save the final dataset
        output_path (str): Optional custom output path
        
    Returns:
        pd.DataFrame or str: Final dataset or path to saved file
    """
    pipeline = StockMarketPipeline(custom_config)
    return pipeline.run(save=save, output_path=output_path)
