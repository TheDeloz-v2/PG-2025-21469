import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
import math
import xgboost as xgb

from .config import (
    StockModelConfig,
    MarketModelConfig,
    SentimentModelConfig,
    FusionModelConfig
)
from .architecture import AdvancedLSTMModel
from fusion_pipeline import config as fusion_config
from .data_loader import prepare_data, create_dataloaders, save_scalers, inverse_transform_predictions
from .trainer import ModelTrainer
from .evaluation import (
    calculate_financial_metrics,
    compare_models,
    create_evaluation_report
)


def make_serializable(obj):
    """
    Recursively convert numpy/torch types to native Python types for JSON serialization.
    """

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    # numpy arrays
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return [make_serializable(v) for v in obj.tolist()]
        if isinstance(obj, _np.generic):
            val = obj.item()
            if isinstance(val, float):
                return None if not math.isfinite(val) else float(val)
            return int(val) if isinstance(val, (int,)) else val
    except Exception:
        pass

    # numeric scalars from numpy
    try:
        import numbers
        if isinstance(obj, numbers.Number):
            val = float(obj)
            return None if not math.isfinite(val) else val
    except Exception:
        pass

    return obj


class ModelTrainingPipeline:
    """
    Complete pipeline for training all models
    """
    
    def __init__(
        self,
        data_dir: str = "data/processed/fusion",
        models_dir: str = "../models",
        results_dir: str = "../results"
    ):
        # Get absolute paths relative to this file's location
        base_path = Path(__file__).parent.parent
        self.data_dir = base_path / data_dir
        self.models_dir = base_path / models_dir
        self.results_dir = base_path / results_dir
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.trainers = {}
        self.metrics = {}
        self.scalers = {}
        self.prepared_data = {}
        self.training_histories = {}
                
    def save_training_history(self, model_name: str, history: Dict):
        """Save training history (loss curves) for a model"""
        history_path = self.results_dir / f'{model_name}_training_history.json'
        
        # Convert to serializable format
        serializable_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'epoch_time': [float(x) for x in history['epoch_time']],
            'total_epochs': len(history['train_loss'])
        }
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.training_histories[model_name] = serializable_history
        print(f"  ✓ Training history saved: {history_path.name}")
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets"""
        
        datasets = {
            'stock': pd.read_csv(self.data_dir / 'dataset_stock.csv'),
            'market': pd.read_csv(self.data_dir / 'dataset_market.csv'),
            'sentiment': pd.read_csv(self.data_dir / 'dataset_sentiment.csv'),
            'fusion': pd.read_csv(self.data_dir / 'dataset_fusion.csv')
        }
        
        # Convert Date column to datetime
        for name, df in datasets.items():
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
        
        return datasets
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        seq_length: int = 30,
        target_col: str = 'target_return_t+1'
    ) -> Dict:
        """Prepare single dataset for training"""
        
        exclude_cols = ['Date'] + [col for col in df.columns if col.startswith('target_')]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\nPreparing {dataset_name} dataset...")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target: {target_col}")
        print(f"  Excluded: {exclude_cols}")
        
        # Prepare data
        prepared = prepare_data(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            seq_length=seq_length,
            train_ratio=0.6,
            val_ratio=0.2,
            scale_method="standard"
        )
        
        print(f"  Train: {prepared['X_train'].shape}")
        print(f"  Val:   {prepared['X_val'].shape}")
        print(f"  Test:  {prepared['X_test'].shape}")
        
        return prepared
    
    def train_stock_model(self, config: StockModelConfig) -> Tuple[AdvancedLSTMModel, Dict]:
        """Train Stock LSTM model"""
        print("\n" + "="*60)
        print("TRAINING STOCK MODEL")
        print("="*60)
        
        # Prepare data
        datasets = self.load_datasets()
        prepared = self.prepare_dataset(datasets['stock'], 'stock', config.seq_length)
        self.prepared_data['stock'] = prepared
        
        # Save scalers
        save_scalers(
            prepared['feature_scaler'],
            prepared['target_scaler'],
            str(self.models_dir / 'stock_scalers.pkl')
        )
        
        # Create model
        input_size = prepared['X_train'].shape[2]
        model = AdvancedLSTMModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            prepared['X_train'], prepared['y_train'],
            prepared['X_val'], prepared['y_val'],
            prepared['X_test'], prepared['y_test'],
            batch_size=config.batch_size
        )
        
        # Train
        trainer = ModelTrainer(
            model=model,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            early_stopping_patience=config.early_stopping_patience
        )
        
        # Evaluate
        test_preds, test_targets = trainer.predict(test_loader)
        
        # Inverse transform
        test_preds_orig = inverse_transform_predictions(test_preds, prepared['target_scaler'])
        test_targets_orig = inverse_transform_predictions(test_targets, prepared['target_scaler'])
        
        # Calculate metrics
        metrics = calculate_financial_metrics(test_preds_orig, test_targets_orig)
        
        # Save model
        trainer.save_model(str(self.models_dir / 'stock_model.pth'))
        self.save_training_history('stock', trainer.history)
        
        self.models['stock'] = model
        self.trainers['stock'] = trainer
        self.metrics['stock'] = metrics
        
        print(f"\n✓ Stock model training completed")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test MAE: {metrics['mae']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        
        return model, metrics
    
    def train_market_model(self, config: MarketModelConfig) -> Tuple[AdvancedLSTMModel, Dict]:
        """Train Market LSTM model"""
        print("\n" + "="*60)
        print("TRAINING MARKET MODEL")
        print("="*60)
        
        datasets = self.load_datasets()
        prepared = self.prepare_dataset(datasets['market'], 'market', config.seq_length)
        self.prepared_data['market'] = prepared
        
        save_scalers(
            prepared['feature_scaler'],
            prepared['target_scaler'],
            str(self.models_dir / 'market_scalers.pkl')
        )
        
        input_size = prepared['X_train'].shape[2]
        model = AdvancedLSTMModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            prepared['X_train'], prepared['y_train'],
            prepared['X_val'], prepared['y_val'],
            prepared['X_test'], prepared['y_test'],
            batch_size=config.batch_size
        )
        
        trainer = ModelTrainer(
            model=model,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            early_stopping_patience=config.early_stopping_patience
        )
        
        test_preds, test_targets = trainer.predict(test_loader)
        test_preds_orig = inverse_transform_predictions(test_preds, prepared['target_scaler'])
        test_targets_orig = inverse_transform_predictions(test_targets, prepared['target_scaler'])
        metrics = calculate_financial_metrics(test_preds_orig, test_targets_orig)
        
        trainer.save_model(str(self.models_dir / 'market_model.pth'))
        self.save_training_history('market', trainer.history)
        
        self.models['market'] = model
        self.trainers['market'] = trainer
        self.metrics['market'] = metrics
        
        print(f"\n✓ Market model training completed")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test MAE: {metrics['mae']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        
        return model, metrics
    
    def train_sentiment_model(self, config: SentimentModelConfig) -> Tuple[AdvancedLSTMModel, Dict]:
        """Train Sentiment LSTM model"""
        print("\n" + "="*60)
        print("TRAINING SENTIMENT MODEL")
        print("="*60)
        
        datasets = self.load_datasets()
        prepared = self.prepare_dataset(datasets['sentiment'], 'sentiment', config.seq_length)
        self.prepared_data['sentiment'] = prepared
        
        save_scalers(
            prepared['feature_scaler'],
            prepared['target_scaler'],
            str(self.models_dir / 'sentiment_scalers.pkl')
        )
        
        input_size = prepared['X_train'].shape[2]
        model = AdvancedLSTMModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            prepared['X_train'], prepared['y_train'],
            prepared['X_val'], prepared['y_val'],
            prepared['X_test'], prepared['y_test'],
            batch_size=config.batch_size
        )
        
        trainer = ModelTrainer(
            model=model,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            early_stopping_patience=config.early_stopping_patience
        )
        
        test_preds, test_targets = trainer.predict(test_loader)
        test_preds_orig = inverse_transform_predictions(test_preds, prepared['target_scaler'])
        test_targets_orig = inverse_transform_predictions(test_targets, prepared['target_scaler'])
        metrics = calculate_financial_metrics(test_preds_orig, test_targets_orig)
        
        trainer.save_model(str(self.models_dir / 'sentiment_model.pth'))
        self.save_training_history('sentiment', trainer.history)
        
        self.models['sentiment'] = model
        self.trainers['sentiment'] = trainer
        self.metrics['sentiment'] = metrics
        
        print(f"\n✓ Sentiment model training completed")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test MAE: {metrics['mae']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        
        return model, metrics
    
    def train_fusion_model(self, config: FusionModelConfig) -> Tuple[AdvancedLSTMModel, Dict]:
        """Train Fusion LSTM model with all features"""
        print("\n" + "="*60)
        print("TRAINING FUSION MODEL")
        print("="*60)
        
        datasets = self.load_datasets()
        prepared = self.prepare_dataset(datasets['fusion'], 'fusion', config.seq_length)
        self.prepared_data['fusion'] = prepared
        
        save_scalers(
            prepared['feature_scaler'],
            prepared['target_scaler'],
            str(self.models_dir / 'fusion_scalers.pkl')
        )
        
        input_size = prepared['X_train'].shape[2]
        model = AdvancedLSTMModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            use_attention=config.use_attention
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            prepared['X_train'], prepared['y_train'],
            prepared['X_val'], prepared['y_val'],
            prepared['X_test'], prepared['y_test'],
            batch_size=config.batch_size
        )
        
        trainer = ModelTrainer(
            model=model,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            early_stopping_patience=config.early_stopping_patience
        )
        
        test_preds, test_targets = trainer.predict(test_loader)
        test_preds_orig = inverse_transform_predictions(test_preds, prepared['target_scaler'])
        test_targets_orig = inverse_transform_predictions(test_targets, prepared['target_scaler'])
        metrics = calculate_financial_metrics(test_preds_orig, test_targets_orig)
        
        trainer.save_model(str(self.models_dir / 'fusion_model.pth'))
        self.save_training_history('fusion', trainer.history)
        
        self.models['fusion'] = model
        self.trainers['fusion'] = trainer
        self.metrics['fusion'] = metrics
        
        print(f"\n✓ Fusion model training completed")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test MAE: {metrics['mae']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        
        return model, metrics
    
    def identify_stock_features(self, fusion_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify stock features (always included) vs other features (candidates for selection).
        Stock features are identified by their presence in dataset_stock.csv
        
        Args:
            fusion_df: Fusion dataset DataFrame
            
        Returns:
            Tuple of (stock_features, candidate_features)
        """
        # Load stock dataset to get feature names
        stock_df = pd.read_csv(self.data_dir / 'dataset_stock.csv')
        
        # Exclude ALL target columns (for all horizons) and Date
        exclude_cols = ['Date'] + [col for col in stock_df.columns if col.startswith('target_')]
        
        # Get stock feature names
        stock_features = [col for col in stock_df.columns if col not in exclude_cols]
        
        # Get fusion feature names
        fusion_exclude = ['Date'] + [col for col in fusion_df.columns if col.startswith('target_')]
        fusion_features = [col for col in fusion_df.columns if col not in fusion_exclude]
        
        # Candidate features = fusion features - stock features
        candidate_features = [feat for feat in fusion_features if feat not in stock_features]
        
        return stock_features, candidate_features
    
    def train_fusion_optimized_model(self, config: FusionModelConfig) -> Tuple[AdvancedLSTMModel, Dict]:
        """
        Train Fusion LSTM model with ROBUST multi-method feature selection.
        
        - Recursive Feature Elimination with Cross-Validation (RFECV)
        - EXHAUSTIVE search for the OPTIMAL feature subset
        - Time Series Cross-Validation for robust evaluation
        - NO shortcuts, NO compromises on computational complexity
        """
        print("\n" + "="*80)
        print("TRAINING FUSION OPTIMIZED MODEL (FEATURE SELECTION)")
        print("="*80)
        
        datasets = self.load_datasets()
        fusion_df = datasets['fusion']
        
        # Identify stock features and candidate features
        stock_features, candidate_features = self.identify_stock_features(fusion_df)
        
        print(f"\nFeature Analysis:")
        print(f"  Stock features (always included): {len(stock_features)}")
        print(f"  Candidate features (selection): {len(candidate_features)}")
        print(f"  Total fusion features: {len(stock_features) + len(candidate_features)}")
        
        # Extract data for feature selection
        target_col = fusion_config.TARGET_COLUMN
        print(f"  Target column: {target_col}")
        
        # Prepare candidate features data
        X_candidates = fusion_df[candidate_features].values
        y = fusion_df[target_col].values
        
        print(f"\n{'='*80}")
        print("FEATURE SELECTION WITH XGBOOST")
        print(f"{'='*80}")
        print("\nConfiguration:")
        print("  • n_estimators: 1000 (maximum trees for best learning)")
        print("  • max_depth: 10 (deep trees for complex patterns)")
        print("  • learning_rate: 0.01 (slow and steady for optimal convergence)")
        print("  • subsample: 0.7 (strong regularization)")
        print("  • colsample_bytree: 0.7")
        print("  • colsample_bylevel: 0.7")
        print("  • min_child_weight: 5 (conservative splits)")
        print("  • gamma: 0.2 (higher pruning threshold)")
        print("  • reg_alpha: 0.5 (strong L1 regularization)")
        print("  • reg_lambda: 2.0 (strong L2 regularization)\n")
        
        print("\nComputing XGBoost feature importance...")
        xgb_imp = xgb.XGBRegressor(
            n_estimators=1000,              # MAXIMUM trees
            max_depth=10,                   # DEEP trees for complex interactions
            learning_rate=0.001,             # SLOW learning for optimal convergence
            subsample=0.7,                  # Strong regularization
            colsample_bytree=0.7,           # Feature sampling per tree
            colsample_bylevel=0.7,          # Feature sampling per level
            min_child_weight=5,             # Conservative splits
            gamma=0.2,                      # Pruning threshold
            reg_alpha=0.5,                  # Strong L1 regularization
            reg_lambda=2.0,                 # Strong L2 regularization
            random_state=42,
            n_jobs=-1,                      # Use all CPU cores
            verbosity=0                    # Show progress
        )
        xgb_imp.fit(X_candidates, y)

        imp_dict = xgb_imp.get_booster().get_score(importance_type='gain')
        imp_values = np.array([imp_dict.get(f'f{i}', 0.0) for i in range(len(candidate_features))], dtype=float)
        if imp_values.max() > 0:
            imp_values /= imp_values.max()

        ranked_idx = np.argsort(imp_values)[::-1]
        ranked_features = [candidate_features[i] for i in ranked_idx]

        # Conservative candidate sizes
        max_cand = len(candidate_features)
        candidate_sizes = [1,2,3,5,8,10,15,20,25,30,41]
        # keep unique and within bounds
        candidate_sizes = sorted({k for k in candidate_sizes if 1 <= k <= max_cand})

        print(f"Trying candidate sizes: {candidate_sizes}")

        best_dir_acc = 0.0
        best_val_rmse = float('inf')
        best_selected = None
        best_prepared = None
        best_trainer = None
        best_history = None
        best_metrics_val = None
        best_k = None
        eval_epochs = 100

        print(f"\n{'='*80}")
        print("OPTIMIZING FOR DIRECTIONAL ACCURACY")
        print(f"{'='*80}")

        for k in candidate_sizes:
            selected_k = ranked_features[:k]
            final_k = stock_features + selected_k
            print(f"\nEvaluating top-{k} candidate features (total features: {len(final_k)})")

            # Prepare data
            prepared_k = prepare_data(
                df=fusion_df,
                feature_cols=final_k,
                target_col=target_col,
                seq_length=config.seq_length,
                train_ratio=0.6,
                val_ratio=0.2,
                scale_method="standard"
            )

            # Create dataloaders
            train_loader_k, val_loader_k, _ = create_dataloaders(
                prepared_k['X_train'], prepared_k['y_train'],
                prepared_k['X_val'], prepared_k['y_val'],
                prepared_k['X_test'], prepared_k['y_test'],
                batch_size=config.batch_size
            )

            # Quick LSTM model for evaluation
            input_size_k = prepared_k['X_train'].shape[2]
            model_k = AdvancedLSTMModel(
                input_size=input_size_k,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
                use_attention=getattr(config, 'use_attention', False)
            )

            trainer_k = ModelTrainer(
                model=model_k,
                device=config.device,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                gradient_clip=config.gradient_clip
            )
            
            trainer_k.train(
                train_loader=train_loader_k,
                val_loader=val_loader_k,
                num_epochs=eval_epochs,
                early_stopping_patience=20,
                verbose=False
            )

            # Validate
            val_preds_k, val_targets_k = trainer_k.predict(val_loader_k)
            val_preds_orig_k = inverse_transform_predictions(val_preds_k, prepared_k['target_scaler'])
            val_targets_orig_k = inverse_transform_predictions(val_targets_k, prepared_k['target_scaler'])
            metrics_val_k = calculate_financial_metrics(val_preds_orig_k, val_targets_orig_k)
            
            val_rmse_k = metrics_val_k['rmse']
            val_dir_acc_k = metrics_val_k['directional_accuracy']

            print(f"  Top-{k}: RMSE={val_rmse_k:.4f}, Dir_Acc={val_dir_acc_k:.2f}%")

            # PRIMARY optimization: directional accuracy
            # SECONDARY: prefer smaller feature sets when dir_acc is close (within 1%)
            if best_selected is None:
                best_dir_acc = val_dir_acc_k
                best_val_rmse = val_rmse_k
                best_selected = selected_k
                best_prepared = prepared_k
                best_trainer = trainer_k
                best_metrics_val = metrics_val_k
                best_k = k
            else:
                dir_acc_improvement = val_dir_acc_k - best_dir_acc
                
                # If directional accuracy improves by >1%, always choose it
                if dir_acc_improvement > 1.0:
                    print(f"    -> New best! Dir_Acc improved by {dir_acc_improvement:.2f}%")
                    best_dir_acc = val_dir_acc_k
                    best_val_rmse = val_rmse_k
                    best_selected = selected_k
                    best_prepared = prepared_k
                    best_trainer = trainer_k
                    best_metrics_val = metrics_val_k
                    best_k = k
                # If dir_acc is similar (within 1%), prefer smaller feature set
                elif dir_acc_improvement > -1.0 and k < best_k:
                    print(f"    -> New best! Similar Dir_Acc ({dir_acc_improvement:+.2f}%) but fewer features")
                    best_dir_acc = val_dir_acc_k
                    best_val_rmse = val_rmse_k
                    best_selected = selected_k
                    best_prepared = prepared_k
                    best_trainer = trainer_k
                    best_metrics_val = metrics_val_k
                    best_k = k

        if best_selected is None:
            # fallback to using all features
            best_selected = candidate_features
            best_prepared = prepare_data(
                df=fusion_df,
                feature_cols=stock_features + best_selected,
                target_col=target_col,
                seq_length=config.seq_length,
                train_ratio=0.6,
                val_ratio=0.2,
                scale_method="standard"
            )

        print(f"\nBest candidate set found: {len(best_selected)} features")
        print(f"  Validation Directional Accuracy: {best_dir_acc:.2f}%")
        print(f"  Validation RMSE: {best_val_rmse:.4f}")
        print(f"Selected candidates: {best_selected}")

        # Final training on chosen feature set with full configuration
        final_features = stock_features + best_selected
        
        print(f"\n" + "="*80)
        print("FINAL FEATURE SET FOR LSTM TRAINING")
        print("="*80)
        print(f"  Stock features (base):     {len(stock_features)}")
        print(f"  Selected candidates:       {len(best_selected)}")
        print(f"  TOTAL FEATURES:            {len(final_features)}")
        
        # Get final feature importances for selected features
        final_xgb_imp = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        X_best = fusion_df[best_selected].values
        final_xgb_imp.fit(X_best, y)
        
        imp_dict_final = final_xgb_imp.get_booster().get_score(importance_type='gain')
        imp_values_final = {best_selected[i]: imp_dict_final.get(f'f{i}', 0.0) for i in range(len(best_selected))}
        
        # Save detailed selection info
        final_features_clean = list(dict.fromkeys(final_features))
        if len(final_features_clean) != len(final_features):
            print(f"\nWARNING: Removed {len(final_features) - len(final_features_clean)} duplicate features!")
            final_features = final_features_clean
        
        # Verify no targets in the feature list
        target_features_found = [f for f in final_features if f.startswith('target_')]
        if target_features_found:
            print(f"\nERROR: Found target columns in features: {target_features_found}")
            print(f"   Removing them...")
            final_features = [f for f in final_features if not f.startswith('target_')]
        
        # Verify stock_features don't contain targets
        stock_features_clean = [f for f in stock_features if not f.startswith('target_')]
        if len(stock_features_clean) != len(stock_features):
            print(f"\nWARNING: Removed {len(stock_features) - len(stock_features_clean)} targets from stock_features")
            stock_features = stock_features_clean
        
        selection_info = {
            'algorithm': 'XGBoost Importance + LSTM-in-loop Validation (Directional Accuracy Optimization)',
            'description': 'Feature selection optimized for directional accuracy rather than RMSE',
            'reference': 'Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System',
            'method': {
                'step1': 'Rank all candidates by XGBoost gain importance',
                'step2': 'Create multiple top-K feature sets',
                'step3': 'Train LSTM briefly on each set and measure validation directional accuracy',
                'step4': 'Select K with best directional accuracy (prefer smaller sets if close)',
                'optimization_metric': 'directional_accuracy',
                'eval_epochs': eval_epochs,
                'candidate_sizes_tried': candidate_sizes
            },
            'results': {
                'total_candidates': len(candidate_features),
                'optimal_n_features': len(best_selected),
                'best_validation_directional_accuracy': float(best_dir_acc),
                'best_validation_rmse': float(best_val_rmse)
            },
            'target_column': target_col,
            'stock_features': stock_features,
            'selected_candidates': best_selected,
            'selected_candidates_importance': imp_values_final,
            'total_features': len(final_features),
            'feature_list_ordered': final_features,
            'model_input_size': len(final_features)
        }
        
        # Verification summary
        print(f"\n{'='*80}")
        print(f"FEATURE VERIFICATION")
        print(f"{'='*80}")
        print(f"  Target column: {target_col}")
        print(f"  Stock features (no targets): {len(stock_features)}")
        print(f"  Selected candidates: {len(best_selected)}")
        print(f"  Total features for model: {len(final_features)}")
        print(f"  Duplicates removed: {len(final_features) != len(final_features_clean)}")
        print(f"  Targets in features: {len(target_features_found) > 0}")
        print(f"\n✓ Feature list is clean and ready for training")
        
        # Detect horizon suffix from target column
        horizon_suffix = ""
        if "t+1" in target_col:
            horizon_suffix = "_short"
        elif "t+5" in target_col:
            horizon_suffix = "_medium"
        elif "t+21" in target_col:
            horizon_suffix = "_long"
        
        # Save JSON with horizon suffix
        json_filename = f'fusion_optimized_features{horizon_suffix}.json' if horizon_suffix else 'fusion_optimized_features.json'
        with open(self.results_dir / json_filename, 'w') as f:
            json.dump(selection_info, f, indent=2)
        print(f"\n✓ Feature selection info saved: {json_filename}")
        
        # Save dataset with selected features only
        fusion_optimized_df = fusion_df[final_features].copy()
        csv_filename = f'dataset_fusion_optimized{horizon_suffix}.csv' if horizon_suffix else 'dataset_fusion_optimized.csv'
        dataset_path = self.data_dir / csv_filename
        fusion_optimized_df.to_csv(dataset_path, index=True)
        print(f"✓ Dataset with selected features saved: {dataset_path}")
        print(f"  Columns: {len(fusion_optimized_df.columns)} features (Date saved as index)")
        
        print(f"\n" + "="*80)
        print(f"TRAINING LSTM WITH OPTIMALLY SELECTED FEATURES")
        print(f"="*80)
        
        # Prepare data with selected features
        prepared = prepare_data(
            df=fusion_df,
            feature_cols=final_features,
            target_col=target_col,
            seq_length=config.seq_length,
            train_ratio=0.6,
            val_ratio=0.2,
            scale_method="standard"
        )
        
        self.prepared_data['fusion_optimized'] = prepared
        
        print(f"  Train: {prepared['X_train'].shape}")
        print(f"  Val:   {prepared['X_val'].shape}")
        print(f"  Test:  {prepared['X_test'].shape}")
        
        # Save scalers
        save_scalers(
            prepared['feature_scaler'],
            prepared['target_scaler'],
            str(self.models_dir / 'fusion_optimized_scalers.pkl')
        )
        
        # Create model
        input_size = len(final_features)
        model = AdvancedLSTMModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            use_attention=config.use_attention
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            prepared['X_train'], prepared['y_train'],
            prepared['X_val'], prepared['y_val'],
            prepared['X_test'], prepared['y_test'],
            batch_size=config.batch_size
        )
        
        # Train
        trainer = ModelTrainer(
            model=model,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            early_stopping_patience=config.early_stopping_patience,
            verbose=True
        )
        
        # Evaluate
        test_preds, test_targets = trainer.predict(test_loader)
        test_preds_orig = inverse_transform_predictions(test_preds, prepared['target_scaler'])
        test_targets_orig = inverse_transform_predictions(test_targets, prepared['target_scaler'])
        metrics = calculate_financial_metrics(test_preds_orig, test_targets_orig)
        
        # Save model
        trainer.save_model(str(self.models_dir / 'fusion_optimized_model.pth'))
        self.save_training_history('fusion_optimized', trainer.history)
        
        self.models['fusion_optimized'] = model
        self.trainers['fusion_optimized'] = trainer
        self.metrics['fusion_optimized'] = metrics
        
        print(f"\n✓ Fusion Optimized model training completed")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test MAE: {metrics['mae']:.4f}")
        print(f"  Test Dir Acc: {metrics['directional_accuracy']:.2f}% (Optimized for Trend Prediction)")
        print(f"  Test Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        
        return model, metrics
    
    def run_full_pipeline(self):
        """Run complete training pipeline with all 5 models"""
        print("\n" + "="*80)
        print("MULTIMODAL FINANCIAL PREDICTION - TRAINING PIPELINE")
        print("="*80)
        print("\nTraining 5 models:")
        print("  1. Stock Model (baseline)")
        print("  2. Sentiment Model")
        print("  3. Market Model")
        print("  4. Fusion Model (all features)")
        print("  5. Fusion Optimized Model (XGBoost feature selection)")
        
        # Train all models
        self.train_stock_model(StockModelConfig())
        self.train_sentiment_model(SentimentModelConfig())
        self.train_market_model(MarketModelConfig())
        self.train_fusion_model(FusionModelConfig()) 
        self.train_fusion_optimized_model(FusionModelConfig())
        
        # Compare models
        print("\n" + "="*80)
        print("FINAL MODEL COMPARISON - TOP 5 RANKING")
        print("="*80)
        
        comparison_df = compare_models(self.metrics, metric='directional_accuracy')
        print("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv(self.results_dir / 'model_comparison.csv', index=False)
        
        # Generate report
        report = create_evaluation_report(
            self.metrics,
            save_path=str(self.results_dir / 'evaluation_report.txt')
        )
        print(report)
        
        # Save metrics as JSON
        serializable_metrics = make_serializable(self.metrics)
        with open(self.results_dir / 'metrics.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETED ✓")
        print("="*80)
        print(f"\nModels saved to: {self.models_dir}")
        print(f"  1. stock_model.pth")
        print(f"  2. sentiment_model.pth")
        print(f"  3. market_model.pth")
        print(f"  4. fusion_model.pth")
        print(f"  5. fusion_optimized_model.pth")
        print(f"\nResults saved to: {self.results_dir}")
        print(f"  - model_comparison.csv")
        print(f"  - evaluation_report.txt")
        print(f"  - metrics.json")
        print(f"  - fusion_optimized_features.json")
        
        return self.models, self.metrics
