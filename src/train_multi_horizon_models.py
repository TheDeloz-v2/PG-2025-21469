import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))

from fusion_pipeline import config as fusion_config
from model import ModelTrainingPipeline


def train_horizon_models(horizon_name, horizon_days):
    """
    Train models for a specific horizon by temporarily modifying the target column
    
    Args:
        horizon_name: Name of horizon ('short', 'medium', 'long')
        horizon_days: Number of days for the horizon (1, 5, 21)
    
    Returns:
        dict: Metrics for all models in this horizon
    """
    print("\n" + "="*80)
    print(f"TRAINING MODELS FOR {horizon_name.upper()} HORIZON ({horizon_days} days)")
    print("="*80)
    
    # Temporarily modify the config target
    original_target = fusion_config.TARGET_COLUMN
    fusion_config.TARGET_COLUMN = f"target_return_t+{horizon_days}"
    
    try:
        # Create pipeline and run
        pipeline = ModelTrainingPipeline()
        models, metrics = pipeline.run_full_pipeline()
        
        # Rename models to include horizon
        renamed_metrics = {}
        for model_name, model_metrics in metrics.items():
            new_name = f"{model_name}_{horizon_name}"
            renamed_metrics[new_name] = model_metrics
            
            # Rename saved model files
            old_model_path = Path(f"models/{model_name}_model.pth")
            new_model_path = Path(f"models/{new_name}_model.pth")
            if old_model_path.exists():
                old_model_path.rename(new_model_path)
                print(f"✓ Renamed {old_model_path.name} → {new_model_path.name}")
            
            # Rename saved scaler files
            old_scaler_path = Path(f"models/{model_name}_scalers.pkl")
            new_scaler_path = Path(f"models/{new_name}_scalers.pkl")
            if old_scaler_path.exists():
                old_scaler_path.rename(new_scaler_path)
                print(f"✓ Renamed {old_scaler_path.name} → {new_scaler_path.name}")
            
            # Rename saved training history files
            old_history_path = Path(f"results/{model_name}_training_history.json")
            new_history_path = Path(f"results/{new_name}_training_history.json")
            if old_history_path.exists():
                old_history_path.rename(new_history_path)
                print(f"✓ Renamed {old_history_path.name} → {new_history_path.name}")
        
        return renamed_metrics
        
    finally:
        # Restore original target
        fusion_config.TARGET_COLUMN = original_target


def main():
    """Main execution function"""
    
    print("="*80)
    print("MULTI-HORIZON MODEL TRAINING")
    print("="*80)
    print(f"\nPrediction horizons:")
    for name, days in fusion_config.TARGET_HORIZONS.items():
        print(f"  {name:8} : {days:2} days")
    print()
    
    # Verify targets exist
    print("\n" + "="*80)
    print("Verifying target columns availability")
    print("="*80)
    
    from pathlib import Path
    script_dir = Path(__file__).parent
    market_path = script_dir / "data" / "processed" / "fusion" / "dataset_market.csv"
    
    if not market_path.exists():
        print(f"ERROR: Dataset not found at {market_path}")
        print("Please run 'python src/main.py' first to generate datasets")
        return
    
    market_df = pd.read_csv(market_path, index_col=0, parse_dates=True)
    available_targets = [c for c in market_df.columns if 'target_return_t+' in c]
    
    print(f"Found {len(available_targets)} target columns:")
    for target in sorted(available_targets):
        n_valid = market_df[target].notna().sum()
        print(f"  {target}: {n_valid} valid samples")
    
    del market_df  # Free memory
    
    # ==================== Train models for each horizon ====================
    all_results = {}
    
    for horizon_name, horizon_days in fusion_config.TARGET_HORIZONS.items():
        target_col = f"target_return_t+{horizon_days}"
        
        if target_col not in available_targets:
            print(f"\nSkipping {horizon_name} - target {target_col} not found")
            continue
        
        # Train models for this horizon
        horizon_metrics = train_horizon_models(horizon_name, horizon_days)
        all_results.update(horizon_metrics)
        
        print(f"\n✓ Completed {horizon_name.upper()} horizon training")
    
    # ==================== Compare results across horizons ====================
    print("\n" + "="*80)
    print("Comparing results across horizons")
    print("="*80)
    
    # Create comparison
    comparison_data = []
    for model_name, metrics in all_results.items():
        row = {'model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('directional_accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("MODEL RANKING BY DIRECTIONAL ACCURACY")
    print("="*80)
    print(comparison_df[['model', 'directional_accuracy', 'sharpe_ratio', 
                         'rmse', 'hit_rate']].to_string(index=False))
    
    # Best model per horizon
    print("\n" + "="*80)
    print("BEST MODEL PER HORIZON")
    print("="*80)
    for horizon_name in fusion_config.TARGET_HORIZONS.keys():
        horizon_models = {k: v for k, v in all_results.items() if k.endswith(f"_{horizon_name}")}
        if not horizon_models:
            continue
        
        best_model_name = max(horizon_models.items(), key=lambda x: x[1]['directional_accuracy'])[0]
        best_metrics = horizon_models[best_model_name]
        
        print(f"\n{horizon_name.upper()} ({fusion_config.TARGET_HORIZONS[horizon_name]} days):")
        print(f"  Best: {best_model_name}")
        print(f"  Dir Acc: {best_metrics['directional_accuracy']:.2f}%")
        print(f"  Sharpe: {best_metrics['sharpe_ratio']:.4f}")
        print(f"  Hit Rate: {best_metrics['hit_rate']:.2f}%")
    
    print("\n" + "="*80)
    print("Saving results")
    print("="*80)
    
    # Save comparison
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    comparison_path = results_dir / "multi_horizon_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Comparison saved to: {comparison_path}")
    
    # Save detailed results
    results_path = results_dir / "multi_horizon_metrics.json"
    json_results = {
        k: {key: float(val) if isinstance(val, (np.floating, np.integer)) else val
            for key, val in v.items()}
        for k, v in all_results.items()
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Detailed results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("✓ MULTI-HORIZON TRAINING COMPLETED")
    print("="*80)
    print(f"\nModels trained: {len(all_results)}")
    print(f"Horizons covered: {len(fusion_config.TARGET_HORIZONS)}")
    print(f"\nAll models saved to: models/")
    print(f"All results saved to: results/")


if __name__ == "__main__":
    main()
