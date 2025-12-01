import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from trading_agent import ModelPredictor, ModelValidator, VolatilityScaler


def demo_volatility_scaling():
    """Demonstrate volatility scaling analysis"""
    print("\n" + "="*80)
    print("STEP 1: VOLATILITY SCALING ANALYSIS")
    print("="*80)
    
    # Create and fit scaler
    scaler = VolatilityScaler(base_horizon=1)
    df = pd.read_csv('data/processed/fusion/dataset_stock.csv')
    scaler.fit(df)
    
    # Show scaling analysis
    print("\nScaling Analysis:")
    analysis = scaler.analyze_scaling()
    print(analysis.to_string(index=False))
    
    return scaler


def demo_predictions():
    """Demonstrate predictions with different models"""
    print("\n" + "="*80)
    print("STEP 2: GENERATING PREDICTIONS")
    print("="*80)
    
    # Load datasets
    df_stock = pd.read_csv('data/processed/fusion/dataset_stock.csv')
    df_sentiment = pd.read_csv('data/processed/fusion/dataset_sentiment.csv')
    
    models = [
        {
            'name': 'sentiment_short',
            'dataset': df_sentiment,
            'horizon': 1,
            'description': 'Sentiment Short (1d)'
        },
        {
            'name': 'stock_medium',
            'dataset': df_stock,
            'horizon': 5,
            'description': 'Stock Medium (5d)'
        },
        {
            'name': 'stock_long',
            'dataset': df_stock,
            'horizon': 21,
            'description': 'Stock Long (21d)'
        }
    ]
    
    predictions = {}
    
    for model_info in models:
        try:
            print(f"\n{model_info['description']}:")
            predictor = ModelPredictor(
                f"models/{model_info['name']}_model.pth",
                f"models/{model_info['name']}_scalers.pkl"
            )
            
            pred = predictor.predict_from_dataframe(model_info['dataset'])
            predictions[model_info['name']] = {
                'prediction': pred['predicted_return'],
                'horizon_days': model_info['horizon'],
                'direction': pred['direction']
            }
            
            print(f"  Prediction: {pred['predicted_return']:+.4f} ({pred['predicted_return_pct']:+.2f}%)")
            print(f"  Direction:  {pred['direction']}")
            print(f"  Signal:     {predictor.generate_signal(pred['predicted_return'])}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    return predictions


def demo_volatility_correction(scaler, predictions):
    """Apply volatility corrections to predictions"""
    print("\n" + "="*80)
    print("STEP 3: APPLYING VOLATILITY CORRECTIONS")
    print("="*80)
    
    corrected = scaler.correct_model_predictions(predictions, verbose=True)
    
    return corrected


def demo_validation():
    """Run validation against real data"""
    print("\n" + "="*80)
    print("STEP 4: VALIDATION AGAINST REAL DATA")
    print("="*80)
    
    validator = ModelValidator(
        tesla_data_path='data/raw/tesla-stock.csv',
        stock_dataset_path='data/processed/fusion/dataset_stock.csv',
        sentiment_dataset_path='data/processed/fusion/dataset_sentiment.csv',
        models_dir='models'
    )
    
    # Define models to validate
    top_models = [
        {
            'name': 'sentiment_short',
            'dataset': pd.read_csv('data/processed/fusion/dataset_sentiment.csv'),
            'fecha_objetivo': '2025-04-18',
            'descripcion': 'Sentiment Short (1d)',
            'expected_acc': 56.0,
            'horizon': 'short'
        },
        {
            'name': 'stock_medium',
            'dataset': pd.read_csv('data/processed/fusion/dataset_stock.csv'),
            'fecha_objetivo': '2025-04-24',
            'descripcion': 'Stock Medium (5d)',
            'expected_acc': 55.0,
            'horizon': 'medium'
        },
        {
            'name': 'stock_long',
            'dataset': pd.read_csv('data/processed/fusion/dataset_stock.csv'),
            'fecha_objetivo': '2025-05-16',
            'descripcion': 'Stock Long (21d)',
            'expected_acc': 54.0,
            'horizon': 'long'
        }
    ]
    
    results = validator.run_validation(
        simulation_date='2025-04-17',
        cutoff_date='2025-04-16',
        top_models=top_models,
        verbose=True
    )
    
    return results


def main():
    """Run complete demo"""
    print("\n" + "="*80)
    print("TRADING AGENT DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how to use the trading agent components:")
    print("  - ModelPredictor: Generate predictions")
    print("  - VolatilityScaler: Apply volatility corrections")
    print("  - ModelValidator: Validate against real data")
    print()
    
    try:
        # Step 1: Volatility scaling
        scaler = demo_volatility_scaling()
        
        # Step 2: Generate predictions
        predictions = demo_predictions()
        
        # Step 3: Apply volatility corrections
        if predictions:
            corrected = demo_volatility_correction(scaler, predictions)
        
        # Step 4: Validation
        validation_results = demo_validation()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Components Demonstrated:")
        print("  ✓ Volatility scaling analysis (√t)")
        print("  ✓ Multi-horizon predictions (1d, 5d, 21d)")
        print("  ✓ Volatility corrections to predictions")
        print("  ✓ Validation against real market data")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
