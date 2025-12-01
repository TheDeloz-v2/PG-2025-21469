import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import ModelTrainingPipeline


def main():
    """Train all models"""
    
    print("\n" + "="*80)
    print(" "*20 + "MULTIMODAL FINANCIAL PREDICTION AGENT")
    print(" "*30 + "MODEL TRAINING")
    print("="*80)
    
    pipeline = ModelTrainingPipeline()
    
    models, metrics = pipeline.run_full_pipeline()
    
    print("\n" + "="*80)
    print(" "*30 + "TRAINING COMPLETED ✓")
    print("="*80)


if __name__ == "__main__":
    main()
