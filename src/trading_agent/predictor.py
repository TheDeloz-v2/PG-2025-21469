import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import AdvancedLSTMModel
from model.data_loader import load_scalers


class ModelPredictor:
    """
    Predictor class for loading models and making predictions
    """
    
    def __init__(self, model_path: str, scalers_path: str = None, device: str = "cpu"):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (.pth file)
            scalers_path: Path to saved scalers (.pkl file)
            device: Device to run model on
        """
        self.model_path = Path(model_path)
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Auto-detect scalers path
        if scalers_path is None:
            #(e.g., stock_short_model.pth -> stock_short)
            model_name = self.model_path.stem.replace('_model', '')
            scalers_path = self.model_path.parent / f"{model_name}_scalers.pkl"
        
        self.scalers_path = Path(scalers_path)
        
        if not self.scalers_path.exists():
            raise FileNotFoundError(f"Scalers not found: {scalers_path}")
        
        # Load scalers
        self.feature_scaler, self.target_scaler = load_scalers(str(self.scalers_path))
        
        # Load model architecture and weights
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✓ Model loaded: {self.model_path.name}")
        print(f"✓ Scalers loaded: {self.scalers_path.name}")
    
    def _load_model(self) -> AdvancedLSTMModel:
        """Load model from checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Input size from first LSTM layer
        input_size = state_dict['lstm.weight_ih_l0'].shape[1]
        
        # Hidden size from first LSTM layer (weight_hh has shape [hidden*4, hidden] for LSTM)
        weight_ih_shape = state_dict['lstm.weight_ih_l0'].shape[0]
        
        # Check if bidirectional
        bidirectional = 'lstm.weight_ih_l0_reverse' in state_dict
        
        # Calculate hidden size
        if bidirectional:
            hidden_size = weight_ih_shape // 8  # 4 gates * 2 directions
        else:
            hidden_size = weight_ih_shape // 4  # 4 gates
        
        # Count number of layers
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('lstm.weight_ih_l'))
        
        # Check if attention is used
        use_attention = any('attention' in k for k in state_dict.keys())
        
        # Dropout (can't infer, use default)
        dropout = 0.2
        
        print(f"  Inferred architecture:")
        print(f"    Input size:     {input_size}")
        print(f"    Hidden size:    {hidden_size}")
        print(f"    Num layers:     {num_layers}")
        print(f"    Bidirectional:  {bidirectional}")
        print(f"    Use attention:  {use_attention}")
        
        # Create model with inferred architecture
        model = AdvancedLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def prepare_features(self, df: pd.DataFrame, seq_length: int = 30) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            df: DataFrame with features (last seq_length rows will be used)
            seq_length: Sequence length for LSTM
        
        Returns:
            Prepared sequence array of shape (1, seq_length, n_features)
        """
        # Exclude target and date columns
        exclude_cols = ['Date', 'target_return_t+1', 'target_return_t+5', 'target_return_t+21']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get last seq_length rows
        if len(df) < seq_length:
            raise ValueError(f"Dataframe has {len(df)} rows but seq_length={seq_length}")
        
        recent_data = df[feature_cols].iloc[-seq_length:].values
        
        # Scale features
        scaled_features = self.feature_scaler.transform(recent_data)
        
        # Reshape to for batch prediction
        sequence = scaled_features.reshape(1, seq_length, -1)
        
        return sequence
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction
        
        Args:
            features: Prepared feature array (1, seq_length, n_features)
        
        Returns:
            Tuple of (predicted_return, predicted_return_original_scale)
        """
        # Convert to tensor
        X = torch.FloatTensor(features).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(X)
        
        # Get predicted value
        pred_scaled = prediction.cpu().numpy().flatten()[0]
        
        # Inverse transform to original scale
        pred_original = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
        
        return pred_scaled, pred_original
    
    def predict_from_dataframe(
        self, 
        df: pd.DataFrame, 
        seq_length: int = 30
    ) -> Dict[str, float]:
        """
        Make prediction from DataFrame
        
        Args:
            df: DataFrame with features
            seq_length: Sequence length
        
        Returns:
            Dictionary with prediction details
        """
        # Prepare features
        features = self.prepare_features(df, seq_length)
        
        # Predict
        pred_scaled, pred_original = self.predict(features)
        
        # Determine direction
        direction = "UP ⬆" if pred_original > 0 else ("DOWN ⬇" if pred_original < 0 else "FLAT ➡")
        
        return {
            'predicted_return': pred_original,
            'predicted_return_pct': pred_original * 100,
            'direction': direction,
            'confidence': abs(pred_original)  # Higher absolute return = higher confidence
        }
    
    def generate_signal(
        self,
        predicted_return: float,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01
    ) -> str:
        """
        Generate trading signal based on prediction
        
        Args:
            predicted_return: Predicted return (as decimal, e.g., 0.023 = 2.3%)
            buy_threshold: Threshold for BUY signal
            sell_threshold: Threshold for SELL signal
        
        Returns:
            Signal: 'BUY', 'SELL', or 'HOLD'
        """
        if predicted_return > buy_threshold:
            return "BUY"
        elif predicted_return < sell_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def predict_with_signal(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
        seq_length: int = 30
    ) -> Dict:
        """
        Make prediction and generate signal
        
        Returns:
            Dictionary with prediction and signal
        """
        # Get prediction
        pred_info = self.predict_from_dataframe(df, seq_length)
        
        # Generate signal
        signal = self.generate_signal(
            pred_info['predicted_return'],
            buy_threshold,
            sell_threshold
        )
        
        # Add signal to result
        pred_info['signal'] = signal
        
        return pred_info
