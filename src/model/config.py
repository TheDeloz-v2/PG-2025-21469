from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for LSTM models"""
    
    # Architecture
    input_size: int = 10
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 200
    weight_decay: float = 1e-4
    
    # Sequences
    seq_length: int = 21
    
    # Regularization
    gradient_clip: float = 1.0
    early_stopping_patience: int = 80

    # Device
    device: str = "cpu"
    

@dataclass
class StockModelConfig(ModelConfig):
    """Configuration for Stock LSTM model"""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class MarketModelConfig(ModelConfig):
    """Configuration for Market LSTM model"""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3


@dataclass
class SentimentModelConfig(ModelConfig):
    """Configuration for Sentiment LSTM model"""
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class FusionModelConfig(ModelConfig):
    """Configuration for Fusion LSTM model"""
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    use_attention: bool = True
    num_attention_heads: int = 4
