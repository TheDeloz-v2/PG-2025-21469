import torch
import torch.nn as nn


class AdvancedLSTMModel(nn.Module):
    """
    LSTM model with optional attention mechanism for financial time series prediction
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True
    ):
        super(AdvancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_size // 2, 1)
            )
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def attention_net(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism
        
        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_size)
            
        Returns:
            Context vector (batch_size, hidden_size)
        """
        # Calculate attention weights
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Calculate context vector
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        device = x.device
        num_directions = 2 if self.bidirectional else 1
        
        # Initialize hidden states
        h0 = torch.zeros(
            self.num_layers * num_directions,
            x.size(0),
            self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.num_layers * num_directions,
            x.size(0),
            self.hidden_size
        ).to(device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Layer normalization
        out = self.layer_norm(out)
        
        if self.use_attention:
            out = self.attention_net(out)
        else:
            out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out