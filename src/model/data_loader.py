import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List
import pickle


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(
    data: np.ndarray,
    seq_length: int,
    target_col: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training
    
    Args:
        data: Input data array (n_samples, n_features)
        seq_length: Length of sequences
        target_col: Column index of target variable
        
    Returns:
        X: Sequences (n_sequences, seq_length, n_features-1)
        y: Targets (n_sequences,)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Features: all columns except target
        if target_col == -1:
            x = data[i:(i + seq_length), :-1]
        else:
            x = np.delete(data[i:(i + seq_length), :], target_col, axis=1)
        
        # Target: next value
        y_val = data[i + seq_length, target_col]
        
        X.append(x)
        y.append(y_val)
    
    return np.array(X), np.array(y)


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_length: int = 60,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    scale_method: str = "standard"
) -> dict:
    """
    Prepare data for training
    
    Args:
        df: Input dataframe
        feature_cols: List of feature column names
        target_col: Target column name
        seq_length: Sequence length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        scale_method: "standard" or "minmax"
        
    Returns:
        Dictionary containing prepared data and scalers
    """
    # Split data temporally
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Initialize scalers
    if scale_method == "standard":
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else:
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    
    # Scale features
    train_features = feature_scaler.fit_transform(train_df[feature_cols])
    val_features = feature_scaler.transform(val_df[feature_cols])
    test_features = feature_scaler.transform(test_df[feature_cols])
    
    # Scale target
    train_target = target_scaler.fit_transform(train_df[[target_col]].values)
    val_target = target_scaler.transform(val_df[[target_col]].values)
    test_target = target_scaler.transform(test_df[[target_col]].values)
    
    # Combine features and target
    train_data = np.concatenate([train_features, train_target], axis=1)
    val_data = np.concatenate([val_features, val_target], axis=1)
    test_data = np.concatenate([test_features, test_target], axis=1)
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_length)
    X_val, y_val = create_sequences(val_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'target_col': target_col
    }


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def inverse_transform_predictions(
    predictions: np.ndarray,
    target_scaler: StandardScaler
) -> np.ndarray:
    """
    Inverse transform scaled predictions to original scale
    
    Args:
        predictions: Scaled predictions
        target_scaler: Fitted scaler
        
    Returns:
        Predictions in original scale
    """
    predictions = predictions.reshape(-1, 1)
    return target_scaler.inverse_transform(predictions).flatten()


def save_scalers(
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    save_path: str
):
    """Save scalers to disk"""
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    with open(save_path, 'wb') as f:
        pickle.dump(scalers, f)


def load_scalers(load_path: str) -> Tuple[StandardScaler, StandardScaler]:
    """Load scalers from disk"""
    with open(load_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers['feature_scaler'], scalers['target_scaler']

