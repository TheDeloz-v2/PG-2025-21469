import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import time
from pathlib import Path


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        return False
    
    def load_best_weights(self, model: nn.Module):
        """Load best model weights"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class ModelTrainer:
    """Trainer for LSTM models with time series considerations"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        
        # Loss function
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_time': []
        }
        
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions.squeeze(), y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader) -> Tuple[float, np.ndarray, np.ndarray]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions.squeeze(), y_batch)
                
                total_loss += loss.item()
                n_batches += 1
                
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        return avg_loss, predictions, targets
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Train model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=verbose
        )
        
        print(f"\nTraining on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_preds, val_targets = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track metrics
            epoch_time = time.time() - start_time
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                early_stopping.load_best_weights(self.model)
                break
        
        # Load best weights
        if not early_stopping.early_stop:
            early_stopping.load_best_weights(self.model)
        
        print("="*60)
        print(f"Training completed. Best val loss: {early_stopping.best_loss:.4f}")
        
        return self.history
    
    def predict(self, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)
                
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets)
    
    def save_model(self, save_path: str):
        """Save model state"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, save_path)
    
    def load_model(self, load_path: str):
        """Load model state"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from: {load_path}")

