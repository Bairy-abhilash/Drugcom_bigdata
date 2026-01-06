"""
Training module for GNN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dgl
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm

from app.gnn_model.model import DrugSynergyGNN
from app.utils.logger import setup_logger
from app.utils.config import settings

logger = setup_logger(__name__)


class SynergyDataset(Dataset):
    """Dataset for drug synergy training."""
    
    def __init__(self, drug_pairs: List[Tuple[int, int]], synergy_scores: List[float]):
        """
        Initialize dataset.
        
        Args:
            drug_pairs: List of (drug1_idx, drug2_idx) tuples
            synergy_scores: List of corresponding synergy scores
        """
        self.drug_pairs = drug_pairs
        self.synergy_scores = synergy_scores
    
    def __len__(self):
        return len(self.drug_pairs)
    
    def __getitem__(self, idx):
        return {
            'drug1_idx': self.drug_pairs[idx][0],
            'drug2_idx': self.drug_pairs[idx][1],
            'synergy_score': self.synergy_scores[idx]
        }


class GNNTrainer:
    """Trainer for Drug Synergy GNN model."""
    
    def __init__(
        self,
        model: DrugSynergyGNN,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: DrugSynergyGNN model
            graph: DGL heterogeneous graph
            node_features: Dictionary of node features
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.graph = graph.to(device)
        self.node_features = {k: v.to(device) for k, v in node_features.items()}
        self.device = device
        
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Initialized GNN Trainer on device: {device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            drug1_idx = batch['drug1_idx'].to(self.device)
            drug2_idx = batch['drug2_idx'].to(self.device)
            target_scores = batch['synergy_score'].to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_scores = self.model.predict_synergy(
                self.graph,
                self.node_features,
                drug1_idx,
                drug2_idx
            )
            
            # Compute loss
            loss = self.criterion(predicted_scores, target_scores)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                drug1_idx = batch['drug1_idx'].to(self.device)
                drug2_idx = batch['drug2_idx'].to(self.device)
                target_scores = batch['synergy_score'].to(self.device).float()
                
                # Predict
                predicted_scores = self.model.predict_synergy(
                    self.graph,
                    self.node_features,
                    drug1_idx,
                    drug2_idx
                )
                
                # Compute loss
                loss = self.criterion(predicted_scores, target_scores)
                total_loss += loss.item()
                
                all_predictions.extend(predicted_scores.cpu().numpy())
                all_targets.extend(target_scores.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        
        # Pearson correlation
        corr = np.corrcoef(all_predictions, all_targets)[0, 1]
        
        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'correlation': corr
        }
        
        return avg_loss, metrics
    
    def train(
        self,
        train_dataset: SynergyDataset,
        val_dataset: SynergyDataset,
        num_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        save_dir: str = 'models'
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary of training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_corr': []
        }
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_metrics['mae'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_corr'].append(val_metrics['correlation'])
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"Val Corr: {val_metrics['correlation']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model
                model_path = save_path / 'best_model.pth'
                self.save_model(model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save training history
        history_path = save_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")