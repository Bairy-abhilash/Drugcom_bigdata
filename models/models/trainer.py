"""
Training pipeline for drug synergy prediction model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import dgl
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynergyTrainer:
    """Trainer for drug synergy prediction model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: Synergy prediction model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
    
    def train_epoch(
        self,
        g: dgl.DGLHeteroGraph,
        drug1_ids: torch.Tensor,
        drug2_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 64
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            g: DGL heterogeneous graph
            drug1_ids: First drug IDs
            drug2_ids: Second drug IDs
            labels: Synergy labels
            batch_size: Batch size
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create batches
        dataset = TensorDataset(drug1_ids, drug2_ids, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for batch_drug1, batch_drug2, batch_labels in dataloader:
            batch_drug1 = batch_drug1.to(self.device)
            batch_drug2 = batch_drug2.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(g, batch_drug1, batch_drug2)
            loss = self.criterion(predictions, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        g: dgl.DGLHeteroGraph,
        drug1_ids: torch.Tensor,
        drug2_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 64
    ) -> Tuple[float, float]:
        """
        Evaluate model.
        
        Args:
            g: DGL heterogeneous graph
            drug1_ids: First drug IDs
            drug2_ids: Second drug IDs
            labels: Synergy labels
            batch_size: Batch size
            
        Returns:
            Tuple of (loss, pearson correlation)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        # Create batches
        dataset = TensorDataset(drug1_ids, drug2_ids, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch_drug1, batch_drug2, batch_labels in dataloader:
            batch_drug1 = batch_drug1.to(self.device)
            batch_drug2 = batch_drug2.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            predictions = self.model(g, batch_drug1, batch_drug2)
            loss = self.criterion(predictions, batch_labels)
            
            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            num_batches += 1
        
        # Calculate Pearson correlation
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        correlation = np.corrcoef(all_preds, all_labels)[0, 1]
        
        return total_loss / num_batches, correlation
    
    def train(
        self,
        g: dgl.DGLHeteroGraph,
        train_data: Dict,
        val_data: Dict,
        num_epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10
    ):
        """
        Train model.
        
        Args:
            g: DGL heterogeneous graph
            train_data: Dictionary with 'drug1_ids', 'drug2_ids', 'labels'
            val_data: Dictionary with 'drug1_ids', 'drug2_ids', 'labels'
            num_epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(
                g,
                train_data['drug1_ids'],
                train_data['drug2_ids'],
                train_data['labels'],
                batch_size
            )
            
            # Validation
            val_loss, val_corr = self.evaluate(
                g,
                val_data['drug1_ids'],
                val_data['drug2_ids'],
                val_data['labels'],
                batch_size
            )
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_corr'].append(val_corr)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Corr: {val_corr:.4f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/checkpoints/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_