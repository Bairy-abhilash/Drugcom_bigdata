"""
Confidence Estimator Module
============================

Implements confidence estimation techniques for drug synergy predictions:
- Monte Carlo Dropout
- Ensemble-based confidence
- Calibration methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer that remains active during inference.
    
    This allows uncertainty estimation by performing multiple forward passes
    with different dropout masks.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Dropout probability
        """
        super(MCDropout, self).__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode."""
        return F.dropout(x, p=self.p, training=True)


class ConfidenceEstimator:
    """
    Estimates prediction confidence using Monte Carlo Dropout.
    
    Performs multiple forward passes with dropout enabled to estimate
    uncertainty and confidence intervals.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_iterations: int = 50,
        dropout_rate: float = 0.3,
        device: str = "cpu"
    ):
        """
        Args:
            model: Trained GNN model
            n_iterations: Number of MC dropout iterations
            dropout_rate: Dropout probability for MC sampling
            device: Device to run inference on
        """
        self.model = model
        self.n_iterations = n_iterations
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Enable dropout during inference
        self._enable_dropout()
        
        logger.info(
            f"ConfidenceEstimator initialized with {n_iterations} iterations, "
            f"dropout={dropout_rate}"
        )
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, MCDropout)):
                module.train()
    
    def predict_with_confidence(
        self,
        graph,
        drug_pair_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict synergy scores with confidence estimates.
        
        Args:
            graph: DGL graph
            drug_pair_indices: Indices of drug pairs to predict (N x 2)
            
        Returns:
            mean_predictions: Mean synergy scores
            std_predictions: Standard deviation (uncertainty)
            confidence_scores: Confidence percentage (0-100)
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iterations):
                output = self.model(graph, drug_pair_indices)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)  # Shape: (n_iterations, n_pairs)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Convert uncertainty to confidence (0-100%)
        # Lower std = higher confidence
        confidence = self._calculate_confidence(std_pred)
        
        return (
            torch.tensor(mean_pred),
            torch.tensor(std_pred),
            torch.tensor(confidence)
        )
    
    def _calculate_confidence(self, std: np.ndarray) -> np.ndarray:
        """
        Convert standard deviation to confidence percentage.
        
        Uses inverse relationship: lower std = higher confidence
        Normalized to 0-100 scale.
        """
        # Normalize std to 0-1 range (assuming synergy scores are 0-1)
        normalized_std = np.clip(std, 0, 1)
        
        # Convert to confidence (inverse relationship)
        confidence = (1 - normalized_std) * 100
        
        return confidence
    
    def get_prediction_interval(
        self,
        graph,
        drug_pair_indices: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate prediction intervals at specified confidence level.
        
        Args:
            graph: DGL graph
            drug_pair_indices: Drug pairs to predict
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            mean: Mean predictions
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iterations):
                output = self.model(graph, drug_pair_indices)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        mean = np.mean(predictions, axis=0)
        lower = np.percentile(predictions, lower_percentile, axis=0)
        upper = np.percentile(predictions, upper_percentile, axis=0)
        
        return (
            torch.tensor(mean),
            torch.tensor(lower),
            torch.tensor(upper)
        )
    
    def calibrate_confidence(
        self,
        graph,
        drug_pair_indices: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calibrate confidence estimates using validation data.
        
        Args:
            graph: DGL graph
            drug_pair_indices: Drug pairs
            true_labels: Ground truth synergy scores
            
        Returns:
            Calibration metrics dictionary
        """
        mean_pred, std_pred, confidence = self.predict_with_confidence(
            graph, drug_pair_indices
        )
        
        errors = np.abs(mean_pred.numpy() - true_labels.numpy())
        
        # Calculate calibration metrics
        calibration = {
            "mean_confidence": float(np.mean(confidence.numpy())),
            "mean_error": float(np.mean(errors)),
            "correlation_conf_accuracy": float(
                np.corrcoef(confidence.numpy(), -errors)[0, 1]
            )
        }
        
        return calibration


class EnsembleConfidence:
    """
    Ensemble-based confidence estimation using multiple trained models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = "cpu"
    ):
        """
        Args:
            models: List of trained models
            device: Device to run inference on
        """
        self.models = models
        self.device = device
        self.n_models = len(models)
        
        logger.info(f"EnsembleConfidence initialized with {self.n_models} models")
    
    def predict_with_confidence(
        self,
        graph,
        drug_pair_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict using ensemble and calculate confidence.
        
        Args:
            graph: DGL graph
            drug_pair_indices: Drug pairs to predict
            
        Returns:
            mean_predictions: Ensemble mean
            std_predictions: Ensemble standard deviation
            confidence_scores: Confidence percentage
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(graph, drug_pair_indices)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence
        confidence = (1 - np.clip(std_pred, 0, 1)) * 100
        
        return (
            torch.tensor(mean_pred),
            torch.tensor(std_pred),
            torch.tensor(confidence)
        )
    
    def get_voting_confidence(
        self,
        graph,
        drug_pair_indices: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate confidence based on model agreement for binary classification.
        
        Args:
            graph: DGL graph
            drug_pair_indices: Drug pairs
            threshold: Classification threshold
            
        Returns:
            predictions: Final predictions
            agreement_scores: Percentage of models agreeing (0-100)
        """
        votes = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(graph, drug_pair_indices)
                binary_pred = (output > threshold).float()
                votes.append(binary_pred.cpu().numpy())
        
        votes = np.array(votes)
        
        # Calculate majority vote
        mean_vote = np.mean(votes, axis=0)
        final_pred = (mean_vote > 0.5).astype(float)
        
        # Agreement score (how many models agree with majority)
        agreement = np.maximum(mean_vote, 1 - mean_vote) * 100
        
        return (
            torch.tensor(final_pred),
            torch.tensor(agreement)
        )


class TemperatureScaling:
    """
    Calibration method using temperature scaling.
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: Trained model to calibrate
        """
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def calibrate(
        self,
        graph,
        drug_pair_indices: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Find optimal temperature using validation set.
        
        Args:
            graph: DGL graph
            drug_pair_indices: Drug pairs
            labels: Ground truth labels
            lr: Learning rate
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            logits = self.model(graph, drug_pair_indices)
            loss = F.mse_loss(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        logger.info(f"Optimal temperature: {self.temperature.item():.4f}")
    
    def predict_calibrated(
        self,
        graph,
        drug_pair_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict with temperature scaling.
        
        Args:
            graph: DGL graph
            drug_pair_indices: Drug pairs
            
        Returns:
            Calibrated predictions
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph, drug_pair_indices)
            calibrated = logits / self.temperature
        
        return calibrated