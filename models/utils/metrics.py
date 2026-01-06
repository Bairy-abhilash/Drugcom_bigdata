"""
Metrics Module
==============

Comprehensive metrics for evaluating drug synergy predictions.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


def calculate_mse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return float(mean_squared_error(y_true, y_pred))


def calculate_mae(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    mse = calculate_mse(y_true, y_pred)
    return float(np.sqrt(mse))


def calculate_pearson_correlation(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation, p-value)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    correlation, p_value = pearsonr(y_true.flatten(), y_pred.flatten())
    return float(correlation), float(p_value)


def calculate_spearman_correlation(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation, p-value)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    correlation, p_value = spearmanr(y_true.flatten(), y_pred.flatten())
    return float(correlation), float(p_value)


def calculate_r2_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate R² (coefficient of determination) score.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return float(r2_score(y_true, y_pred))


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    task: str = "regression"
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for model evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        task: 'regression' or 'classification'
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    if task == "regression":
        metrics["mse"] = calculate_mse(y_true, y_pred)
        metrics["mae"] = calculate_mae(y_true, y_pred)
        metrics["rmse"] = calculate_rmse(y_true, y_pred)
        metrics["r2"] = calculate_r2_score(y_true, y_pred)
        
        pearson_corr, pearson_p = calculate_pearson_correlation(y_true, y_pred)
        metrics["pearson_correlation"] = pearson_corr
        metrics["pearson_p_value"] = pearson_p
        
        spearman_corr, spearman_p = calculate_spearman_correlation(y_true, y_pred)
        metrics["spearman_correlation"] = spearman_corr
        metrics["spearman_p_value"] = spearman_p
        
    elif task == "classification":
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred_binary))
        metrics["precision"] = float(precision_score(y_true, y_pred_binary, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred_binary, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred_binary, zero_division=0))
        
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            metrics["auc_roc"] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
    
    return metrics


class MetricsCalculator:
    """
    Comprehensive metrics calculator with tracking over epochs.
    """
    
    def __init__(self, task: str = "regression"):
        """
        Args:
            task: 'regression' or 'classification'
        """
        self.task = task
        self.history = {
            "train": [],
            "val": [],
            "test": []
        }
    
    def calculate(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate metrics for current predictions.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            Metrics dictionary
        """
        return calculate_metrics(y_true, y_pred, self.task)
    
    def update_history(
        self,
        split: str,
        metrics: Dict[str, float]
    ):
        """
        Update metrics history.
        
        Args:
            split: 'train', 'val', or 'test'
            metrics: Metrics dictionary
        """
        if split not in self.history:
            raise ValueError(f"Invalid split: {split}")
        
        self.history[split].append(metrics)
    
    def get_best_epoch(self, split: str = "val", metric: str = "mse") -> int:
        """
        Get epoch with best metric value.
        
        Args:
            split: Data split
            metric: Metric name
            
        Returns:
            Best epoch index
        """
        if not self.history[split]:
            return 0
        
        values = [m.get(metric, float('inf')) for m in self.history[split]]
        
        # Lower is better for loss metrics
        if metric in ["mse", "mae", "rmse"]:
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))
    
    def get_summary(self, split: str = "test") -> Dict[str, float]:
        """
        Get summary statistics for a split.
        
        Args:
            split: Data split
            
        Returns:
            Summary dictionary
        """
        if not self.history[split]:
            return {}
        
        # Get last epoch metrics
        return self.history[split][-1]


class SynergyMetrics:
    """
    Specialized metrics for drug synergy prediction.
    """
    
    @staticmethod
    def calculate_synergy_score_distribution(
        scores: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyze distribution of synergy scores.
        
        Args:
            scores: Synergy scores
            
        Returns:
            Distribution statistics
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75))
        }
    
    @staticmethod
    def calculate_top_k_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Calculate top-k accuracy for synergy ranking.
        
        Args:
            y_true: True synergy scores
            y_pred: Predicted synergy scores
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy (0-1)
        """
        # Get indices of top-k true synergies
        true_top_k = set(np.argsort(y_true)[-k:])
        
        # Get indices of top-k predicted synergies
        pred_top_k = set(np.argsort(y_pred)[-k:])
        
        # Calculate overlap
        overlap = len(true_top_k.intersection(pred_top_k))
        
        return float(overlap / k)
    
    @staticmethod
    def calculate_ndcg(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            y_true: True synergy scores
            y_pred: Predicted synergy scores
            k: Consider only top-k items
            
        Returns:
            NDCG score
        """
        def dcg(scores, k=None):
            if k is None:
                k = len(scores)
            scores = scores[:k]
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
        
        # Sort by predicted scores
        order = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[order]
        
        # Calculate DCG
        dcg_score = dcg(y_true_sorted, k)
        
        # Calculate ideal DCG
        ideal_order = np.argsort(y_true)[::-1]
        y_true_ideal = y_true[ideal_order]
        idcg_score = dcg(y_true_ideal, k)
        
        if idcg_score == 0:
            return 0.0
        
        return float(dcg_score / idcg_score)