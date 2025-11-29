"""
Evaluation metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float] = None
) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC)

    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (binary)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # AUC (if probabilities provided)
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

    return metrics


class MetricsTracker:
    """
    Track metrics during training
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.history = {
            'train': [],
            'val': []
        }

    def update(self, split: str, metrics: Dict[str, float]):
        """
        Update metrics

        Args:
            split: 'train' or 'val'
            metrics: Dictionary of metrics
        """
        self.history[split].append(metrics)

    def get_best(self, split: str, metric: str = 'accuracy') -> float:
        """
        Get best metric value

        Args:
            split: 'train' or 'val'
            metric: Metric name

        Returns:
            best_value: Best metric value
        """
        if not self.history[split]:
            return 0.0

        values = [m.get(metric, 0.0) for m in self.history[split]]
        return max(values)

    def get_latest(self, split: str, metric: str = 'accuracy') -> float:
        """
        Get latest metric value

        Args:
            split: 'train' or 'val'
            metric: Metric name

        Returns:
            latest_value: Latest metric value
        """
        if not self.history[split]:
            return 0.0

        return self.history[split][-1].get(metric, 0.0)
