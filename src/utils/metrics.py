"""Evaluation metrics for model performance"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class MetricsCalculator:
    """Calculate various performance metrics"""
    
    def __init__(self, num_classes: int = 5):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of prediction classes
        """
        self.num_classes = num_classes
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate overall accuracy
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return accuracy_score(y_true.flatten(), y_pred.flatten())
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (up/down/neutral)
        
        Args:
            y_true: True labels (0-4)
            y_pred: Predicted labels (0-4)
            
        Returns:
            Directional accuracy
        """
        # Convert to directional: 0-1 -> down, 2 -> neutral, 3-4 -> up
        def to_direction(labels):
            directions = np.zeros_like(labels)
            directions[labels <= 1] = 0  # down
            directions[labels == 2] = 1  # neutral
            directions[labels >= 3] = 2  # up
            return directions
        
        true_dir = to_direction(y_true)
        pred_dir = to_direction(y_pred)
        
        return accuracy_score(true_dir.flatten(), pred_dir.flatten())
    
    def calculate_per_class_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate accuracy for each class separately
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with accuracy per class
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        class_accuracies = {}
        for class_idx in range(self.num_classes):
            # Find samples where true label is this class
            mask = y_true_flat == class_idx
            if mask.sum() > 0:
                # Calculate accuracy only for this class
                class_acc = accuracy_score(y_true_flat[mask], y_pred_flat[mask])
                class_accuracies[f'accuracy_class_{class_idx}'] = class_acc
            else:
                class_accuracies[f'accuracy_class_{class_idx}'] = 0.0
        
        return class_accuracies
    
    def calculate_actionable_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     hold_class: int = 1) -> Dict[str, float]:
        """
        Calculate accuracy only on actionable classes (excluding hold)
        For 3-class system: 0=sell, 1=hold, 2=buy
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            hold_class: The class index to exclude (default 1 for hold)
            
        Returns:
            Dictionary with actionable accuracy metrics
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Mask for actionable classes (exclude hold)
        actionable_mask = y_true_flat != hold_class
        
        if actionable_mask.sum() > 0:
            actionable_acc = accuracy_score(
                y_true_flat[actionable_mask], 
                y_pred_flat[actionable_mask]
            )
        else:
            actionable_acc = 0.0
        
        # Also calculate accuracy for sell and buy separately
        sell_buy_accuracies = {}
        for class_idx in range(self.num_classes):
            if class_idx != hold_class:
                mask = y_true_flat == class_idx
                if mask.sum() > 0:
                    class_acc = accuracy_score(y_true_flat[mask], y_pred_flat[mask])
                    sell_buy_accuracies[f'accuracy_class_{class_idx}'] = class_acc
                else:
                    sell_buy_accuracies[f'accuracy_class_{class_idx}'] = 0.0
        
        return {
            'actionable_accuracy': actionable_acc,
            'actionable_samples': int(actionable_mask.sum()),
            **sell_buy_accuracies
        }
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate precision, recall, F1 per class
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with precision, recall, f1-score arrays
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true.flatten(), 
            y_pred.flatten(),
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
    
    def calculate_per_coin_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate accuracy per coin
        
        Args:
            y_true: True labels, shape (samples, num_coins)
            y_pred: Predicted labels, shape (samples, num_coins)
            
        Returns:
            Array of accuracies per coin
        """
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            return np.array([accuracy_score(y_true, y_pred)])
        
        num_coins = y_true.shape[1]
        accuracies = np.zeros(num_coins)
        
        for i in range(num_coins):
            accuracies[i] = accuracy_score(y_true[:, i], y_pred[:, i])
        
        return accuracies
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(
            y_true.flatten(), 
            y_pred.flatten(),
            labels=list(range(self.num_classes))
        )
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate all metrics at once
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': self.calculate_accuracy(y_true, y_pred),
            'directional_accuracy': self.calculate_directional_accuracy(y_true, y_pred),
            'confusion_matrix': self.calculate_confusion_matrix(y_true, y_pred)
        }
        
        # Per-class precision, recall, F1
        per_class = self.calculate_per_class_metrics(y_true, y_pred)
        metrics.update({
            f'precision_class_{i}': per_class['precision'][i] 
            for i in range(self.num_classes)
        })
        metrics.update({
            f'recall_class_{i}': per_class['recall'][i] 
            for i in range(self.num_classes)
        })
        metrics.update({
            f'f1_class_{i}': per_class['f1_score'][i] 
            for i in range(self.num_classes)
        })
        
        # Per-class accuracy (new)
        per_class_acc = self.calculate_per_class_accuracy(y_true, y_pred)
        metrics.update(per_class_acc)
        
        # Actionable accuracy (sell and buy only, excluding hold)
        if self.num_classes == 3:
            actionable_metrics = self.calculate_actionable_accuracy(y_true, y_pred, hold_class=1)
            metrics.update(actionable_metrics)
        
        # Per-coin accuracy if applicable
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            per_coin = self.calculate_per_coin_accuracy(y_true, y_pred)
            metrics['per_coin_accuracy'] = per_coin
            metrics['mean_per_coin_accuracy'] = np.mean(per_coin)
        
        return metrics

