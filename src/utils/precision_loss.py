"""Precision-Focused Loss Functions for High-Quality Trading Signals

These loss functions are specifically designed to maximize precision by heavily
penalizing false positive predictions (wrong BUY/SELL signals).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrecisionFocusedLoss(nn.Module):
    """
    Loss function designed to maximize precision by heavily penalizing false positives.
    
    For trading signals, false positives (wrong BUY signals) are much more costly
    than false negatives (missed opportunities). This loss function addresses that.
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        self.precision_weight = config.get('precision_weight', 3.0)
        self.false_positive_penalty = config.get('false_positive_penalty', 5.0)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - model predictions
            targets: (batch,) - true labels (0=NO-BUY, 1=BUY)
        """
        # Get predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Calculate precision-focused penalty
        batch_size = targets.size(0)
        precision_penalty = torch.zeros_like(ce_loss)
        
        for i in range(batch_size):
            target = targets[i].item()
            pred = preds[i].item()
            
            if target == 1 and pred == 0:  # False negative (missed BUY)
                precision_penalty[i] = 1.0  # Standard penalty
            elif target == 0 and pred == 1:  # False positive (wrong BUY) - HEAVILY PENALIZE
                precision_penalty[i] = self.false_positive_penalty
            else:  # Correct prediction
                precision_penalty[i] = 0.1  # Minimal penalty for correct predictions
        
        # Weighted loss
        weighted_loss = ce_loss * (1 + self.precision_weight * precision_penalty)
        
        return weighted_loss.mean()


class AsymmetricPrecisionLoss(nn.Module):
    """
    Asymmetric loss that treats BUY and NO-BUY predictions differently.
    
    - BUY predictions: High penalty for wrong calls (maximize precision)
    - NO-BUY predictions: Lower penalty for wrong calls (allow more false negatives)
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        self.buy_penalty = config.get('buy_penalty', 10.0)
        self.no_buy_penalty = config.get('no_buy_penalty', 1.0)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 2) - binary classification logits [NO-BUY, BUY]
            targets: (batch,) - true labels (0=NO-BUY, 1=BUY)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        buy_probs = probs[:, 1]  # Probability of BUY
        
        # Asymmetric penalties
        loss = torch.zeros_like(buy_probs)
        
        # For true BUY cases: penalize low BUY probability
        buy_mask = targets == 1
        loss[buy_mask] = -torch.log(buy_probs[buy_mask] + 1e-8) * self.no_buy_penalty
        
        # For true NO-BUY cases: heavily penalize high BUY probability
        no_buy_mask = targets == 0
        loss[no_buy_mask] = -torch.log(1 - buy_probs[no_buy_mask] + 1e-8) * self.buy_penalty
        
        return loss.mean()


class ConfidenceWeightedLoss(nn.Module):
    """
    Loss that weights predictions by model confidence.
    
    High-confidence wrong predictions are penalized more heavily,
    encouraging the model to be more conservative with high-confidence calls.
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.high_confidence_penalty = config.get('high_confidence_penalty', 3.0)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - model predictions
            targets: (batch,) - true labels
        """
        # Get probabilities and confidence
        probs = F.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Confidence-based weighting
        confidence_weights = torch.ones_like(ce_loss)
        
        # High confidence wrong predictions get extra penalty
        wrong_predictions = (preds != targets)
        high_confidence = (max_probs > self.confidence_threshold)
        high_conf_wrong = wrong_predictions & high_confidence
        
        confidence_weights[high_conf_wrong] = self.high_confidence_penalty
        
        # Weighted loss
        weighted_loss = ce_loss * confidence_weights
        
        return weighted_loss.mean()


class PrecisionRecallLoss(nn.Module):
    """
    Direct optimization of precision-recall tradeoff.
    
    Uses F-beta score with beta < 1 to emphasize precision over recall.
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        self.beta = config.get('beta', 0.5)  # beta < 1 emphasizes precision
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 2) - binary classification logits
            targets: (batch,) - true labels (0=NO-BUY, 1=BUY)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        buy_probs = probs[:, 1]
        
        # Convert to binary predictions
        preds = (buy_probs > 0.5).float()
        
        # Calculate precision and recall components
        true_positives = (preds * targets).sum()
        false_positives = (preds * (1 - targets)).sum()
        false_negatives = ((1 - preds) * targets).sum()
        
        # Precision
        precision = true_positives / (true_positives + false_positives + 1e-8)
        
        # Recall
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        
        # F-beta score (beta < 1 emphasizes precision)
        f_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + 1e-8)
        
        # Return negative F-beta as loss (maximize F-beta = minimize negative F-beta)
        return -f_beta


def create_precision_loss(loss_type: str = 'asymmetric', config: dict = None):
    """
    Factory function to create precision-focused loss functions.
    
    Args:
        loss_type: 'precision_focused', 'asymmetric', 'confidence_weighted', 'precision_recall'
        config: Configuration dictionary with loss-specific parameters
    
    Returns:
        Loss module optimized for precision
    """
    config = config or {}
    
    if loss_type == 'precision_focused':
        return PrecisionFocusedLoss(config)
    elif loss_type == 'asymmetric':
        return AsymmetricPrecisionLoss(config)
    elif loss_type == 'confidence_weighted':
        return ConfidenceWeightedLoss(config)
    elif loss_type == 'precision_recall':
        return PrecisionRecallLoss(config)
    else:
        raise ValueError(f"Unknown precision loss type: {loss_type}")
