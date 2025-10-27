"""Ordinal/Distance-Weighted Loss Functions for Token Prediction

For tokenized continuous values (like prices), predicting nearby tokens should 
have lower loss than predicting distant tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftOrdinalCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with soft targets based on distance from true token.
    
    Instead of hard one-hot target [0, 0, 1, 0, 0] for token 2,
    creates soft distribution like [0.1, 0.3, 1.0, 0.3, 0.1] (normalized)
    
    Args:
        num_classes: Number of token classes (256)
        sigma: Controls how quickly probability decays with distance
               - sigma=1: very sharp peak (almost one-hot)
               - sigma=5: moderate spread
               - sigma=10: wide spread
        label_smoothing: Additional uniform smoothing (optional)
    """
    
    def __init__(self, num_classes: int = 256, sigma: float = 5.0, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.label_smoothing = label_smoothing
        
        # Pre-compute distance matrix: [num_classes, num_classes]
        # distances[i, j] = |i - j|
        indices = torch.arange(num_classes).float()
        self.register_buffer('indices', indices)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - raw model outputs
            targets: (batch,) - true token indices (0-255)
        
        Returns:
            loss: scalar
        """
        B, C = logits.shape
        assert C == self.num_classes
        
        # Compute distance from each target to all classes
        # distances[b, c] = |targets[b] - c|
        targets_expanded = targets.unsqueeze(1).float()  # (batch, 1)
        indices = self.indices.to(logits.device)  # Ensure same device
        distances = torch.abs(indices.unsqueeze(0) - targets_expanded)  # (batch, num_classes)
        
        # Gaussian decay: weight = exp(-distance^2 / (2*sigma^2))
        soft_targets = torch.exp(-distances ** 2 / (2 * self.sigma ** 2))  # (batch, num_classes)
        
        # Normalize to sum to 1
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        
        # Optional label smoothing
        if self.label_smoothing > 0:
            uniform = torch.ones_like(soft_targets) / self.num_classes
            soft_targets = (1 - self.label_smoothing) * soft_targets + self.label_smoothing * uniform
        
        # KL divergence between soft targets and predicted probabilities
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        
        return loss


class DistanceWeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy but with loss weighted by prediction distance.
    
    loss = -log(p_true) * (1 + alpha * |pred_token - true_token|)
    
    Args:
        alpha: Weight for distance penalty (0 = standard CE, higher = more penalty)
    """
    
    def __init__(self, alpha: float = 0.05, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,)
        
        Returns:
            loss: scalar
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Get predicted tokens
        pred_tokens = torch.argmax(logits, dim=1)
        
        # Distance penalty
        distance = torch.abs(pred_tokens - targets).float()
        weight = 1.0 + self.alpha * distance
        
        # Weighted loss
        weighted_loss = (ce_loss * weight).mean()
        
        return weighted_loss


class OrdinalRegressionLoss(nn.Module):
    """
    Treat token prediction as ordinal regression.
    
    For token k, we want:
    - P(y <= k) should be high if k >= true_token
    - P(y <= k) should be low if k < true_token
    
    Uses cumulative link model with log-sigmoid.
    """
    
    def __init__(self, num_classes: int = 256):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,)
        """
        B = logits.size(0)
        
        # Compute cumulative logits
        cumulative_logits = torch.cumsum(logits, dim=1)
        
        # For each sample, compute ordinal loss
        losses = []
        for b in range(B):
            target = targets[b].item()
            cum_log = cumulative_logits[b]
            
            # For classes < target: want P(y <= k) to be low → maximize -log_sigmoid(cum_log[k])
            # For classes >= target: want P(y <= k) to be high → maximize log_sigmoid(cum_log[k])
            
            loss = 0
            for k in range(self.num_classes - 1):
                if k < target:
                    # Want low cumulative probability
                    loss += -F.logsigmoid(-cum_log[k])
                else:
                    # Want high cumulative probability
                    loss += -F.logsigmoid(cum_log[k])
            
            losses.append(loss)
        
        return torch.stack(losses).mean()


def create_ordinal_loss(loss_type: str = 'soft_ordinal', **kwargs):
    """
    Factory function to create ordinal loss
    
    Args:
        loss_type: 'soft_ordinal', 'distance_weighted', or 'ordinal_regression'
        **kwargs: Loss-specific parameters
    
    Returns:
        Loss module
    """
    if loss_type == 'soft_ordinal':
        return SoftOrdinalCrossEntropyLoss(**kwargs)
    elif loss_type == 'distance_weighted':
        return DistanceWeightedCrossEntropyLoss(**kwargs)
    elif loss_type == 'ordinal_regression':
        return OrdinalRegressionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

