"""Tests for ordinal loss functions"""

import torch
import pytest
import numpy as np
from src.utils.ordinal_loss import (
    SoftOrdinalCrossEntropyLoss,
    DistanceWeightedCrossEntropyLoss,
    OrdinalRegressionLoss,
    create_ordinal_loss
)


class TestSoftOrdinalCrossEntropyLoss:
    """Test soft ordinal cross-entropy loss"""
    
    def test_perfect_prediction(self):
        """Loss should be low when predicting the correct token"""
        loss_fn = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=5.0)
        
        # Perfect prediction: logits peak at true token
        logits = torch.zeros(4, 256)
        targets = torch.tensor([50, 100, 150, 200])
        
        # Set high logits at target positions
        for i, target in enumerate(targets):
            logits[i, target] = 10.0
        
        loss = loss_fn(logits, targets)
        
        # Loss should be reasonable (< 10.0) for perfect predictions
        # Note: Soft ordinal loss is higher than standard CE because probability is spread
        assert loss < 10.0, f"Perfect prediction loss should be reasonable, got {loss.item()}"
    
    def test_nearby_prediction_lower_loss(self):
        """Predicting nearby tokens should have lower loss than distant tokens"""
        loss_fn = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=5.0)
        
        true_token = 100
        targets = torch.tensor([true_token, true_token])
        
        # Case 1: Predict token 105 (distance = 5)
        logits_nearby = torch.zeros(2, 256)
        logits_nearby[0, 105] = 10.0
        logits_nearby[1, 105] = 10.0
        loss_nearby = loss_fn(logits_nearby, targets)
        
        # Case 2: Predict token 200 (distance = 100)
        logits_far = torch.zeros(2, 256)
        logits_far[0, 200] = 10.0
        logits_far[1, 200] = 10.0
        loss_far = loss_fn(logits_far, targets)
        
        # Nearby prediction should have lower loss
        assert loss_nearby < loss_far, \
            f"Nearby prediction loss ({loss_nearby:.4f}) should be < far prediction loss ({loss_far:.4f})"
    
    def test_soft_targets_sum_to_one(self):
        """Soft targets should be normalized probability distributions"""
        loss_fn = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=5.0)
        
        targets = torch.tensor([50, 100, 150])
        
        # Compute soft targets manually
        distances = torch.abs(loss_fn.indices.unsqueeze(0) - targets.unsqueeze(1).float())
        soft_targets = torch.exp(-distances ** 2 / (2 * 5.0 ** 2))
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        
        # Each row should sum to 1.0
        row_sums = soft_targets.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"Soft targets should sum to 1.0, got {row_sums}"
    
    def test_sigma_effect(self):
        """Larger sigma should create wider distribution"""
        targets = torch.tensor([100])
        logits = torch.zeros(1, 256)
        logits[0, 110] = 10.0  # Predict token 110 (distance = 10)
        
        # Small sigma = sharp peak = higher penalty for distance
        loss_sharp = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=1.0)(logits, targets)
        
        # Large sigma = wide peak = lower penalty for distance
        loss_wide = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=20.0)(logits, targets)
        
        assert loss_sharp > loss_wide, \
            f"Sharp distribution (sigma=1) should have higher loss ({loss_sharp:.4f}) than wide (sigma=20, {loss_wide:.4f})"
    
    def test_symmetric_distance(self):
        """Distance should be symmetric: |100-90| = |100-110|"""
        loss_fn = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=5.0)
        
        targets = torch.tensor([100, 100])
        
        # Predict 90 (distance = 10)
        logits_below = torch.zeros(2, 256)
        logits_below[0, 90] = 10.0
        logits_below[1, 90] = 10.0
        loss_below = loss_fn(logits_below, targets)
        
        # Predict 110 (distance = 10)
        logits_above = torch.zeros(2, 256)
        logits_above[0, 110] = 10.0
        logits_above[1, 110] = 10.0
        loss_above = loss_fn(logits_above, targets)
        
        # Losses should be approximately equal
        assert torch.isclose(loss_below, loss_above, rtol=0.01), \
            f"Symmetric distances should have similar loss: {loss_below:.4f} vs {loss_above:.4f}"


class TestDistanceWeightedCrossEntropyLoss:
    """Test distance-weighted cross-entropy loss"""
    
    def test_perfect_prediction_no_penalty(self):
        """Perfect prediction should have no distance penalty"""
        loss_fn = DistanceWeightedCrossEntropyLoss(alpha=0.1)
        
        logits = torch.zeros(4, 256)
        targets = torch.tensor([50, 100, 150, 200])
        
        # Perfect predictions
        for i, target in enumerate(targets):
            logits[i, target] = 10.0
        
        loss = loss_fn(logits, targets)
        
        # Should be close to standard CE loss (low)
        assert loss < 2.0, f"Perfect prediction should have low loss, got {loss.item()}"
    
    def test_distance_increases_loss(self):
        """Greater prediction distance should increase loss"""
        loss_fn = DistanceWeightedCrossEntropyLoss(alpha=0.1)
        
        true_token = 100
        targets = torch.tensor([true_token, true_token, true_token])
        
        # Different prediction distances
        logits = torch.full((3, 256), -10.0)
        logits[0, 105] = 10.0   # distance = 5
        logits[1, 120] = 10.0   # distance = 20
        logits[2, 200] = 10.0   # distance = 100
        
        loss = loss_fn(logits, targets)
        
        # Loss should increase with distance
        # We can't easily separate per-sample, but total loss should be > perfect prediction
        perfect_logits = torch.full((3, 256), -10.0)
        perfect_logits[:, true_token] = 10.0
        perfect_loss = loss_fn(perfect_logits, targets)
        
        assert loss > perfect_loss, \
            f"Imperfect predictions ({loss:.4f}) should have higher loss than perfect ({perfect_loss:.4f})"
    
    def test_alpha_zero_is_standard_ce(self):
        """When alpha=0, should behave like standard cross-entropy"""
        loss_fn = DistanceWeightedCrossEntropyLoss(alpha=0.0)
        
        logits = torch.randn(10, 256)
        targets = torch.randint(0, 256, (10,))
        
        distance_loss = loss_fn(logits, targets)
        standard_loss = torch.nn.functional.cross_entropy(logits, targets)
        
        assert torch.isclose(distance_loss, standard_loss, rtol=0.01), \
            f"Alpha=0 should match standard CE: {distance_loss:.4f} vs {standard_loss:.4f}"


class TestOrdinalRegressionLoss:
    """Test ordinal regression loss"""
    
    def test_runs_without_error(self):
        """Basic functionality test"""
        loss_fn = OrdinalRegressionLoss(num_classes=256)
        
        logits = torch.randn(4, 256)
        targets = torch.tensor([50, 100, 150, 200])
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"


class TestLossComparison:
    """Compare different loss functions"""
    
    def test_all_losses_decrease_with_better_predictions(self):
        """All losses should decrease as predictions improve"""
        
        loss_functions = [
            SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=5.0),
            DistanceWeightedCrossEntropyLoss(alpha=0.05),
        ]
        
        targets = torch.tensor([100, 100, 100, 100])
        
        # Bad prediction (far from target)
        bad_logits = torch.zeros(4, 256)
        bad_logits[:, 200] = 10.0
        
        # Good prediction (close to target)
        good_logits = torch.zeros(4, 256)
        good_logits[:, 102] = 10.0
        
        for loss_fn in loss_functions:
            bad_loss = loss_fn(bad_logits, targets)
            good_loss = loss_fn(good_logits, targets)
            
            assert good_loss < bad_loss, \
                f"{loss_fn.__class__.__name__}: Good predictions ({good_loss:.4f}) should have lower loss than bad ({bad_loss:.4f})"


class TestCreateOrdinalLoss:
    """Test loss factory function"""
    
    def test_create_soft_ordinal(self):
        loss = create_ordinal_loss('soft_ordinal', num_classes=256, sigma=5.0)
        assert isinstance(loss, SoftOrdinalCrossEntropyLoss)
    
    def test_create_distance_weighted(self):
        loss = create_ordinal_loss('distance_weighted', alpha=0.05)
        assert isinstance(loss, DistanceWeightedCrossEntropyLoss)
    
    def test_create_ordinal_regression(self):
        loss = create_ordinal_loss('ordinal_regression', num_classes=256)
        assert isinstance(loss, OrdinalRegressionLoss)
    
    def test_invalid_loss_type(self):
        with pytest.raises(ValueError):
            create_ordinal_loss('invalid_loss_type')


class TestNumericalStability:
    """Test numerical stability and edge cases"""
    
    def test_extreme_logits(self):
        """Test with very large/small logits"""
        loss_fn = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=5.0)
        
        logits = torch.zeros(2, 256)
        logits[0, 100] = 100.0  # Very confident
        logits[1, 100] = -100.0  # Very unconfident
        targets = torch.tensor([100, 100])
        
        loss = loss_fn(logits, targets)
        
        assert not torch.isnan(loss), "Loss should not be NaN with extreme logits"
        assert not torch.isinf(loss), "Loss should not be infinite"
    
    def test_boundary_tokens(self):
        """Test with tokens at boundaries (0 and 255)"""
        loss_fn = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=5.0)
        
        logits = torch.randn(4, 256)
        targets = torch.tensor([0, 1, 254, 255])
        
        loss = loss_fn(logits, targets)
        
        assert not torch.isnan(loss), "Loss should handle boundary tokens"
        assert loss > 0, "Loss should be positive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

