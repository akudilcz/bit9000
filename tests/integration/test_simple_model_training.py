"""Simple model training test to increase coverage"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.v1_transformer import CryptoTransformerV1


def test_early_stopping():
    """Test early stopping functionality using PyTorch Lightning"""
    from pytorch_lightning.callbacks import EarlyStopping
    
    es = EarlyStopping(patience=2, min_delta=0.01, monitor='val_loss')
    assert es.patience == 2
    assert es.min_delta == -0.01  # PyTorch Lightning uses negative min_delta
    assert es.monitor == 'val_loss'


def test_token_predictor_initialization():
    """Test CryptoTransformerV1 model creation"""
    config = {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP']
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 2
        },
        'model': {
            'embedding_dim': 32,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'feedforward_dim': 256,
            'dropout': 0.1
        }
    }
    
    model = CryptoTransformerV1(config)
    
    # Test model was created
    assert model is not None
    assert model.vocab_size == 3
    assert model.input_length == 24
    assert model.output_length == 8
    assert model.num_coins == 3
    assert model.num_channels == 2


def test_token_predictor_forward_pass():
    """Test CryptoTransformerV1 forward pass"""
    config = {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP']
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 2
        },
        'model': {
            'embedding_dim': 32,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'feedforward_dim': 256,
            'dropout': 0.1
        }
    }
    
    model = CryptoTransformerV1(config)
    
    # Test forward pass with training mode (teacher forcing)
    batch_size = 4
    input_length = 24
    num_coins = 3
    output_length = 8
    
    # Create input: (batch, input_length, num_coins, 2)
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, 2), dtype=torch.long)
    
    # Create targets: (batch, output_length)
    targets = torch.randint(0, 3, (batch_size, output_length), dtype=torch.long)
    
    # Forward pass with teacher forcing
    logits = model(x, targets)
    
    # Check output shape
    assert logits.shape == (batch_size, output_length, 3)  # (batch, output_length, vocab_size)


def test_token_predictor_inference():
    """Test CryptoTransformerV1 inference mode"""
    config = {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP']
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 2
        },
        'model': {
            'embedding_dim': 32,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'feedforward_dim': 256,
            'dropout': 0.1
        }
    }
    
    model = CryptoTransformerV1(config)
    model.eval()
    
    # Test inference mode (no targets)
    batch_size = 2
    input_length = 24
    num_coins = 3
    
    # Create input: (batch, input_length, num_coins, 2)
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, 2), dtype=torch.long)
    
    # Forward pass without targets
    logits = model(x)
    
    # Check output shape (should predict next token only)
    assert logits.shape == (batch_size, 1, 3)  # (batch, 1, vocab_size)


def test_token_predictor_generate():
    """Test CryptoTransformerV1 autoregressive generation"""
    config = {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP']
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 2
        },
        'model': {
            'embedding_dim': 32,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'feedforward_dim': 256,
            'dropout': 0.1
        }
    }
    
    model = CryptoTransformerV1(config)
    
    # Test autoregressive generation
    batch_size = 2
    input_length = 24
    num_coins = 3
    
    # Create input: (batch, input_length, num_coins, 2)
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, 2), dtype=torch.long)
    
    # Generate predictions
    predictions = model.generate(x, max_length=8)
    
    # Check output shape
    assert predictions.shape == (batch_size, 8)
    
    # Check that predictions are valid tokens (0, 1, or 2)
    assert torch.all((predictions >= 0) & (predictions < 3))


def test_token_predictor_dimensions():
    """Test that model dimensions are properly configured"""
    config = {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP', 'ADA', 'SOL']
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 2
        },
        'model': {
            'embedding_dim': 64,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'feedforward_dim': 512,
            'dropout': 0.1
        }
    }
    
    model = CryptoTransformerV1(config)
    
    # Verify dimensions
    assert model.embedding_dim == 64
    assert model.d_model == 256
    assert model.nhead == 8
    assert model.num_layers == 4
    assert model.dim_feedforward == 512
    assert model.dropout == 0.1
    
    # Verify coin count
    assert model.num_coins == 5


def test_token_predictor_invalid_dimensions():
    """Test that invalid dimensions raise errors"""
    config = {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP']
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 2
        },
        'model': {
            'embedding_dim': 32,
            'd_model': 129,  # Not divisible by num_heads
            'num_heads': 4,
            'num_layers': 2,
            'feedforward_dim': 256,
            'dropout': 0.1
        }
    }
    
    # Should raise ValueError because d_model is not divisible by num_heads
    with pytest.raises(ValueError, match="d_model.*must be divisible by nhead"):
        model = CryptoTransformerV1(config)

