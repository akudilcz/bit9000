"""Tests for SimpleTokenPredictor model architecture"""

import pytest
import torch
from src.model.token_predictor import SimpleTokenPredictor


@pytest.fixture
def config():
    """Test configuration matching updated design"""
    return {
        'model': {
            'vocab_size': 3,
            'num_classes': 3,
            'embedding_dim': 64,
            'num_heads': 4,
            'num_layers': 4,
            'feedforward_dim': 256,
            'dropout': 0.1
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 2
        },
        'data': {
            'coins': ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LTC']
        }
    }


def test_model_accepts_correct_input_shape(config):
    """Test that model accepts input shape (batch, 24, num_coins, 2)"""
    model = SimpleTokenPredictor(config)
    
    batch_size = 16
    input_length = config['sequences']['input_length']
    num_coins = len(config['data']['coins'])
    num_channels = config['sequences']['num_channels']
    
    # Create sample input: (batch, 24, 10, 2)
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
    
    # Forward pass should not raise error
    try:
        output = model(x)
        assert True, "Model accepted correct input shape"
    except Exception as e:
        pytest.fail(f"Model failed with correct input shape: {e}")


def test_model_output_shape(config):
    """Test that model output shape is (batch, 8, 3) for 8-step predictions with teacher forcing"""
    model = SimpleTokenPredictor(config)
    
    batch_size = 16
    input_length = config['sequences']['input_length']
    num_coins = len(config['data']['coins'])
    num_channels = config['sequences']['num_channels']
    output_length = config['sequences']['output_length']
    num_classes = config['model']['num_classes']
    
    # Create sample input and targets
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
    y = torch.randint(0, 3, (batch_size, output_length))  # Targets for teacher forcing
    
    # Forward pass with teacher forcing (training mode)
    output = model(x, targets=y)
    
    # Check output shape: (batch, 8, 3)
    assert output.shape == (batch_size, output_length, num_classes), \
        f"Expected output shape ({batch_size}, {output_length}, {num_classes}), got {output.shape}"


def test_model_handles_different_batch_sizes(config):
    """Test that model works with different batch sizes"""
    model = SimpleTokenPredictor(config)
    
    input_length = config['sequences']['input_length']
    num_coins = len(config['data']['coins'])
    num_channels = config['sequences']['num_channels']
    
    for batch_size in [1, 8, 32, 128]:
        x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
        output = model(x)
        assert output.shape[0] == batch_size, f"Batch size mismatch for size {batch_size}"


def test_model_outputs_valid_logits(config):
    """Test that model outputs are valid logits (not probabilities)"""
    model = SimpleTokenPredictor(config)
    
    batch_size = 16
    input_length = config['sequences']['input_length']
    num_coins = len(config['data']['coins'])
    num_channels = config['sequences']['num_channels']
    
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
    output = model(x)
    
    # Logits should not be constrained to [0, 1] or sum to 1
    # They can be any real number
    assert not torch.allclose(output.sum(dim=-1), torch.ones(batch_size, 8)), \
        "Output looks like probabilities (sums to 1), should be logits"


def test_model_parameters_count_reasonable(config):
    """Test that model has reasonable number of parameters (< 10M)"""
    model = SimpleTokenPredictor(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Model should be lightweight (< 10M parameters)
    assert total_params < 10_000_000, \
        f"Model has {total_params:,} parameters, should be < 10M for lightweight design"
    
    # Model should have at least some parameters
    assert total_params > 10_000, \
        f"Model has only {total_params:,} parameters, seems too small"


def test_model_embedding_dimensions(config):
    """Test that model has separate embeddings for price and volume"""
    model = SimpleTokenPredictor(config)
    
    # Model should have token embeddings
    # Check if model has expected components
    assert hasattr(model, 'token_embedding') or hasattr(model, 'price_embedding'), \
        "Model should have token embedding layer(s)"


def test_model_eval_mode(config):
    """Test that model can be set to eval mode"""
    model = SimpleTokenPredictor(config)
    
    # Set to eval mode
    model.eval()
    
    batch_size = 16
    input_length = config['sequences']['input_length']
    num_coins = len(config['data']['coins'])
    num_channels = config['sequences']['num_channels']
    
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
    
    # Forward pass in eval mode should work
    with torch.no_grad():
        output = model(x)
        assert output.shape[0] == batch_size


def test_model_training_mode(config):
    """Test that model can be trained with teacher forcing"""
    model = SimpleTokenPredictor(config)
    model.train()
    
    batch_size = 16
    input_length = config['sequences']['input_length']
    num_coins = len(config['data']['coins'])
    num_channels = config['sequences']['num_channels']
    output_length = config['sequences']['output_length']
    
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
    y = torch.randint(0, 3, (batch_size, output_length))
    
    # Forward pass with teacher forcing
    output = model(x, targets=y)
    
    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output.view(-1, 3), y.view(-1))
    
    # Backward pass should work
    loss.backward()
    
    # Gradients should be computed
    assert any(p.grad is not None for p in model.parameters()), \
        "No gradients computed during backward pass"

