import pytest
import torch
import torch.nn as nn

from src.model.v4_transformer import CryptoTransformerV4


@pytest.fixture
def config():
    """Test configuration matching updated design"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP'],
            'target_coin': 'XRP'
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 9
        },
        'model': {
            'vocab_size': 3,  # For simplicity in testing
            'num_classes': 3,
            'd_model': 128,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'coin_embedding_dim': 16,
            'positional_encoding': 'sinusoidal',
            'max_seq_len': 1024,
            'multi_horizon_enabled': False,
            'btc_attention_enabled': True,
            'time_features_enabled': False
        }
    }


def test_model_accepts_correct_input_shape(config):
    """Test that model accepts input shape (batch, 24, num_coins, 9)"""
    model = CryptoTransformerV4(config)

    batch_size = 16
    input_length = 24
    num_coins = 3  # Use model's num_coins, not config
    num_channels = 9

    # Create sample input: (batch, 24, 3, 9)
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))

    # Forward pass should not raise error
    try:
        output = model(x)
        assert isinstance(output, dict), "Model should return a dictionary"
        assert 'horizon_1h' in output, "Model should return horizon_1h prediction"
        assert True, "Model accepted correct input shape"
    except Exception as e:
        pytest.fail(f"Model failed with correct input shape: {e}")


def test_model_output_shape(config):
    """Test that model output shape is correct for multi-horizon predictions"""
    model = CryptoTransformerV4(config)

    batch_size = 16
    input_length = 24
    num_coins = 3
    num_channels = 9
    num_classes = 3

    # Create sample input
    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))

    # Forward pass
    output = model(x)
    
    # Check output structure
    assert isinstance(output, dict), "Model should return a dictionary"
    assert 'horizon_1h' in output, "Model should return horizon_1h prediction"
    
    # Check output shape for horizon_1h
    horizon_1h = output['horizon_1h']['logits']
    assert horizon_1h.shape == (batch_size, num_classes), f"Expected ({batch_size}, {num_classes}), got {horizon_1h.shape}"


def test_model_handles_different_batch_sizes(config):
    """Test that model works with different batch sizes"""
    model = CryptoTransformerV4(config)

    input_length = 24
    num_coins = 3
    num_channels = 9

    for batch_size in [1, 8, 32, 128]:
        x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
        output = model(x)
        
        # Check output structure
        assert isinstance(output, dict), f"Batch size {batch_size}: Model should return a dictionary"
        assert 'horizon_1h' in output, f"Batch size {batch_size}: Model should return horizon_1h prediction"
        
        # Check output shape
        horizon_1h = output['horizon_1h']['logits']
        assert horizon_1h.shape[0] == batch_size, f"Batch size {batch_size}: Expected batch dimension {batch_size}, got {horizon_1h.shape[0]}"


def test_model_outputs_valid_logits(config):
    """Test that model outputs are valid logits (not probabilities)"""
    model = CryptoTransformerV4(config)

    batch_size = 16
    input_length = 24
    num_coins = 3
    num_channels = 9

    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))
    output = model(x)
    
    # Check that outputs are logits (can be negative, don't sum to 1)
    horizon_1h = output['horizon_1h']['logits']
    assert horizon_1h.dtype == torch.float32, "Output should be float32"
    
    # Logits can be negative and don't sum to 1
    assert torch.any(horizon_1h < 0), "Logits should contain negative values"
    
    # Check that softmax would give valid probabilities
    probs = torch.softmax(horizon_1h, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size)), "Softmax should sum to 1"


def test_model_parameters_count_reasonable(config):
    """Test that model has reasonable number of parameters"""
    model = CryptoTransformerV4(config)

    total_params = sum(p.numel() for p in model.parameters())
    
    # Should have reasonable number of parameters (not too few, not too many)
    assert 100_000 < total_params < 10_000_000, f"Model has {total_params} parameters, expected between 100k and 10M"


def test_model_embedding_dimensions(config):
    """Test that model has correct embedding structure"""
    model = CryptoTransformerV4(config)

    # Model should have coin embedding
    assert hasattr(model, 'coin_embedding'), "Model should have coin embedding"
    assert isinstance(model.coin_embedding, nn.Embedding), "Coin embedding should be nn.Embedding"
    
    # Model should have channel embeddings
    assert hasattr(model, 'channel_embeddings'), "Model should have channel embeddings"
    assert isinstance(model.channel_embeddings, nn.ModuleList), "Channel embeddings should be ModuleList"
    assert len(model.channel_embeddings) == 9, "Should have 9 channel embeddings"
    
    # Model should have channel fusion
    assert hasattr(model, 'channel_fusion'), "Model should have channel fusion layer"


def test_model_eval_mode(config):
    """Test that model can be set to eval mode"""
    model = CryptoTransformerV4(config)

    # Set to eval mode
    model.eval()
    assert not model.training, "Model should be in eval mode"

    batch_size = 16
    input_length = 24
    num_coins = 3
    num_channels = 9

    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))

    # Forward pass in eval mode should work
    with torch.no_grad():
        output = model(x)
        assert isinstance(output, dict), "Model should return a dictionary in eval mode"
        assert 'horizon_1h' in output, "Model should return horizon_1h prediction in eval mode"


def test_model_training_mode(config):
    """Test that model can be set to training mode"""
    model = CryptoTransformerV4(config)
    
    model.train()
    assert model.training, "Model should be in training mode"

    batch_size = 16
    input_length = 24
    num_coins = 3
    num_channels = 9

    x = torch.randint(0, 3, (batch_size, input_length, num_coins, num_channels))

    # Forward pass in training mode should work
    output = model(x)
    assert isinstance(output, dict), "Model should return a dictionary in training mode"
    assert 'horizon_1h' in output, "Model should return horizon_1h prediction in training mode"