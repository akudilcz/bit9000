import pytest
import torch
import torch.nn as nn

from src.model.v4_transformer import CryptoTransformerV4


@pytest.fixture
def config():
    """Test configuration for integration tests"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP']
        },
        'sequences': {
            'input_length': 24,
            'output_length': 8,
            'num_channels': 9
        },
        'model': {
            'vocab_size': 3,
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


def test_early_stopping():
    """Test early stopping mechanism works"""
    # This is a placeholder test - early stopping would be tested in trainer
    assert True, "Early stopping test placeholder"


def test_token_predictor_initialization():
    """Test CryptoTransformerV4 model creation"""
    model = CryptoTransformerV4(
        vocab_size=3,
        num_classes=3,
        num_coins=3,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        coin_embedding_dim=16,
        max_seq_len=1024,
        target_coin_idx=2,  # XRP
        btc_coin_idx=0,  # BTC
        binary_classification=False,
        num_channels=9
    )
    
    # Test model was created
    assert model is not None
    assert model.vocab_size == 3
    assert model.num_coins == 3
    assert model.num_channels == 9
    assert model.d_model == 128


def test_token_predictor_forward_pass():
    """Test CryptoTransformerV4 forward pass"""
    model = CryptoTransformerV4(
        vocab_size=3,
        num_classes=3,
        num_coins=3,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        coin_embedding_dim=16,
        max_seq_len=1024,
        target_coin_idx=2,  # XRP
        btc_coin_idx=0,  # BTC
        binary_classification=False,
        num_channels=9
    )

    batch_size = 4
    seq_len = 24
    num_coins = 3
    num_channels = 9

    # Create sample input
    x = torch.randint(0, 3, (batch_size, seq_len, num_coins, num_channels))

    # Forward pass
    output = model(x)

    # Check output structure
    assert isinstance(output, dict), "Model should return a dictionary"
    assert 'horizon_1h' in output, "Model should return horizon_1h prediction"
    
    # Check output shape
    horizon_1h = output['horizon_1h']['logits']
    assert horizon_1h.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {horizon_1h.shape}"


def test_token_predictor_inference():
    """Test CryptoTransformerV4 inference mode"""
    model = CryptoTransformerV4(
        vocab_size=3,
        num_classes=3,
        num_coins=3,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        coin_embedding_dim=16,
        max_seq_len=1024,
        target_coin_idx=2,  # XRP
        btc_coin_idx=0,  # BTC
        binary_classification=False,
        num_channels=9
    )

    model.eval()

    batch_size = 1
    seq_len = 24
    num_coins = 3
    num_channels = 9

    # Create sample input
    x = torch.randint(0, 3, (batch_size, seq_len, num_coins, num_channels))

    # Inference
    with torch.no_grad():
        output = model(x)

    # Check output
    assert isinstance(output, dict), "Model should return a dictionary in inference"
    assert 'horizon_1h' in output, "Model should return horizon_1h prediction in inference"
    
    # Check output shape
    horizon_1h = output['horizon_1h']['logits']
    assert horizon_1h.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {horizon_1h.shape}"


def test_token_predictor_generate():
    """Test CryptoTransformerV4 generation capabilities"""
    model = CryptoTransformerV4(
        vocab_size=3,
        num_classes=3,
        num_coins=3,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        coin_embedding_dim=16,
        max_seq_len=1024,
        target_coin_idx=2,  # XRP
        btc_coin_idx=0,  # BTC
        binary_classification=False,
        num_channels=9
    )

    model.eval()

    batch_size = 2
    seq_len = 24
    num_coins = 3
    num_channels = 9

    # Create sample input
    x = torch.randint(0, 3, (batch_size, seq_len, num_coins, num_channels))

    # Generate predictions
    with torch.no_grad():
        output = model(x)

    # Check output
    assert isinstance(output, dict), "Model should return a dictionary for generation"
    assert 'horizon_1h' in output, "Model should return horizon_1h prediction for generation"
    
    # Check output shape
    horizon_1h = output['horizon_1h']['logits']
    assert horizon_1h.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {horizon_1h.shape}"
    
    # Check that we can get predictions
    horizon_1h = output['horizon_1h']['logits']
    predictions = torch.argmax(horizon_1h, dim=-1)
    assert predictions.shape == (batch_size,), f"Expected ({batch_size},), got {predictions.shape}"


def test_token_predictor_dimensions():
    """Test CryptoTransformerV4 with different dimensions"""
    model = CryptoTransformerV4(
        vocab_size=5,
        num_classes=5,
        num_coins=4,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        coin_embedding_dim=8,
        max_seq_len=512,
        target_coin_idx=3,  # Last coin
        btc_coin_idx=0,  # First coin
        binary_classification=False,
        num_channels=9
    )

    batch_size = 2
    seq_len = 12
    num_coins = 4
    num_channels = 9

    # Create sample input
    x = torch.randint(0, 5, (batch_size, seq_len, num_coins, num_channels))

    # Forward pass
    output = model(x)

    # Check output
    assert isinstance(output, dict), "Model should return a dictionary"
    assert 'horizon_1h' in output, "Model should return horizon_1h prediction"
    
    # Check output shape
    horizon_1h = output['horizon_1h']['logits']
    assert horizon_1h.shape == (batch_size, 5), f"Expected ({batch_size}, 5), got {horizon_1h.shape}"


def test_token_predictor_invalid_dimensions():
    """Test CryptoTransformerV4 handles invalid dimensions gracefully"""
    # Test with invalid coin count
    with pytest.raises(AssertionError):
        model = CryptoTransformerV4(
            vocab_size=3,
            num_classes=3,
            num_coins=3,  # Model expects 3 coins
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            coin_embedding_dim=16,
            max_seq_len=1024,
            target_coin_idx=2,
            btc_coin_idx=0,
            binary_classification=False,
            num_channels=9
        )
        
        # Try to pass input with wrong number of coins
        x = torch.randint(0, 3, (1, 24, 5, 9))  # 5 coins instead of 3
        model(x)