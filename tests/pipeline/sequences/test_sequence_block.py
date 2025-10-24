"""Tests for SequenceBlock - verifies correct shapes (24, num_coins, 2)"""

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import json

from src.pipeline.step_05_sequences.sequence_block import SequenceBlock
from src.pipeline.step_04_tokenize.tokenize_block import TokenizeArtifact
from src.pipeline.schemas import ArtifactMetadata
from src.pipeline.io import ArtifactIO


@pytest.fixture
def config():
    """Test configuration"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP'],
            'target_coin': 'XRP'
        },
        'sequences': {
            'input_length': 24,  # 24 hours input
            'output_length': 8,   # 8 hours output
            'num_channels': 2     # price + volume
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for artifacts"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def tokenized_data(temp_dir):
    """Create sample tokenized data (price + volume channels)"""
    np.random.seed(42)
    num_timesteps = 100
    coins = ['BTC', 'ETH', 'XRP']
    
    # Create tokenized data with both price and volume
    train_data = {'timestamp': pd.date_range('2024-01-01', periods=num_timesteps, freq='H')}
    val_data = {'timestamp': pd.date_range('2024-01-05', periods=50, freq='H')}
    
    for coin in coins:
        # Price tokens (0, 1, 2)
        train_data[f'{coin}_price'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_price'] = np.random.choice([0, 1, 2], size=50)
        
        # Volume tokens (0, 1, 2)
        train_data[f'{coin}_volume'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_volume'] = np.random.choice([0, 1, 2], size=50)
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # Save to parquet
    train_path = Path(temp_dir) / 'train_tokens.parquet'
    val_path = Path(temp_dir) / 'val_tokens.parquet'
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create dummy thresholds
    thresholds = {
        coin: {
            'price': [-0.01, 0.01],
            'volume': [-0.1, 0.1]
        }
        for coin in coins
    }
    thresholds_path = Path(temp_dir) / 'thresholds.json'
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f)
    
    return train_path, val_path, thresholds_path


def test_sequence_shape_is_correct(config, temp_dir, tokenized_data):
    """Test that sequences have shape (N, 24, num_coins, 2)"""
    train_path, val_path, thresholds_path = tokenized_data
    
    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 6),  # 3 coins × 2 channels
        val_shape=(50, 6),
        thresholds_path=thresholds_path,
        token_distribution={0: {'train': 0.33, 'val': 0.30}},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )
    
    # Run sequence creation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = SequenceBlock(config, artifact_io)
    result = block.run(tokenize_artifact)
    
    # Load sequences
    train_X = torch.load(result.train_X_path)
    train_y = torch.load(result.train_y_path)
    val_X = torch.load(result.val_X_path)
    val_y = torch.load(result.val_y_path)
    
    # Check shapes
    num_coins = len(config['data']['coins'])
    input_length = config['sequences']['input_length']
    output_length = config['sequences']['output_length']
    num_channels = config['sequences']['num_channels']
    
    # Input shape: (N, 24, num_coins, 2)
    assert train_X.dim() == 4, f"Expected 4D tensor, got {train_X.dim()}D"
    assert train_X.shape[1] == input_length, f"Expected input_length={input_length}, got {train_X.shape[1]}"
    assert train_X.shape[2] == num_coins, f"Expected {num_coins} coins, got {train_X.shape[2]}"
    assert train_X.shape[3] == num_channels, f"Expected {num_channels} channels, got {train_X.shape[3]}"
    
    # Output shape: (N, 8) for XRP price only
    assert train_y.dim() == 2, f"Expected 2D tensor, got {train_y.dim()}D"
    assert train_y.shape[1] == output_length, f"Expected output_length={output_length}, got {train_y.shape[1]}"
    
    # Same for validation
    assert val_X.shape[1:] == train_X.shape[1:], "Val and train should have same sequence dimensions"
    assert val_y.shape[1] == train_y.shape[1], "Val and train targets should have same length"


def test_sequence_num_samples_correct(config, temp_dir, tokenized_data):
    """Test that number of sequences is computed correctly"""
    train_path, val_path, thresholds_path = tokenized_data
    
    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 6),
        val_shape=(50, 6),
        thresholds_path=thresholds_path,
        token_distribution={0: {'train': 0.33, 'val': 0.30}},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )
    
    # Run sequence creation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = SequenceBlock(config, artifact_io)
    result = block.run(tokenize_artifact)
    
    # Number of sequences = len(data) - input_length - output_length + 1
    # Train: 100 - 24 - 8 + 1 = 69
    # Val: 50 - 24 - 8 + 1 = 19
    expected_train_samples = 100 - 24 - 8 + 1
    expected_val_samples = 50 - 24 - 8 + 1
    
    assert result.train_num_samples == expected_train_samples, \
        f"Expected {expected_train_samples} train samples, got {result.train_num_samples}"
    assert result.val_num_samples == expected_val_samples, \
        f"Expected {expected_val_samples} val samples, got {result.val_num_samples}"


def test_sequence_target_is_xrp_price_only(config, temp_dir, tokenized_data):
    """Test that target is XRP price tokens only (not volume)"""
    train_path, val_path, thresholds_path = tokenized_data
    
    # Load tokenized data to get XRP price tokens
    train_tokens = pd.read_parquet(train_path)
    
    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 6),
        val_shape=(50, 6),
        thresholds_path=thresholds_path,
        token_distribution={0: {'train': 0.33, 'val': 0.30}},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )
    
    # Run sequence creation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = SequenceBlock(config, artifact_io)
    result = block.run(tokenize_artifact)
    
    # Check that target coin is XRP
    assert result.target_coin == 'XRP'
    
    # Load targets
    train_y = torch.load(result.train_y_path)
    
    # Check that targets are in valid range [0, 1, 2]
    assert train_y.min() >= 0 and train_y.max() <= 2
    
    # Check that targets match XRP_price column (not XRP_volume)
    # We can verify by checking first sequence
    first_target = train_y[0].numpy()  # 8 hours of targets
    
    # Get corresponding XRP price tokens from original data
    # First sequence starts at index 24, targets are indices 24:32
    expected_target = train_tokens['XRP_price'].iloc[24:32].values
    
    np.testing.assert_array_equal(first_target, expected_target,
                                   "Targets should match XRP price tokens")


def test_sequence_channels_separated(config, temp_dir, tokenized_data):
    """Test that price and volume channels are correctly separated"""
    train_path, val_path, thresholds_path = tokenized_data
    
    # Load tokenized data
    train_tokens = pd.read_parquet(train_path)
    
    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 6),
        val_shape=(50, 6),
        thresholds_path=thresholds_path,
        token_distribution={0: {'train': 0.33, 'val': 0.30}},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )
    
    # Run sequence creation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = SequenceBlock(config, artifact_io)
    result = block.run(tokenize_artifact)
    
    # Load sequences
    train_X = torch.load(result.train_X_path)
    
    # Check first sequence
    first_seq = train_X[0].numpy()  # (24, 3, 2)
    
    # Verify that channel 0 is price and channel 1 is volume
    # by comparing to original tokenized data
    for t in range(24):
        for coin_idx, coin in enumerate(config['data']['coins']):
            # Channel 0 should be price
            expected_price = train_tokens[f'{coin}_price'].iloc[t]
            actual_price = first_seq[t, coin_idx, 0]
            assert expected_price == actual_price, \
                f"Price mismatch at t={t}, coin={coin}: expected {expected_price}, got {actual_price}"
            
            # Channel 1 should be volume
            expected_volume = train_tokens[f'{coin}_volume'].iloc[t]
            actual_volume = first_seq[t, coin_idx, 1]
            assert expected_volume == actual_volume, \
                f"Volume mismatch at t={t}, coin={coin}: expected {expected_volume}, got {actual_volume}"


def test_sequence_metadata_correct(config, temp_dir, tokenized_data):
    """Test that sequence artifact metadata is correct"""
    train_path, val_path, thresholds_path = tokenized_data
    
    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 6),
        val_shape=(50, 6),
        thresholds_path=thresholds_path,
        token_distribution={0: {'train': 0.33, 'val': 0.30}},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )
    
    # Run sequence creation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = SequenceBlock(config, artifact_io)
    result = block.run(tokenize_artifact)
    
    # Check metadata
    assert result.input_length == 24
    assert result.output_length == 8
    assert result.num_coins == 3
    assert result.target_coin == 'XRP'




