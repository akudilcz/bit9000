"""Tests for SequenceBlock - verifies correct shapes (24, num_coins, 2)"""

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import json

from src.pipeline.step_06_sequences.sequence_block import SequenceBlock
from src.pipeline.step_05_tokenize.tokenize_block import TokenizeArtifact
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
            'num_channels': 9     # price + volume + 7 technical indicators
        },
        'model': {
            'num_classes': 3  # For testing
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
        
        # Technical indicator tokens (0, 1, 2)
        train_data[f'{coin}_rsi'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_rsi'] = np.random.choice([0, 1, 2], size=50)
        
        train_data[f'{coin}_macd'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_macd'] = np.random.choice([0, 1, 2], size=50)
        
        train_data[f'{coin}_bb_position'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_bb_position'] = np.random.choice([0, 1, 2], size=50)
        
        train_data[f'{coin}_ema_9'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_ema_9'] = np.random.choice([0, 1, 2], size=50)
        
        train_data[f'{coin}_ema_21'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_ema_21'] = np.random.choice([0, 1, 2], size=50)
        
        train_data[f'{coin}_ema_50'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_ema_50'] = np.random.choice([0, 1, 2], size=50)
        
        train_data[f'{coin}_ema_ratio'] = np.random.choice([0, 1, 2], size=num_timesteps)
        val_data[f'{coin}_ema_ratio'] = np.random.choice([0, 1, 2], size=50)
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # Save to parquet
    train_path = Path(temp_dir) / 'train_tokens.parquet'
    val_path = Path(temp_dir) / 'val_tokens.parquet'
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create dummy thresholds for all 9 channels
    thresholds = {
        coin: {
            'price': [-0.01, 0.01],
            'volume': [-0.1, 0.1],
            'rsi': [30, 70],
            'macd': [-0.1, 0.1],
            'bb_position': [0.2, 0.8],
            'ema_9': [-0.05, 0.05],
            'ema_21': [-0.05, 0.05],
            'ema_50': [-0.05, 0.05],
            'ema_ratio': [0.9, 1.1]
        }
        for coin in coins
    }
    thresholds_path = Path(temp_dir) / 'thresholds.json'
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f)
    
    return train_path, val_path, thresholds_path


def test_sequence_shape_is_correct(config, temp_dir, tokenized_data):
    """Test that sequences have correct shapes for single-horizon prediction"""
    train_path, val_path, thresholds_path = tokenized_data

    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 27),  # 3 coins × 9 channels
        val_shape=(50, 27),
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
    num_channels = config['sequences']['num_channels']

    # Input shape: (N, 24, num_coins, 9)
    assert train_X.dim() == 4, f"Expected 4D tensor, got {train_X.dim()}D"
    assert train_X.shape[1] == input_length, f"Expected input_length={input_length}, got {train_X.shape[1]}"
    assert train_X.shape[2] == num_coins, f"Expected {num_coins} coins, got {train_X.shape[2]}"
    assert train_X.shape[3] == num_channels, f"Expected {num_channels} channels, got {train_X.shape[3]}"

    # Output shape: (N,) for single-horizon 3-class prediction
    assert train_y.dim() == 1, f"Expected 1D tensor, got {train_y.dim()}D"
    assert train_y.shape[0] == train_X.shape[0], f"Batch size mismatch: X={train_X.shape[0]}, y={train_y.shape[0]}"
    
    # Same for validation
    assert val_X.shape[1:] == train_X.shape[1:], "Val and train should have same sequence dimensions"
    assert val_y.shape[0] == val_X.shape[0], "Val y should match val X batch size"


def test_sequence_num_samples_correct(config, temp_dir, tokenized_data):
    """Test that number of sequences is computed correctly for single-horizon prediction"""
    train_path, val_path, thresholds_path = tokenized_data

    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 27),  # 3 coins × 9 channels
        val_shape=(50, 27),
        thresholds_path=thresholds_path,
        token_distribution={0: {'train': 0.33, 'val': 0.30}},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )

    # Run sequence creation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = SequenceBlock(config, artifact_io)
    result = block.run(tokenize_artifact)

    # For single-horizon prediction: num_samples = len(data) - input_length - prediction_horizon + 1
    # prediction_horizon = 1 (1h ahead)
    # Train: 100 - 24 - 1 + 1 = 76
    # Val: 50 - 24 - 1 + 1 = 26
    input_length = config['sequences']['input_length']
    prediction_horizon = 1  # Single horizon prediction
    
    expected_train_samples = 100 - input_length - prediction_horizon + 1
    expected_val_samples = 50 - input_length - prediction_horizon + 1

    assert result.train_num_samples == expected_train_samples, \
        f"Expected {expected_train_samples} train samples, got {result.train_num_samples}"
    assert result.val_num_samples == expected_val_samples, \
        f"Expected {expected_val_samples} val samples, got {result.val_num_samples}"


def test_sequence_target_is_xrp_price_only(config, temp_dir, tokenized_data):
    """Test that target is 3-class directional prediction for XRP price"""
    train_path, val_path, thresholds_path = tokenized_data

    # Create tokenize artifact
    tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(100, 27),  # 3 coins × 9 channels
        val_shape=(50, 27),
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

    # Check that targets are in valid range [0, 1, 2] for 3-class prediction
    assert train_y.min() >= 0 and train_y.max() <= 2, f"Targets should be in range [0,2], got [{train_y.min()}, {train_y.max()}]"
    
    # Check that we have reasonable class distribution (not all same class)
    unique_classes = torch.unique(train_y)
    assert len(unique_classes) > 1, f"Should have multiple classes, got only {unique_classes.tolist()}"
    
    # Check that targets are integers (class labels)
    assert train_y.dtype == torch.int64, f"Targets should be int64, got {train_y.dtype}"


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




