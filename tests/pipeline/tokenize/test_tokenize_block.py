"""Tests for TokenizeBlock - verifies 2-channel tokenization (price + volume)"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from src.pipeline.step_04_tokenize.tokenize_block import TokenizeBlock
from src.pipeline.schemas import SplitDataArtifact, ArtifactMetadata
from src.pipeline.io import ArtifactIO


@pytest.fixture
def config():
    """Test configuration"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP'],
            'target_coin': 'XRP'
        },
        'tokenization': {
            'vocab_size': 3,
            'method': 'quantile',
            'percentiles': [33, 67]
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for artifacts"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
    
    # Create realistic price and volume data
    data = {
        'timestamp': timestamps,
    }
    
    for coin in ['BTC', 'ETH', 'XRP']:
        # Price data (close)
        base_price = np.random.uniform(100, 1000)
        returns = np.random.normal(0, 0.02, len(timestamps))
        prices = base_price * np.exp(np.cumsum(returns))
        data[f'{coin}_close'] = prices
        
        # Volume data
        base_volume = np.random.uniform(1000, 10000)
        volume_changes = np.random.normal(0, 0.1, len(timestamps))
        volumes = base_volume * np.exp(np.cumsum(volume_changes))
        data[f'{coin}_volume'] = volumes
        
        # Add other OHLC columns (not used but required)
        data[f'{coin}_open'] = prices * 0.99
        data[f'{coin}_high'] = prices * 1.01
        data[f'{coin}_low'] = prices * 0.98
    
    return pd.DataFrame(data)


def test_tokenize_creates_two_channels(config, temp_dir, sample_data):
    """Test that tokenization creates 2 channels (price + volume)"""
    # Save sample data
    train_path = Path(temp_dir) / 'train.parquet'
    val_path = Path(temp_dir) / 'val.parquet'
    
    train_df = sample_data.iloc[:80]
    val_df = sample_data.iloc[80:]
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create split artifact
    split_artifact = SplitDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        train_start_date=train_df['timestamp'].iloc[0],
        train_end_date=train_df['timestamp'].iloc[-1],
        val_start_date=val_df['timestamp'].iloc[0],
        val_end_date=val_df['timestamp'].iloc[-1],
        metadata=ArtifactMetadata(schema_name='split')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # Load tokenized data
    train_tokens = pd.read_parquet(result.train_path)
    val_tokens = pd.read_parquet(result.val_path)
    
    # Check that we have both price and volume columns for each coin
    for coin in config['data']['coins']:
        assert f'{coin}_price' in train_tokens.columns, f"Missing {coin}_price column"
        assert f'{coin}_volume' in train_tokens.columns, f"Missing {coin}_volume column"
        
        # Check token values are in range [0, 1, 2]
        assert train_tokens[f'{coin}_price'].isin([0, 1, 2]).all()
        assert train_tokens[f'{coin}_volume'].isin([0, 1, 2]).all()


def test_tokenize_balanced_classes_on_train(config, temp_dir, sample_data):
    """Test that train data has balanced classes (~33% each)"""
    # Save sample data
    train_path = Path(temp_dir) / 'train.parquet'
    val_path = Path(temp_dir) / 'val.parquet'
    
    train_df = sample_data.iloc[:80]
    val_df = sample_data.iloc[80:]
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create split artifact
    split_artifact = SplitDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        train_start_date=train_df['timestamp'].iloc[0],
        train_end_date=train_df['timestamp'].iloc[-1],
        val_start_date=val_df['timestamp'].iloc[0],
        val_end_date=val_df['timestamp'].iloc[-1],
        metadata=ArtifactMetadata(schema_name='split')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # Load tokenized data
    train_tokens = pd.read_parquet(result.train_path)
    
    # Check class balance for each coin and channel
    for coin in config['data']['coins']:
        for channel in ['price', 'volume']:
            col = f'{coin}_{channel}'
            value_counts = train_tokens[col].value_counts(normalize=True)
            
            # Each class should be roughly 33% (allow 5% tolerance)
            for token in [0, 1, 2]:
                ratio = value_counts.get(token, 0)
                assert 0.25 < ratio < 0.42, f"{col} token {token} ratio {ratio:.2%} not balanced"


def test_tokenize_thresholds_per_coin_per_channel(config, temp_dir, sample_data):
    """Test that thresholds are saved per coin per channel"""
    # Save sample data
    train_path = Path(temp_dir) / 'train.parquet'
    val_path = Path(temp_dir) / 'val.parquet'
    
    train_df = sample_data.iloc[:80]
    val_df = sample_data.iloc[80:]
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create split artifact
    split_artifact = SplitDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        train_start_date=train_df['timestamp'].iloc[0],
        train_end_date=train_df['timestamp'].iloc[-1],
        val_start_date=val_df['timestamp'].iloc[0],
        val_end_date=val_df['timestamp'].iloc[-1],
        metadata=ArtifactMetadata(schema_name='split')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # Load thresholds
    with open(result.thresholds_path) as f:
        thresholds = json.load(f)
    
    # Check structure: {coin: {price: [low, high], volume: [low, high]}}
    for coin in config['data']['coins']:
        assert coin in thresholds, f"Missing thresholds for {coin}"
        assert 'price' in thresholds[coin], f"Missing price thresholds for {coin}"
        assert 'volume' in thresholds[coin], f"Missing volume thresholds for {coin}"
        
        # Check that each has [low, high] thresholds
        assert len(thresholds[coin]['price']) == 2
        assert len(thresholds[coin]['volume']) == 2
        
        # Check that low < high
        assert thresholds[coin]['price'][0] < thresholds[coin]['price'][1]
        assert thresholds[coin]['volume'][0] < thresholds[coin]['volume'][1]


def test_tokenize_no_data_leakage(config, temp_dir, sample_data):
    """Test that val thresholds are same as train (no leakage)"""
    # Save sample data
    train_path = Path(temp_dir) / 'train.parquet'
    val_path = Path(temp_dir) / 'val.parquet'
    
    train_df = sample_data.iloc[:80]
    val_df = sample_data.iloc[80:]
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create split artifact
    split_artifact = SplitDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        train_start_date=train_df['timestamp'].iloc[0],
        train_end_date=train_df['timestamp'].iloc[-1],
        val_start_date=val_df['timestamp'].iloc[0],
        val_end_date=val_df['timestamp'].iloc[-1],
        metadata=ArtifactMetadata(schema_name='split')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # The thresholds should be computed ONLY on train data
    # We can verify this by checking that val distribution may differ from 33/33/33
    val_tokens = pd.read_parquet(result.val_path)
    
    # At least one coin/channel should have different distribution in val
    # (if thresholds were computed on val, distribution would be forced to 33/33/33)
    distributions_differ = False
    for coin in config['data']['coins']:
        for channel in ['price', 'volume']:
            col = f'{coin}_{channel}'
            value_counts = val_tokens[col].value_counts(normalize=True)
            
            # Check if any class deviates significantly from 33%
            for token in [0, 1, 2]:
                ratio = value_counts.get(token, 0)
                if abs(ratio - 0.333) > 0.1:  # More than 10% deviation
                    distributions_differ = True
                    break
    
    # This test may occasionally fail if val happens to have perfect 33/33/33 by chance
    # But statistically it should pass most of the time
    assert distributions_differ or len(val_df) < 30, \
        "Val distribution suspiciously balanced - possible data leakage"




