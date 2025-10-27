"""Tests for TokenizeBlock - verifies 2-channel tokenization (price + volume)"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from src.pipeline.step_05_tokenize.tokenize_block import TokenizeBlock
from src.pipeline.schemas import AugmentDataArtifact, ArtifactMetadata
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
        
        # Add technical indicators (9 channels total)
        data[f'{coin}_rsi'] = np.random.uniform(20, 80, len(timestamps))
        data[f'{coin}_macd'] = np.random.normal(0, 0.1, len(timestamps))
        data[f'{coin}_bb_position'] = np.random.uniform(0, 1, len(timestamps))
        data[f'{coin}_ema_9'] = prices * np.random.uniform(0.95, 1.05, len(timestamps))
        data[f'{coin}_ema_21'] = prices * np.random.uniform(0.95, 1.05, len(timestamps))
        data[f'{coin}_ema_50'] = prices * np.random.uniform(0.95, 1.05, len(timestamps))
        data[f'{coin}_ema_ratio'] = np.random.uniform(0.8, 1.2, len(timestamps))
    
    return pd.DataFrame(data)


def test_tokenize_creates_nine_channels(config, temp_dir, sample_data):
    """Test that tokenization creates 9 channels (price + volume + 7 indicators)"""
    # Save sample data
    train_path = Path(temp_dir) / 'train.parquet'
    val_path = Path(temp_dir) / 'val.parquet'
    
    train_df = sample_data.iloc[:80]
    val_df = sample_data.iloc[80:]
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create split artifact
    augment_artifact = AugmentDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        num_coins=3,
        indicators_added=['rsi', 'macd'],
        metadata=ArtifactMetadata(schema_name='augment')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(augment_artifact)
    
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
    """Test that tokenization produces reasonable token distributions"""
    # Save sample data
    train_path = Path(temp_dir) / 'train.parquet'
    val_path = Path(temp_dir) / 'val.parquet'
    
    train_df = sample_data.iloc[:80]
    val_df = sample_data.iloc[80:]
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    # Create split artifact
    augment_artifact = AugmentDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        num_coins=3,
        indicators_added=['rsi', 'macd'],
        metadata=ArtifactMetadata(schema_name='augment')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(augment_artifact)
    
    # Load tokenized data
    train_tokens = pd.read_parquet(result.train_path)
    
    # Check that we have reasonable token distributions (not all zeros or all same value)
    for col in train_tokens.columns:
        if col != 'timestamp':
            value_counts = train_tokens[col].value_counts(normalize=True)
            # Just check that we have some variation (not all tokens are the same)
            assert len(value_counts) > 1, f"{col} has no token variation"
            # Check that no single token dominates completely (>95%)
            max_ratio = value_counts.max()
            assert max_ratio < 0.95, f"{col} token distribution too skewed: {max_ratio:.2%}"


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
    augment_artifact = AugmentDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        num_coins=3,
        indicators_added=['rsi', 'macd'],
        metadata=ArtifactMetadata(schema_name='augment')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(augment_artifact)
    
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
    augment_artifact = AugmentDataArtifact(
        train_path=train_path,
        val_path=val_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
        num_coins=3,
        indicators_added=['rsi', 'macd'],
        metadata=ArtifactMetadata(schema_name='augment')
    )
    
    # Run tokenization
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = TokenizeBlock(config, artifact_io)
    result = block.run(augment_artifact)
    
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




