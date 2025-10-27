import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.pipeline.step_04_augment.augment_block import AugmentBlock
from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import SplitDataArtifact, ArtifactMetadata


@pytest.fixture
def config():
    """Test configuration"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP'],
            'target_coin': 'XRP'
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_split_data():
    """Create sample split data for testing"""
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
    
    data = {
        'timestamp': timestamps,
    }
    
    for coin in ['BTC', 'ETH', 'XRP']:
        # Price data
        base_price = np.random.uniform(100, 1000)
        returns = np.random.normal(0, 0.02, len(timestamps))
        prices = base_price * np.exp(np.cumsum(returns))
        data[f'{coin}_close'] = prices
        
        # Volume data
        base_volume = np.random.uniform(1000, 10000)
        volume_changes = np.random.normal(0, 0.1, len(timestamps))
        volumes = base_volume * np.exp(np.cumsum(volume_changes))
        data[f'{coin}_volume'] = volumes
        
        # Add other OHLC columns
        data[f'{coin}_open'] = prices * 0.99
        data[f'{coin}_high'] = prices * 1.01
        data[f'{coin}_low'] = prices * 0.98
    
    return pd.DataFrame(data)


def test_augment_block_adds_technical_indicators(config, temp_dir, sample_split_data):
    """Test that AugmentBlock adds technical indicators to split data"""
    # Split data into train/val
    train_df = sample_split_data.iloc[:80]
    val_df = sample_split_data.iloc[80:]
    
    # Save split data
    train_path = temp_dir / 'train.parquet'
    val_path = temp_dir / 'val.parquet'
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
    
    # Run augmentation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = AugmentBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # Check artifact properties
    assert result.train_samples == len(train_df)
    assert result.val_samples == len(val_df)
    assert result.num_coins == 3
    assert len(result.indicators_added) > 0
    
    # Check that files exist
    assert result.train_path.exists()
    assert result.val_path.exists()
    
    # Load and verify augmented data
    train_augmented = pd.read_parquet(result.train_path)
    val_augmented = pd.read_parquet(result.val_path)
    
    # Check that technical indicators were added
    expected_indicators = ['rsi', 'macd', 'bb_position', 'ema_9', 'ema_21', 'ema_50', 'ema_ratio']
    for coin in ['BTC', 'ETH', 'XRP']:
        for indicator in expected_indicators:
            col_name = f'{coin}_{indicator}'
            assert col_name in train_augmented.columns
            assert col_name in val_augmented.columns
    
    # Check that original columns are preserved
    original_cols = set(train_df.columns)
    augmented_cols = set(train_augmented.columns)
    assert original_cols.issubset(augmented_cols)


def test_augment_block_preserves_data_integrity(config, temp_dir, sample_split_data):
    """Test that AugmentBlock preserves data integrity"""
    # Split data into train/val
    train_df = sample_split_data.iloc[:80]
    val_df = sample_split_data.iloc[80:]
    
    # Save split data
    train_path = temp_dir / 'train.parquet'
    val_path = temp_dir / 'val.parquet'
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
    
    # Run augmentation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = AugmentBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # Load augmented data
    train_augmented = pd.read_parquet(result.train_path)
    val_augmented = pd.read_parquet(result.val_path)
    
    # Check that original data is preserved
    for col in train_df.columns:
        if col != 'timestamp':  # Skip timestamp comparison due to potential dtype differences
            np.testing.assert_array_almost_equal(
                train_df[col].values, 
                train_augmented[col].values, 
                decimal=10
            )
    
    # Check that timestamps are preserved
    assert train_df['timestamp'].equals(train_augmented['timestamp'])
    assert val_df['timestamp'].equals(val_augmented['timestamp'])


def test_augment_block_handles_nan_values(config, temp_dir):
    """Test that AugmentBlock handles NaN values in input data"""
    # Create data with some NaN values
    timestamps = pd.date_range('2024-01-01', periods=50, freq='h')
    
    data = {
        'timestamp': timestamps,
        'BTC_close': np.random.uniform(100, 1000, 50),
        'BTC_volume': np.random.uniform(1000, 10000, 50),
        'BTC_open': np.random.uniform(100, 1000, 50),
        'BTC_high': np.random.uniform(100, 1000, 50),
        'BTC_low': np.random.uniform(100, 1000, 50),
        'ETH_close': np.random.uniform(200, 2000, 50),
        'ETH_volume': np.random.uniform(2000, 20000, 50),
        'ETH_open': np.random.uniform(200, 2000, 50),
        'ETH_high': np.random.uniform(200, 2000, 50),
        'ETH_low': np.random.uniform(200, 2000, 50),
        'XRP_close': np.random.uniform(50, 500, 50),
        'XRP_volume': np.random.uniform(500, 5000, 50),
        'XRP_open': np.random.uniform(50, 500, 50),
        'XRP_high': np.random.uniform(50, 500, 50),
        'XRP_low': np.random.uniform(50, 500, 50)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some NaN values
    df.loc[10:15, 'BTC_close'] = np.nan
    df.loc[20:25, 'ETH_volume'] = np.nan
    
    # Split data
    train_df = df.iloc[:30]
    val_df = df.iloc[30:]
    
    # Save split data
    train_path = temp_dir / 'train.parquet'
    val_path = temp_dir / 'val.parquet'
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
    
    # Run augmentation - should handle NaN values gracefully
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = AugmentBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # Should complete without errors
    assert result.train_samples == len(train_df)
    assert result.val_samples == len(val_df)


def test_augment_block_metadata_correct(config, temp_dir, sample_split_data):
    """Test that AugmentBlock creates correct metadata"""
    # Split data into train/val
    train_df = sample_split_data.iloc[:80]
    val_df = sample_split_data.iloc[80:]
    
    # Save split data
    train_path = temp_dir / 'train.parquet'
    val_path = temp_dir / 'val.parquet'
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
    
    # Run augmentation
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = AugmentBlock(config, artifact_io)
    result = block.run(split_artifact)
    
    # Check metadata
    assert result.metadata.schema_name == 'augment'
    assert result.metadata.created_at is not None
    assert 'train_split' in result.metadata.upstream_inputs
    
    # Check indicators list
    expected_indicators = ['rsi', 'macd', 'bb_position', 'ema_9', 'ema_21', 'ema_50', 'ema_ratio']
    assert all(indicator in result.indicators_added for indicator in expected_indicators)
