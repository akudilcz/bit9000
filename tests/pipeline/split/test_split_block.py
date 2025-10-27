import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.pipeline.step_03_split.split_block import EarlySplitBlock
from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import CleanDataArtifact, ArtifactMetadata


@pytest.fixture
def config():
    """Test configuration"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP'],
            'target_coin': 'XRP'
        },
        'split': {
            'train_ratio': 0.8,
            'val_ratio': 0.2
        },
        'training': {
            'walk_forward': {
                'val_split_hours': 2  # Very small validation period for edge case tests
            }
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_clean_data():
    """Create sample clean data for testing"""
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


def test_split_block_creates_train_val_split(config, temp_dir, sample_clean_data):
    """Test that SplitBlock creates proper train/validation split"""
    # Save sample data
    clean_path = temp_dir / 'clean_data.parquet'
    sample_clean_data.to_parquet(clean_path)
    
    # Create clean artifact
    clean_artifact = CleanDataArtifact(
        path=clean_path,
        start_date=sample_clean_data['timestamp'].iloc[0],
        end_date=sample_clean_data['timestamp'].iloc[-1],
        num_timesteps=len(sample_clean_data),
        num_coins=3,
        quality_metrics={'completeness': 1.0, 'consistency': 1.0},
        metadata=ArtifactMetadata(schema_name='clean')
    )
    
    # Run split
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = EarlySplitBlock(config, artifact_io)
    result = block.run(clean_artifact)
    
    # Check artifact properties
    assert result.train_samples + result.val_samples == len(sample_clean_data)
    assert result.train_samples > result.val_samples  # Train should be larger
    
    # Check that files exist
    assert result.train_path.exists()
    assert result.val_path.exists()
    
    # Load and verify split data
    train_df = pd.read_parquet(result.train_path)
    val_df = pd.read_parquet(result.val_path)
    
    assert len(train_df) == result.train_samples
    assert len(val_df) == result.val_samples
    
    # Check temporal ordering (no data leakage)
    assert train_df['timestamp'].max() < val_df['timestamp'].min()


def test_split_block_preserves_data_integrity(config, temp_dir, sample_clean_data):
    """Test that SplitBlock preserves data integrity"""
    # Save sample data
    clean_path = temp_dir / 'clean_data.parquet'
    sample_clean_data.to_parquet(clean_path)
    
    # Create clean artifact
    clean_artifact = CleanDataArtifact(
        path=clean_path,
        start_date=sample_clean_data['timestamp'].iloc[0],
        end_date=sample_clean_data['timestamp'].iloc[-1],
        num_timesteps=len(sample_clean_data),
        num_coins=3,
        quality_metrics={'completeness': 1.0, 'consistency': 1.0},
        metadata=ArtifactMetadata(schema_name='clean')
    )
    
    # Run split
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = EarlySplitBlock(config, artifact_io)
    result = block.run(clean_artifact)
    
    # Load split data
    train_df = pd.read_parquet(result.train_path)
    val_df = pd.read_parquet(result.val_path)
    
    # Combine and sort by timestamp
    combined_df = pd.concat([train_df, val_df]).sort_values('timestamp').reset_index(drop=True)
    
    # Check that all original data is preserved
    assert len(combined_df) == len(sample_clean_data)
    
    # Check that columns are preserved
    expected_cols = set(sample_clean_data.columns)
    actual_cols = set(combined_df.columns)
    assert expected_cols == actual_cols


def test_split_block_handles_edge_cases(config, temp_dir):
    """Test SplitBlock with edge cases"""
    # Test with minimal data
    minimal_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'BTC_close': [100, 101, 102, 103, 104],
        'BTC_volume': [1000, 1100, 1200, 1300, 1400],
        'BTC_open': [99, 100, 101, 102, 103],
        'BTC_high': [101, 102, 103, 104, 105],
        'BTC_low': [98, 99, 100, 101, 102],
        'ETH_close': [200, 201, 202, 203, 204],
        'ETH_volume': [2000, 2100, 2200, 2300, 2400],
        'ETH_open': [199, 200, 201, 202, 203],
        'ETH_high': [201, 202, 203, 204, 205],
        'ETH_low': [198, 199, 200, 201, 202],
        'XRP_close': [50, 51, 52, 53, 54],
        'XRP_volume': [500, 510, 520, 530, 540],
        'XRP_open': [49, 50, 51, 52, 53],
        'XRP_high': [51, 52, 53, 54, 55],
        'XRP_low': [48, 49, 50, 51, 52]
    })
    
    # Save sample data
    clean_path = temp_dir / 'clean_data.parquet'
    minimal_data.to_parquet(clean_path)
    
    # Create clean artifact
    clean_artifact = CleanDataArtifact(
        path=clean_path,
        start_date=minimal_data['timestamp'].iloc[0],
        end_date=minimal_data['timestamp'].iloc[-1],
        num_timesteps=len(minimal_data),
        num_coins=3,
        quality_metrics={'completeness': 1.0, 'consistency': 1.0},
        metadata=ArtifactMetadata(schema_name='clean')
    )
    
    # Run split
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = EarlySplitBlock(config, artifact_io)
    result = block.run(clean_artifact)
    
    # Should still work with minimal data
    assert result.train_samples > 0
    assert result.val_samples > 0
    assert result.train_samples + result.val_samples == len(minimal_data)


def test_split_block_metadata_correct(config, temp_dir, sample_clean_data):
    """Test that SplitBlock creates correct metadata"""
    # Save sample data
    clean_path = temp_dir / 'clean_data.parquet'
    sample_clean_data.to_parquet(clean_path)
    
    # Create clean artifact
    clean_artifact = CleanDataArtifact(
        path=clean_path,
        start_date=sample_clean_data['timestamp'].iloc[0],
        end_date=sample_clean_data['timestamp'].iloc[-1],
        num_timesteps=len(sample_clean_data),
        num_coins=3,
        quality_metrics={'completeness': 1.0, 'consistency': 1.0},
        metadata=ArtifactMetadata(schema_name='clean')
    )
    
    # Run split
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = EarlySplitBlock(config, artifact_io)
    result = block.run(clean_artifact)
    
    # Check metadata
    assert result.metadata.schema_name == 'earlysplit'
    assert result.metadata.created_at is not None
    assert 'clean_data' in result.metadata.upstream_inputs
    
    # Check date ranges
    train_df = pd.read_parquet(result.train_path)
    val_df = pd.read_parquet(result.val_path)
    
    assert result.train_start_date == train_df['timestamp'].iloc[0]
    assert result.train_end_date == train_df['timestamp'].iloc[-1]
    assert result.val_start_date == val_df['timestamp'].iloc[0]
    assert result.val_end_date == val_df['timestamp'].iloc[-1]
