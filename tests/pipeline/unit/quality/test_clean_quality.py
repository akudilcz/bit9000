"""Test data quality after cleaning"""

import pytest
import pandas as pd
import numpy as np

from src.pipeline.io import ArtifactIO
from src.pipeline.step_01_download.download_block import DownloadBlock
from src.pipeline.step_02_clean.clean_block import CleanBlock


def test_clean_removes_nans(test_config, artifact_io, canned_anomalies_data, tmp_path):
    """Test that cleaning removes or fills NaNs"""
    # Save anomalies data
    data_path = tmp_path / "anomalies.parquet"
    canned_anomalies_data.to_parquet(data_path)
    
    # Manually create a raw artifact
    from src.pipeline.schemas import RawDataArtifact, ArtifactMetadata
    raw_artifact = RawDataArtifact(
        path=data_path,
        start_date=canned_anomalies_data.index[0],
        end_date=canned_anomalies_data.index[-1],
        num_timesteps=len(canned_anomalies_data),
        num_coins=2,
        columns=list(canned_anomalies_data.columns),
        freq="H",
        metadata=ArtifactMetadata(schema_name="test")
    )
    
    # Run clean block
    clean_block = CleanBlock(test_config, artifact_io)
    clean_artifact = clean_block.run(raw_artifact)
    
    # Load cleaned data
    cleaned_df = artifact_io.read_dataframe(clean_artifact.path)
    
    # Assert no NaNs remain
    assert cleaned_df.isna().sum().sum() == 0, "Cleaned data should have no NaNs"


def test_clean_removes_duplicates(test_config, artifact_io, canned_anomalies_data, tmp_path):
    """Test that cleaning removes duplicate timestamps"""
    data_path = tmp_path / "anomalies.parquet"
    canned_anomalies_data.to_parquet(data_path)
    
    from src.pipeline.schemas import RawDataArtifact, ArtifactMetadata
    raw_artifact = RawDataArtifact(
        path=data_path,
        start_date=canned_anomalies_data.index[0],
        end_date=canned_anomalies_data.index[-1],
        num_timesteps=len(canned_anomalies_data),
        num_coins=2,
        columns=list(canned_anomalies_data.columns),
        freq="H",
        metadata=ArtifactMetadata(schema_name="test")
    )
    
    clean_block = CleanBlock(test_config, artifact_io)
    clean_artifact = clean_block.run(raw_artifact)
    
    cleaned_df = artifact_io.read_dataframe(clean_artifact.path)
    
    # Assert no duplicate indices
    assert not cleaned_df.index.duplicated().any(), "No duplicate timestamps"


def test_clean_ensures_monotonic_index(test_config, artifact_io, canned_anomalies_data, tmp_path):
    """Test that index is sorted and monotonic"""
    data_path = tmp_path / "anomalies.parquet"
    canned_anomalies_data.to_parquet(data_path)
    
    from src.pipeline.schemas import RawDataArtifact, ArtifactMetadata
    raw_artifact = RawDataArtifact(
        path=data_path,
        start_date=canned_anomalies_data.index[0],
        end_date=canned_anomalies_data.index[-1],
        num_timesteps=len(canned_anomalies_data),
        num_coins=2,
        columns=list(canned_anomalies_data.columns),
        freq="H",
        metadata=ArtifactMetadata(schema_name="test")
    )
    
    clean_block = CleanBlock(test_config, artifact_io)
    clean_artifact = clean_block.run(raw_artifact)
    
    cleaned_df = artifact_io.read_dataframe(clean_artifact.path)
    
    # Assert monotonic increasing
    assert cleaned_df.index.is_monotonic_increasing, "Index should be monotonic"


def test_clean_quality_metrics(test_config, artifact_io, canned_anomalies_data, tmp_path):
    """Test that quality metrics are recorded"""
    data_path = tmp_path / "anomalies.parquet"
    canned_anomalies_data.to_parquet(data_path)
    
    from src.pipeline.schemas import RawDataArtifact, ArtifactMetadata
    raw_artifact = RawDataArtifact(
        path=data_path,
        start_date=canned_anomalies_data.index[0],
        end_date=canned_anomalies_data.index[-1],
        num_timesteps=len(canned_anomalies_data),
        num_coins=2,
        columns=list(canned_anomalies_data.columns),
        freq="H",
        metadata=ArtifactMetadata(schema_name="test")
    )
    
    clean_block = CleanBlock(test_config, artifact_io)
    clean_artifact = clean_block.run(raw_artifact)
    
    # Check quality metrics
    assert 'initial_nans' in clean_artifact.quality_metrics
    assert 'remaining_nans' in clean_artifact.quality_metrics
    assert 'duplicate_rows_removed' in clean_artifact.quality_metrics
    assert clean_artifact.quality_metrics['remaining_nans'] == 0






















