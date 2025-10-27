import pytest
import tempfile
from pathlib import Path
import os

from src.pipeline.step_00_reset.reset_block import ResetBlock
from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import ArtifactMetadata


@pytest.fixture
def config():
    """Test configuration"""
    return {
        'data': {
            'data_dir': 'artifacts'
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def test_reset_block_clears_artifacts(config, temp_dir):
    """Test that ResetBlock clears existing artifacts"""
    # Create artifacts base directory
    artifacts_base = temp_dir / 'artifacts_base'
    artifacts_base.mkdir()
    
    # Create dummy files
    dummy_files = [
        'step_01_download/raw_data.parquet',
        'step_02_clean/clean_data.parquet',
        'step_03_split/train.parquet',
        'step_04_augment/train_augmented.parquet',
        'step_05_tokenize/train_tokens.parquet',
        'step_06_sequences/train_X.pt',
        'step_07_train/model.pt',
        'step_08_evaluate/eval_report.json',
        'step_09_inference/predictions.json'
    ]
    
    for file_path in dummy_files:
        full_path = artifacts_base / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text('dummy content')
    
    # Verify files exist
    for file_path in dummy_files:
        assert (artifacts_base / file_path).exists()
    
    # Run reset
    artifact_io = ArtifactIO(base_dir=str(artifacts_base))
    block = ResetBlock(config, artifact_io)
    result = block.run()
    
    # Verify artifacts directory is cleared (except for reset artifact)
    assert artifacts_base.exists()  # Directory is recreated for reset artifact
    # Check that only the reset artifact remains
    remaining_files = list(artifacts_base.rglob('*'))
    reset_artifact_files = [f for f in remaining_files if 'reset_artifact' in str(f)]
    assert len(reset_artifact_files) > 0, "Reset artifact should be created"
    # All other files should be gone (except step_00_reset directory for reset artifact)
    non_reset_files = [f for f in remaining_files if 'reset_artifact' not in str(f) and 'step_00_reset' not in str(f)]
    assert len(non_reset_files) == 0, f"Only reset artifact should remain, found: {non_reset_files}"


def test_reset_block_creates_metadata(config, temp_dir):
    """Test that ResetBlock creates proper metadata"""
    # Create a subdirectory for artifacts
    artifacts_base = temp_dir / 'artifacts_base'
    artifacts_base.mkdir()
    
    artifact_io = ArtifactIO(base_dir=str(artifacts_base))
    block = ResetBlock(config, artifact_io)
    result = block.run()
    
    # Check metadata
    assert result.metadata.schema_name == 'reset'
    assert result.reset_timestamp is not None
    assert result.artifacts_cleaned is not None
    assert isinstance(result.cleaned_directories, list)


def test_reset_block_handles_missing_artifacts(config, temp_dir):
    """Test that ResetBlock handles case when no artifacts exist"""
    # Create a subdirectory for artifacts
    artifacts_base = temp_dir / 'artifacts_base'
    artifacts_base.mkdir()
    
    artifact_io = ArtifactIO(base_dir=str(artifacts_base))
    block = ResetBlock(config, artifact_io)
    
    # Should not raise any errors
    result = block.run()
    
    # Should still create metadata
    assert result.metadata.schema_name == 'reset'
    assert result.reset_timestamp is not None
    assert result.artifacts_cleaned == True  # Directory was cleaned and recreated


def test_reset_block_preserves_other_directories(config, temp_dir):
    """Test that ResetBlock only clears artifacts directory, not other directories"""
    # Create artifacts base directory and other directories
    artifacts_base = temp_dir / 'artifacts_base'
    other_dir = temp_dir / 'other_data'
    config_dir = temp_dir / 'config'
    
    artifacts_base.mkdir()
    other_dir.mkdir()
    config_dir.mkdir()
    
    # Create files in each directory
    (artifacts_base / 'dummy.parquet').write_text('artifacts content')
    (other_dir / 'important.txt').write_text('important content')
    (config_dir / 'config.yaml').write_text('config content')
    
    # Run reset
    artifact_io = ArtifactIO(base_dir=str(artifacts_base))
    block = ResetBlock(config, artifact_io)
    result = block.run()
    
    # Verify artifacts base directory is cleaned (but recreated for reset artifact)
    assert artifacts_base.exists()  # Directory is recreated for reset artifact
    # Check that only the reset artifact remains
    remaining_files = list(artifacts_base.rglob('*'))
    reset_artifact_files = [f for f in remaining_files if 'reset_artifact' in str(f)]
    assert len(reset_artifact_files) > 0, "Reset artifact should be created"
    # All other files should be gone (except step_00_reset directory for reset artifact)
    non_reset_files = [f for f in remaining_files if 'reset_artifact' not in str(f) and 'step_00_reset' not in str(f)]
    assert len(non_reset_files) == 0, f"Only reset artifact should remain, found: {non_reset_files}"
    # Verify other directories are preserved
    assert other_dir.exists()
    assert config_dir.exists()
    assert (other_dir / 'important.txt').exists()
    assert (config_dir / 'config.yaml').exists()
