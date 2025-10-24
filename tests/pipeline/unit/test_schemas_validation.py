from datetime import datetime
from pathlib import Path
import pytest

from src.pipeline.schemas import RawDataArtifact, TokenizeArtifact, ArtifactMetadata


def test_raw_data_artifact_requires_all_column_suffixes():
    metadata = ArtifactMetadata(schema_name="download")
    # Missing the required "_low" suffix
    columns = [
        "BTC_open", "BTC_high", "BTC_close", "BTC_volume",
        "ETH_open", "ETH_high", "ETH_close", "ETH_volume",
    ]

    with pytest.raises(ValueError):
        RawDataArtifact(
            path=Path("/tmp/raw.parquet"),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            num_timesteps=10,
            num_coins=2,
            columns=columns,
            freq="H",
            metadata=metadata,
        )


def test_tokenize_artifact_valid():
    """Test TokenizeArtifact creation with valid data - replaces FeaturesArtifact test"""
    metadata = ArtifactMetadata(schema_name="tokenize")
    
    # Valid tokenize artifact
    artifact = TokenizeArtifact(
        train_path=Path("/tmp/train_tokens.parquet"),
        val_path=Path("/tmp/val_tokens.parquet"),
        train_shape=(1000, 10),
        val_shape=(200, 10),
        thresholds_path=Path("/tmp/thresholds.json"),
        token_distribution={0: {"train": 0.33, "val": 0.30}, 1: {"train": 0.34, "val": 0.40}, 2: {"train": 0.33, "val": 0.30}},
        metadata=metadata,
    )
    
    assert artifact.train_shape == (1000, 10)
    assert artifact.val_shape == (200, 10)
    assert len(artifact.token_distribution) == 3



