"""Early Train/Val Split Block: Split data immediately after cleaning"""

import pandas as pd
from pathlib import Path

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import CleanDataArtifact, SplitDataArtifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlySplitBlock(PipelineBlock):
    """Split cleaned data into train/val immediately after cleaning
    
    This ensures complete temporal separation throughout all downstream processing.
    All statistics, features, and transformations will be fit on training data only.
    """
    
    def run(self, clean_artifact: CleanDataArtifact = None) -> SplitDataArtifact:
        """
        Split cleaned data temporally into train and validation sets
        
        Args:
            clean_artifact: CleanDataArtifact from clean block (optional, will load from disk if not provided)
            
        Returns:
            SplitDataArtifact with paths to train and val parquet files
        """
        logger.info("Running early train/val split block")
        
        # Load clean artifact if not provided
        if clean_artifact is None:
            clean_artifact_data = self.artifact_io.read_json('artifacts/step_02_clean/clean_data_artifact.json')
            clean_artifact = CleanDataArtifact(**clean_artifact_data)
        
        # Load cleaned data
        df = self.artifact_io.read_dataframe(clean_artifact.path)
        
        # Get validation period from config (default: 6 months = 4320 hours)
        val_hours = self.config.get('training', {}).get('walk_forward', {}).get('val_split_hours', 4320)
        
        # Calculate split index (reserve last val_hours for validation)
        total_samples = len(df)
        split_idx = total_samples - val_hours
        
        if split_idx <= 0:
            raise ValueError(
                f"Not enough data for split. Total samples: {total_samples}, "
                f"val_hours: {val_hours}. Need at least {val_hours + 1} samples."
            )
        
        # Temporal split: everything before split_idx is training, after is validation
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        # Determine timestamp series for logging and metadata
        if 'timestamp' in train_df.columns:
            ts_train = pd.to_datetime(train_df['timestamp'])
        else:
            ts_train = pd.DatetimeIndex(train_df.index)
        if 'timestamp' in val_df.columns:
            ts_val = pd.to_datetime(val_df['timestamp'])
        else:
            ts_val = pd.DatetimeIndex(val_df.index)

        logger.info(f"Split data temporally:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Train samples: {len(train_df)} ({len(train_df)/24:.1f} days)")
        logger.info(f"  Val samples: {len(val_df)} ({len(val_df)/24:.1f} days)")
        train_start = ts_train[0]
        train_end = ts_train[-1]
        val_start = ts_val[0]
        val_end = ts_val[-1]
        logger.info(f"  Train period: {train_start} to {train_end}")
        logger.info(f"  Val period: {val_start} to {val_end}")
        
        # Save as parquet with full metadata preservation
        train_path = self.artifact_io.write_dataframe(
            train_df,
            block_name="step_03_split",
            artifact_name="train_clean"
        )
        val_path = self.artifact_io.write_dataframe(
            val_df,
            block_name="step_03_split",
            artifact_name="val_clean"
        )
        
        # Create artifact
        artifact = SplitDataArtifact(
            train_path=train_path,
            val_path=val_path,
            train_samples=len(train_df),
            val_samples=len(val_df),
            train_start_date=train_start,
            train_end_date=train_end,
            val_start_date=val_start,
            val_end_date=val_end,
            metadata=self.create_metadata(
                upstream_inputs={"clean_data": str(clean_artifact.path)}
            )
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_03_split",
            artifact_name="split_artifact"
        )
        
        logger.info(f"Early split complete: {len(train_df)} train, {len(val_df)} val samples")
        return artifact

