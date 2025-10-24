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
    
    def run(self, clean_artifact: CleanDataArtifact) -> SplitDataArtifact:
        """
        Split cleaned data temporally into train and validation sets
        
        Args:
            clean_artifact: CleanDataArtifact from clean block
            
        Returns:
            SplitDataArtifact with paths to train and val parquet files
        """
        logger.info("Running early train/val split block")
        
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
        
        logger.info(f"Split data temporally:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Train samples: {len(train_df)} ({len(train_df)/24:.1f} days)")
        logger.info(f"  Val samples: {len(val_df)} ({len(val_df)/24:.1f} days)")
        logger.info(f"  Train period: {train_df.index[0]} to {train_df.index[-1]}")
        logger.info(f"  Val period: {val_df.index[0]} to {val_df.index[-1]}")
        
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
            train_start_date=train_df.index[0],
            train_end_date=train_df.index[-1],
            val_start_date=val_df.index[0],
            val_end_date=val_df.index[-1],
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

