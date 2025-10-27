"""Step 4: Augment - Add technical indicators to cleaned data"""

import pandas as pd
from pathlib import Path

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import AugmentDataArtifact, SplitDataArtifact
from src.utils.logger import get_logger
from src.utils.technical_indicators import add_technical_indicators

logger = get_logger(__name__)


class AugmentBlock(PipelineBlock):
    """Add technical indicators to split data"""

    def run(self, split_artifact: SplitDataArtifact) -> AugmentDataArtifact:
        """
        Add technical indicators to train and validation data

        Args:
            split_artifact: SplitDataArtifact from split block

        Returns:
            AugmentDataArtifact
        """
        logger.info("Running augment block - adding technical indicators")

        # Load split data
        train_df = self.artifact_io.read_dataframe(split_artifact.train_path)
        val_df = self.artifact_io.read_dataframe(split_artifact.val_path)

        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Val data shape: {val_df.shape}")

        # Add technical indicators to both train and val
        logger.info("Adding technical indicators to training data...")
        train_augmented = add_technical_indicators(train_df, config=self.config)

        logger.info("Adding technical indicators to validation data...")
        val_augmented = add_technical_indicators(val_df, config=self.config)

        # Verify we added the expected number of channels
        original_cols = len([c for c in train_df.columns if c.endswith('_close')])
        augmented_cols = len([c for c in train_augmented.columns if c.endswith('_close')])

        if original_cols != augmented_cols:
            raise ValueError(f"Coin count changed during augmentation: {original_cols} -> {augmented_cols}")

        # Check indicator columns were added
        expected_indicators = ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'ema_9', 'ema_21', 'ema_50', 'ema_ratio']
        sample_coin = [c.replace('_close', '') for c in train_df.columns if c.endswith('_close')][0]

        added_indicators = 0
        for indicator in expected_indicators:
            col_name = f"{sample_coin}_{indicator}"
            if col_name in train_augmented.columns:
                added_indicators += 1

        logger.info(f"Added {added_indicators}/{len(expected_indicators)} expected indicators per coin")

        # Write augmented data
        train_path = self.artifact_io.write_dataframe(
            train_augmented,
            block_name="step_04_augment",
            artifact_name="train_augmented",
            metadata=self.create_metadata(
                upstream_inputs={"split_data": str(split_artifact.train_path)}
            )
        )

        val_path = self.artifact_io.write_dataframe(
            val_augmented,
            block_name="step_04_augment",
            artifact_name="val_augmented",
            metadata=self.create_metadata(
                upstream_inputs={"split_data": str(split_artifact.val_path)}
            )
        )

        # Create artifact
        artifact = AugmentDataArtifact(
            train_path=train_path,
            val_path=val_path,
            train_samples=len(train_augmented),
            val_samples=len(val_augmented),
            num_coins=augmented_cols,
            indicators_added=expected_indicators,
            metadata=self.create_metadata(
                upstream_inputs={
                    "train_split": str(split_artifact.train_path),
                    "val_split": str(split_artifact.val_path)
                }
            )
        )

        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_04_augment",
            artifact_name="augment_artifact"
        )

        logger.info(f"Augment complete: {len(expected_indicators)} indicators added to {augmented_cols} coins")
        return artifact
