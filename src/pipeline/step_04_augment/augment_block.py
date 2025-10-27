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
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features (hour of day, day of week) for each coin
        
        Args:
            df: DataFrame with timestamp index
            
        Returns:
            DataFrame with time features added
        """
        # Extract time features from timestamp index
        hour_of_day = df.index.hour  # 0-23
        day_of_week = df.index.dayofweek  # 0=Monday, 6=Sunday
        
        # Normalize to 0-1 range for better tokenization
        hour_normalized = hour_of_day / 23.0  # 0.0 to 1.0
        day_normalized = day_of_week / 6.0  # 0.0 to 1.0
        
        # Get list of coins
        coins = self.config['data']['coins']
        
        # Add time features for each coin (they're the same across all coins, but we keep them per-coin for consistency)
        for coin in coins:
            df[f'{coin}_hour'] = hour_normalized
            df[f'{coin}_day_of_week'] = day_normalized
        
        logger.info(f"  Added hour (0-23) and day_of_week (0-6) features for {len(coins)} coins")
        
        return df

    def run(self, split_artifact: SplitDataArtifact = None) -> AugmentDataArtifact:
        """
        Add technical indicators to train and validation data

        Args:
            split_artifact: SplitDataArtifact from split block (optional, will load from disk if not provided)

        Returns:
            AugmentDataArtifact
        """
        logger.info("Running augment block - adding technical indicators")

        # Load split artifact if not provided
        if split_artifact is None:
            split_artifact_data = self.artifact_io.read_json('artifacts/step_03_split/split_artifact.json')
            split_artifact = SplitDataArtifact(**split_artifact_data)

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
        
        # Add time features (hour of day, day of week) for each coin
        logger.info("Adding time features (hour, day_of_week) to training data...")
        train_augmented = self._add_time_features(train_augmented)
        
        logger.info("Adding time features (hour, day_of_week) to validation data...")
        val_augmented = self._add_time_features(val_augmented)

        # Verify we added the expected number of channels
        original_cols = len([c for c in train_df.columns if c.endswith('_close')])
        augmented_cols = len([c for c in train_augmented.columns if c.endswith('_close')])

        if original_cols != augmented_cols:
            raise ValueError(f"Coin count changed during augmentation: {original_cols} -> {augmented_cols}")

        # Check indicator columns were added
        expected_indicators = ['rsi', 'macd', 'bb_position', 'ema_9', 'ema_21', 'ema_50', 'ema_ratio',
                               'stochastic', 'williams_r', 'atr', 'adx', 'obv', 'volume_roc', 'vwap',
                               'price_momentum', 'support_resistance', 'volatility_regime', 'hour', 'day_of_week']
        sample_coin = [c.replace('_close', '') for c in train_df.columns if c.endswith('_close')][0]

        added_indicators = 0
        for indicator in expected_indicators:
            col_name = f"{sample_coin}_{indicator}"
            if col_name in train_augmented.columns:
                added_indicators += 1

        logger.info(f"Added {added_indicators}/{len(expected_indicators)} expected features per coin (17 indicators + 2 time features)")

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
