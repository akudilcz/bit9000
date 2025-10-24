"""Step 4: Tokenize - Convert prices and volume to balanced token sequences

Philosophy: Simple price and volume movements → discrete tokens (down/steady/up)
- 2 channels: price (log returns) and volume (log changes)
- Quantile thresholds ensure ~33% distribution per coin per channel
- No engineered features, just raw movements
- Fit thresholds on train data only (no leakage)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import SplitDataArtifact, ArtifactMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenizeArtifact:
    """Artifact for tokenized data with 2 channels"""
    def __init__(self, train_path: Path, val_path: Path, 
                 train_shape: Tuple[int, int], val_shape: Tuple[int, int],
                 thresholds_path: Path, token_distribution: Dict[int, Dict[str, float]],
                 metadata: ArtifactMetadata):
        self.train_path = train_path
        self.val_path = val_path
        self.train_shape = train_shape
        self.val_shape = val_shape
        self.thresholds_path = thresholds_path
        self.token_distribution = token_distribution
        self.metadata = metadata
    
    def model_dump(self, mode='json'):
        return {
            'train_path': str(self.train_path),
            'val_path': str(self.val_path),
            'train_shape': list(self.train_shape),
            'val_shape': list(self.val_shape),
            'thresholds_path': str(self.thresholds_path),
            'token_distribution': self.token_distribution,
            'metadata': self.metadata.model_dump(mode=mode)
        }


class TokenizeBlock(PipelineBlock):
    """Convert prices and volume to balanced 3-token sequences using quantile thresholds"""
    
    def run(self, split_artifact: SplitDataArtifact):
        """
        Tokenize prices and volume using quantile-based thresholds
        
        Process:
        1. FIT: Calculate quantile thresholds on TRAINING data only (per coin, per channel)
        2. TRANSFORM: Apply thresholds to both train and val data
        3. Each coin × channel gets independent thresholds (adaptive to volatility)
        4. Output: 0=down, 1=steady, 2=up (balanced ~33/33/33 distribution)
        
        Channels:
        - Price channel: log returns of close prices
        - Volume channel: log changes in volume
        
        Args:
            split_artifact: SplitDataArtifact from step_03_split
            
        Returns:
            TokenizeArtifact
        """
        logger.info("="*70)
        logger.info("STEP 4: TOKENIZE - Converting prices and volume to tokens")
        logger.info("="*70)
        
        # Load train and val data
        logger.info("\n[1/4] Loading split data...")
        train_df = self.artifact_io.read_dataframe(split_artifact.train_path)
        val_df = self.artifact_io.read_dataframe(split_artifact.val_path)
        
        logger.info(f"  Train: {train_df.shape}")
        logger.info(f"  Val: {val_df.shape}")
        
        # Get coin list from config
        coins = self.config['data']['coins']
        logger.info(f"  Coins: {coins}")
        logger.info(f"  Channels: price + volume")
        
        # FIT: Calculate quantile thresholds on training data
        logger.info("\n[2/4] FIT: Calculating quantile thresholds on TRAINING data...")
        thresholds = self._fit_thresholds(train_df, coins)
        
        # Log thresholds
        for coin in coins:
            if coin in thresholds:
                price_low, price_high = thresholds[coin]['price']
                vol_low, vol_high = thresholds[coin]['volume']
                logger.info(f"  {coin}:")
                logger.info(f"    Price:  tau_low={price_low:.6f}, tau_high={price_high:.6f}")
                logger.info(f"    Volume: tau_low={vol_low:.6f}, tau_high={vol_high:.6f}")
        
        # TRANSFORM: Apply thresholds to train and val
        logger.info("\n[3/4] TRANSFORM: Tokenizing train and val data...")
        train_tokens = self._transform_to_tokens(train_df, coins, thresholds)
        val_tokens = self._transform_to_tokens(val_df, coins, thresholds)
        
        logger.info(f"  Train tokens: {train_tokens.shape} (columns: {list(train_tokens.columns)})")
        logger.info(f"  Val tokens: {val_tokens.shape}")
        
        # Verify balanced distribution on training data
        token_distribution = self._compute_distribution(train_tokens, val_tokens)
        logger.info("\n  Token distribution:")
        for token in [0, 1, 2]:
            token_name = ['down', 'steady', 'up'][token]
            train_pct = token_distribution[token]['train'] * 100
            val_pct = token_distribution[token]['val'] * 100
            logger.info(f"    {token} ({token_name:6}): Train={train_pct:5.1f}%, Val={val_pct:5.1f}%")
        
        # Save artifacts
        logger.info("\n[4/4] Saving artifacts...")
        block_dir = self.artifact_io.get_block_dir("step_04_tokenize", clean=True)
        
        # Save token dataframes
        train_path = block_dir / "train_tokens.parquet"
        val_path = block_dir / "val_tokens.parquet"
        
        train_tokens.to_parquet(train_path, engine='pyarrow', compression='snappy', index=True)
        val_tokens.to_parquet(val_path, engine='pyarrow', compression='snappy', index=True)
        
        logger.info(f"  Saved train tokens: {train_path}")
        logger.info(f"  Saved val tokens: {val_path}")
        
        # Save thresholds (for inference)
        thresholds_path = block_dir / "thresholds.json"
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        logger.info(f"  Saved thresholds: {thresholds_path}")
        
        # Create artifact
        num_channels = 2  # price + volume
        artifact = TokenizeArtifact(
            train_path=train_path,
            val_path=val_path,
            train_shape=(train_tokens.shape[0], len(coins) * num_channels),
            val_shape=(val_tokens.shape[0], len(coins) * num_channels),
            thresholds_path=thresholds_path,
            token_distribution=token_distribution,
            metadata=self.create_metadata(
                upstream_inputs={
                    "train_clean": str(split_artifact.train_path),
                    "val_clean": str(split_artifact.val_path)
                }
            )
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_04_tokenize",
            artifact_name="tokenize_artifact"
        )
        
        logger.info("\n" + "="*70)
        logger.info("TOKENIZATION COMPLETE")
        logger.info(f"  Train: {train_tokens.shape[0]:,} timesteps × {len(coins)} coins × 2 channels")
        logger.info(f"  Val: {val_tokens.shape[0]:,} timesteps × {len(coins)} coins × 2 channels")
        logger.info("="*70 + "\n")
        
        return artifact
    
    def _fit_thresholds(self, train_df: pd.DataFrame, coins: list) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        FIT: Calculate quantile thresholds on training data for price and volume
        
        For each coin:
        - Price channel: Compute hourly log returns: r = log(close[t] / close[t-1])
        - Volume channel: Compute hourly log changes: v = log(volume[t] / volume[t-1])
        - For each channel, find 33rd and 67th percentiles
        
        Args:
            train_df: Training data with COIN_close and COIN_volume columns
            coins: List of coin symbols
            
        Returns:
            Dictionary {coin: {price: (tau_low, tau_high), volume: (tau_low, tau_high)}}
        """
        thresholds = {}
        
        for coin in coins:
            close_col = f"{coin}_close"
            volume_col = f"{coin}_volume"
            
            if close_col not in train_df.columns or volume_col not in train_df.columns:
                logger.warning(f"  Missing columns for {coin}, skipping")
                continue
            
            # Compute log returns for price
            prices = train_df[close_col]
            price_returns = np.log(prices / prices.shift(1))
            price_returns = price_returns.dropna()
            
            # Compute log changes for volume
            volumes = train_df[volume_col]
            volume_changes = np.log(volumes / volumes.shift(1))
            volume_changes = volume_changes.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(price_returns) == 0 or len(volume_changes) == 0:
                logger.warning(f"  No valid data for {coin}, skipping")
                continue
            
            # Calculate quantile thresholds
            price_tau_low = price_returns.quantile(0.33)
            price_tau_high = price_returns.quantile(0.67)
            
            volume_tau_low = volume_changes.quantile(0.33)
            volume_tau_high = volume_changes.quantile(0.67)
            
            thresholds[coin] = {
                'price': (float(price_tau_low), float(price_tau_high)),
                'volume': (float(volume_tau_low), float(volume_tau_high))
            }
        
        return thresholds
    
    def _transform_to_tokens(self, df: pd.DataFrame, coins: list, 
                            thresholds: Dict[str, Dict[str, Tuple[float, float]]]) -> pd.DataFrame:
        """
        TRANSFORM: Convert prices and volume to tokens using fitted thresholds
        
        Token mapping (applied independently to each channel):
        - 0 (down): value ≤ tau_low
        - 1 (steady): tau_low < value ≤ tau_high
        - 2 (up): value > tau_high
        
        Args:
            df: DataFrame with COIN_close and COIN_volume columns
            coins: List of coin symbols
            thresholds: Fitted thresholds {coin: {price: (...), volume: (...)}}
            
        Returns:
            DataFrame with columns: COIN_price, COIN_volume (token values 0/1/2)
        """
        tokens_dict = {}
        
        for coin in coins:
            close_col = f"{coin}_close"
            volume_col = f"{coin}_volume"
            
            if close_col not in df.columns or volume_col not in df.columns:
                logger.warning(f"  Missing columns for {coin}, skipping")
                continue
            
            if coin not in thresholds:
                logger.warning(f"  No thresholds for {coin}, skipping")
                continue
            
            # PRICE CHANNEL
            prices = df[close_col]
            price_returns = np.log(prices / prices.shift(1))
            
            price_tau_low, price_tau_high = thresholds[coin]['price']
            price_tokens = np.full(len(price_returns), np.nan)
            price_tokens[price_returns <= price_tau_low] = 0  # down
            price_tokens[(price_returns > price_tau_low) & (price_returns <= price_tau_high)] = 1  # steady
            price_tokens[price_returns > price_tau_high] = 2  # up
            
            tokens_dict[f"{coin}_price"] = price_tokens
            
            # VOLUME CHANNEL
            volumes = df[volume_col]
            volume_changes = np.log(volumes / volumes.shift(1))
            volume_changes = volume_changes.replace([np.inf, -np.inf], np.nan)
            
            volume_tau_low, volume_tau_high = thresholds[coin]['volume']
            volume_tokens = np.full(len(volume_changes), np.nan)
            volume_tokens[volume_changes <= volume_tau_low] = 0  # down
            volume_tokens[(volume_changes > volume_tau_low) & (volume_changes <= volume_tau_high)] = 1  # steady
            volume_tokens[volume_changes > volume_tau_high] = 2  # up
            
            tokens_dict[f"{coin}_volume"] = volume_tokens
        
        # Create DataFrame with column order: COIN1_price, COIN1_volume, COIN2_price, COIN2_volume, ...
        ordered_cols = []
        for coin in coins:
            if f"{coin}_price" in tokens_dict:
                ordered_cols.append(f"{coin}_price")
                ordered_cols.append(f"{coin}_volume")
        
        tokens_df = pd.DataFrame({col: tokens_dict[col] for col in ordered_cols}, index=df.index)
        
        # Drop rows with NaN (first row from returns calculation)
        tokens_df = tokens_df.dropna()
        
        return tokens_df
    
    def _compute_distribution(self, train_tokens_df: pd.DataFrame, 
                             val_tokens_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Compute token distribution across all coins, channels, and timesteps
        
        Args:
            train_tokens_df: Training DataFrame with token values
            val_tokens_df: Validation DataFrame with token values
            
        Returns:
            Dictionary {token: {train: ratio, val: ratio}}
        """
        def get_ratios(tokens_df):
            all_tokens = tokens_df.values.flatten()
            valid_tokens = all_tokens[~np.isnan(all_tokens)].astype(int)
            
            total = len(valid_tokens)
            if total == 0:
                return {0: 0.0, 1: 0.0, 2: 0.0}
            
            unique, counts = np.unique(valid_tokens, return_counts=True)
            ratios = {int(token): float(count / total) for token, count in zip(unique, counts)}
            
            # Ensure all tokens present
            for token in [0, 1, 2]:
                if token not in ratios:
                    ratios[token] = 0.0
            
            return ratios
        
        train_ratios = get_ratios(train_tokens_df)
        val_ratios = get_ratios(val_tokens_df)
        
        return {
            0: {'train': train_ratios[0], 'val': val_ratios[0]},
            1: {'train': train_ratios[1], 'val': val_ratios[1]},
            2: {'train': train_ratios[2], 'val': val_ratios[2]}
        }

