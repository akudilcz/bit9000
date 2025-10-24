"""Step 5: Sequences - Create rolling windows for supervised learning

Philosophy: Simple sliding windows
- Input: 24 consecutive tokens × N coins × 2 channels (price + volume)
- Target: 8 consecutive XRP price tokens (next 8 hours)
- Stack into PyTorch tensors for training
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import ArtifactMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SequencesArtifact:
    """Artifact for sequence data"""
    def __init__(self, train_X_path: Path, train_y_path: Path,
                 val_X_path: Path, val_y_path: Path,
                 train_num_samples: int, val_num_samples: int,
                 input_length: int, output_length: int, num_coins: int,
                 num_channels: int, target_coin: str, metadata: ArtifactMetadata):
        self.train_X_path = train_X_path
        self.train_y_path = train_y_path
        self.val_X_path = val_X_path
        self.val_y_path = val_y_path
        self.train_num_samples = train_num_samples
        self.val_num_samples = val_num_samples
        self.input_length = input_length
        self.output_length = output_length
        self.num_coins = num_coins
        self.num_channels = num_channels
        self.target_coin = target_coin
        self.metadata = metadata
    
    def model_dump(self, mode='json'):
        return {
            'train_X_path': str(self.train_X_path),
            'train_y_path': str(self.train_y_path),
            'val_X_path': str(self.val_X_path),
            'val_y_path': str(self.val_y_path),
            'train_num_samples': self.train_num_samples,
            'val_num_samples': self.val_num_samples,
            'input_length': self.input_length,
            'output_length': self.output_length,
            'num_coins': self.num_coins,
            'num_channels': self.num_channels,
            'target_coin': self.target_coin,
            'metadata': self.metadata.model_dump(mode=mode)
        }


class SequenceBlock(PipelineBlock):
    """Create rolling window sequences for supervised learning"""
    
    def run(self, tokenize_artifact):
        """
        Create sequences from tokenized data
        
        Process:
        1. Load tokenized data (timesteps × coins × channels)
        2. Create rolling windows:
           - Input: 24 hours × all coins × 2 channels (price + volume)
           - Target: 8 hours × target coin price only
        3. Save as PyTorch tensors
        
        Args:
            tokenize_artifact: TokenizeArtifact from step_04_tokenize
            
        Returns:
            SequencesArtifact
        """
        logger.info("="*70)
        logger.info("STEP 5: SEQUENCES - Creating rolling windows")
        logger.info("="*70)
        
        # Get config parameters
        input_length = self.config['sequences']['input_length']  # 24 hours
        output_length = self.config['sequences']['output_length']  # 8 hours
        num_channels = self.config['sequences'].get('num_channels', 2)  # price + volume
        target_coin = self.config['data']['target_coin']  # XRP
        
        logger.info(f"\n  Input length: {input_length} hours")
        logger.info(f"  Output length: {output_length} hours")
        logger.info(f"  Channels: {num_channels} (price + volume)")
        logger.info(f"  Target coin: {target_coin}")
        
        # Load tokenized data
        logger.info("\n[1/3] Loading tokenized data...")
        train_tokens = pd.read_parquet(tokenize_artifact.train_path)
        val_tokens = pd.read_parquet(tokenize_artifact.val_path)
        
        logger.info(f"  Train tokens: {train_tokens.shape}")
        logger.info(f"  Val tokens: {val_tokens.shape}")
        logger.info(f"  Columns: {list(train_tokens.columns)}")
        
        # Verify target coin price column exists
        target_price_col = f"{target_coin}_price"
        if target_price_col not in train_tokens.columns:
            raise ValueError(f"Target coin price column {target_price_col} not found in tokens. Available: {list(train_tokens.columns)}")
        
        # Create sequences
        logger.info("\n[2/3] Creating sequences...")
        train_X, train_y = self._create_sequences(
            train_tokens, input_length, output_length, target_coin, num_channels
        )
        val_X, val_y = self._create_sequences(
            val_tokens, input_length, output_length, target_coin, num_channels
        )
        
        logger.info(f"  Train: X={train_X.shape}, y={train_y.shape}")
        logger.info(f"  Val: X={val_X.shape}, y={val_y.shape}")
        
        # Save as PyTorch tensors
        logger.info("\n[3/3] Saving tensors...")
        block_dir = self.artifact_io.get_block_dir("step_05_sequences", clean=True)
        
        train_X_path = block_dir / "train_X.pt"
        train_y_path = block_dir / "train_y.pt"
        val_X_path = block_dir / "val_X.pt"
        val_y_path = block_dir / "val_y.pt"
        
        torch.save(torch.tensor(train_X, dtype=torch.long), train_X_path)
        torch.save(torch.tensor(train_y, dtype=torch.long), train_y_path)
        torch.save(torch.tensor(val_X, dtype=torch.long), val_X_path)
        torch.save(torch.tensor(val_y, dtype=torch.long), val_y_path)
        
        logger.info(f"  Saved train_X.pt: {train_X.shape}")
        logger.info(f"  Saved train_y.pt: {train_y.shape}")
        logger.info(f"  Saved val_X.pt: {val_X.shape}")
        logger.info(f"  Saved val_y.pt: {val_y.shape}")
        
        # Extract num_coins from shape
        num_coins = train_X.shape[2]
        
        # Create artifact
        artifact = SequencesArtifact(
            train_X_path=train_X_path,
            train_y_path=train_y_path,
            val_X_path=val_X_path,
            val_y_path=val_y_path,
            train_num_samples=train_X.shape[0],
            val_num_samples=val_X.shape[0],
            input_length=input_length,
            output_length=output_length,
            num_coins=num_coins,
            num_channels=num_channels,
            target_coin=target_coin,
            metadata=self.create_metadata(
                upstream_inputs={
                    "train_tokens": str(tokenize_artifact.train_path),
                    "val_tokens": str(tokenize_artifact.val_path)
                }
            )
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_05_sequences",
            artifact_name="sequences_artifact"
        )
        
        logger.info("\n" + "="*70)
        logger.info("SEQUENCES COMPLETE")
        logger.info(f"  Train: {train_X.shape[0]:,} samples")
        logger.info(f"  Val: {val_X.shape[0]:,} samples")
        logger.info(f"  Input: {input_length}h × {num_coins} coins × {num_channels} channels")
        logger.info(f"  Output: {output_length}h × {target_coin} (price only)")
        logger.info("="*70 + "\n")
        
        return artifact
    
    def _create_sequences(self, tokens_df: pd.DataFrame, input_length: int,
                         output_length: int, target_coin: str, num_channels: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling window sequences with 2-channel support
        
        For each valid position i:
        - X[i] = tokens[i:i+input_length, all_coins, all_channels]
          Shape: (input_length, num_coins, num_channels)
        - y[i] = tokens[i+input_length:i+input_length+output_length, target_coin_price]
          Shape: (output_length,) - price only
        
        Args:
            tokens_df: DataFrame with token values (timesteps × coin_channels)
                      Columns like: BTC_price, BTC_volume, ETH_price, ETH_volume, ...
            input_length: Number of input timesteps (24)
            output_length: Number of output timesteps (8)
            target_coin: Coin to predict (e.g., 'XRP')
            num_channels: Number of channels (2 = price + volume)
            
        Returns:
            (X, y) tuple of numpy arrays
            X shape: (num_samples, input_length, num_coins, num_channels)
            y shape: (num_samples, output_length)
        """
        # Extract coin names from columns (assuming format: COIN_price, COIN_volume)
        all_columns = list(tokens_df.columns)
        coin_names = sorted(set(col.rsplit('_', 1)[0] for col in all_columns if '_' in col))
        num_coins = len(coin_names)
        
        logger.info(f"    Detected {num_coins} coins: {coin_names}")
        
        # Verify all required columns exist
        for coin in coin_names:
            for channel in ['price', 'volume']:
                col = f"{coin}_{channel}"
                if col not in all_columns:
                    raise ValueError(f"Missing column: {col}")
        
        # Build 3D array: (timesteps, num_coins, num_channels)
        T = len(tokens_df)
        tokens_array = np.zeros((T, num_coins, num_channels), dtype=np.int64)
        
        for coin_idx, coin in enumerate(coin_names):
            tokens_array[:, coin_idx, 0] = tokens_df[f"{coin}_price"].values
            tokens_array[:, coin_idx, 1] = tokens_df[f"{coin}_volume"].values
        
        # Find target coin index for output
        target_coin_idx = coin_names.index(target_coin)
        
        # Calculate number of valid windows
        window_total = input_length + output_length
        num_samples = T - window_total + 1
        
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for sequences. Need {window_total} timesteps, "
                f"have {T}. Increase data range or decrease window sizes."
            )
        
        logger.info(f"    Creating {num_samples:,} windows from {T} timesteps...")
        logger.info(f"    Input: {input_length} steps × {num_coins} coins × {num_channels} channels")
        logger.info(f"    Output: {output_length} steps × 1 coin ({target_coin} price only)")
        
        # Pre-allocate arrays
        X = np.zeros((num_samples, input_length, num_coins, num_channels), dtype=np.int64)
        y = np.zeros((num_samples, output_length), dtype=np.int64)
        
        # Create windows using vectorized slicing
        for i in range(num_samples):
            # Input window: all coins, all channels (including target coin)
            X[i] = tokens_array[i:i+input_length, :, :]
            
            # Output window: target coin price only (channel 0)
            # No leakage: output hours are AFTER input hours (timesteps i+24 to i+31)
            y[i] = tokens_array[i+input_length:i+input_length+output_length, target_coin_idx, 0]
        
        # Verify no NaNs
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN values detected in sequences!")
        
        return X, y

