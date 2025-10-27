"""Step 5: Sequences - Create rolling windows for supervised learning

Philosophy: Simple sliding windows
- Input: 48 consecutive tokens × N coins × 9 channels (price + volume + indicators)
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
            tokenize_artifact: TokenizeArtifact from step_05_tokenize
            
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
        prediction_horizon = self.config['sequences'].get('prediction_horizon', 1)  # hours ahead to predict
        
        logger.info(f"\n  Input length: {input_length} hours")
        logger.info(f"  Output length: {output_length} hours")
        logger.info(f"  Prediction horizon: {prediction_horizon} hours ahead")
        logger.info(f"  Channels: {num_channels} (price + volume + rsi + macd + bb_position + ema_9 + ema_21 + ema_50 + ema_ratio)")
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
            train_tokens, input_length, output_length, target_coin, num_channels, prediction_horizon
        )
        val_X, val_y = self._create_sequences(
            val_tokens, input_length, output_length, target_coin, num_channels, prediction_horizon
        )
        
        logger.info(f"  Train: X={train_X.shape}, y={train_y.shape}")
        logger.info(f"  Val: X={val_X.shape}, y={val_y.shape}")
        
        # Save as PyTorch tensors
        logger.info("\n[3/3] Saving tensors...")
        block_dir = self.artifact_io.get_block_dir("step_06_sequences", clean=True)
        
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
            block_name="step_06_sequences",
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
                         output_length: int, target_coin: str, num_channels: int,
                         prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling window sequences with 9-channel support
        
        For each valid position i:
        - X[i] = tokens[i:i+input_length, all_coins, all_channels]
          Shape: (input_length, num_coins, num_channels)
        - y[i] = tokens[i+input_length:i+input_length+output_length, target_coin_price]
          Shape: (output_length,) - price only
        
        Args:
            tokens_df: DataFrame with token values (timesteps × coin_channels)
                      Columns like: BTC_price, BTC_volume, BTC_rsi, BTC_macd, BTC_bb_position, 
                                    BTC_ema_9, BTC_ema_21, BTC_ema_50, BTC_ema_ratio, ...
            input_length: Number of input timesteps (48)
            output_length: Number of output timesteps (8)
            target_coin: Coin to predict (e.g., 'XRP')
            num_channels: Number of channels (9 = price + volume + rsi + macd + bb_position + ema_9 + ema_21 + ema_50 + ema_ratio)
            
        Returns:
            (X, y) tuple of numpy arrays
            X shape: (num_samples, input_length, num_coins, num_channels)
            y shape: (num_samples, output_length)
        """
        # Extract coin names from columns - look for coins from config
        coins_from_config = self.config['data']['coins']
        all_columns = list(tokens_df.columns)
        
        # Verify coins have all required channels (9 channels per coin)
        coin_names = []
        channels_list = ['price', 'volume', 'rsi', 'macd', 'bb_position', 'ema_9', 'ema_21', 'ema_50', 'ema_ratio']
        
        for coin in coins_from_config:
            has_all_channels = all(f"{coin}_{ch}" in all_columns for ch in channels_list)
            if has_all_channels:
                coin_names.append(coin)
            else:
                logger.warning(f"  Skipping {coin}: missing some channels")
        
        num_coins = len(coin_names)
        logger.info(f"    Detected {num_coins} coins: {coin_names}")
        
        # Build 3D array: (timesteps, num_coins, num_channels)
        T = len(tokens_df)
        tokens_array = np.zeros((T, num_coins, num_channels), dtype=np.int64)
        
        for coin_idx, coin in enumerate(coin_names):
            tokens_array[:, coin_idx, 0] = tokens_df[f"{coin}_price"].values
            tokens_array[:, coin_idx, 1] = tokens_df[f"{coin}_volume"].values
            tokens_array[:, coin_idx, 2] = tokens_df[f"{coin}_rsi"].values
            tokens_array[:, coin_idx, 3] = tokens_df[f"{coin}_macd"].values
            tokens_array[:, coin_idx, 4] = tokens_df[f"{coin}_bb_position"].values
            tokens_array[:, coin_idx, 5] = tokens_df[f"{coin}_ema_9"].values
            tokens_array[:, coin_idx, 6] = tokens_df[f"{coin}_ema_21"].values
            tokens_array[:, coin_idx, 7] = tokens_df[f"{coin}_ema_50"].values
            tokens_array[:, coin_idx, 8] = tokens_df[f"{coin}_ema_ratio"].values
        
        # Find target coin index for output
        target_coin_idx = coin_names.index(target_coin)
        
        # Calculate number of valid windows
        # Need: input_length + (prediction_horizon - 1) steps to look prediction_horizon ahead
        window_total = input_length + prediction_horizon - 1
        num_samples = T - window_total
        
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for sequences. Need {window_total} timesteps, "
                f"have {T}. Increase data range or decrease window sizes."
            )
        
        logger.info(f"    Creating {num_samples:,} windows from {T} timesteps...")
        logger.info(f"    Input: {input_length} steps × {num_coins} coins × {num_channels} channels")
        logger.info(f"    Output: Single-horizon prediction {prediction_horizon}h ahead for {target_coin} price")
        
        # Pre-allocate arrays
        X = np.zeros((num_samples, input_length, num_coins, num_channels), dtype=np.int64)
        
        # Targets: single-horizon 8h ahead (binary BUY/NO-BUY)
        y_1h = np.zeros(num_samples, dtype=np.int64)
        # Current price token (at end of input window)
        y_current = np.zeros(num_samples, dtype=np.int64)
        
        # Create windows using vectorized slicing
        for i in range(num_samples):
            # Input window: all coins, all channels (including target coin)
            X[i] = tokens_array[i:i+input_length, :, :]
            
            # Single-horizon output: predict at prediction_horizon hours after input window
            y_1h[i] = tokens_array[i+input_length+prediction_horizon-1, target_coin_idx, 0]  # prediction_horizon hours ahead
            # Current price at end of input window (for directional comparison)
            y_current[i] = tokens_array[i+input_length-1, target_coin_idx, 0]  # current price
        
        # Single horizon array: (num_samples,)
        y_single = y_1h
        
        # Convert 256-bin targets to 3-class (SELL, HOLD, BUY) labels
        num_classes = self.config['model'].get('num_classes', 256)
        binary_classification = self.config['model'].get('binary_classification', False)
        
        if binary_classification:
            # Binary classification: DIRECTIONAL - BUY if price goes UP, NO-BUY if flat/down
            y_binary = np.zeros_like(y_single, dtype=np.int64)
            y_binary[y_1h > y_current] = 1  # BUY: future price > current price
            y_binary[y_1h <= y_current] = 0  # NO-BUY: future price <= current price
            y = y_binary
            
            # Log binary class distribution
            class_counts = np.bincount(y.flatten())
            total_samples = y.size
            buy_count = class_counts[1] if len(class_counts) > 1 else 0
            no_buy_count = class_counts[0] if len(class_counts) > 0 else 0
            logger.info(f"    {prediction_horizon}h horizon - Binary distribution:")
            logger.info(f"      Class 0 (NO-BUY): {no_buy_count:,} ({100*no_buy_count/total_samples:.1f}%)")
            logger.info(f"      Class 1 (BUY): {buy_count:,} ({100*buy_count/total_samples:.1f}%)")
        elif num_classes == 3:
            # 3-class DIRECTIONAL classification (no magnitude threshold): SELL, HOLD, BUY
            # Use token difference sign as a proxy for raw price direction
            token_diff = y_1h.astype(np.float32) - y_current.astype(np.float32)
            
            # Class 0 (SELL): future token < current token
            # Class 1 (HOLD): future token == current token
            # Class 2 (BUY):  future token > current token
            y_3class = np.ones_like(y_single, dtype=np.int64)  # HOLD by default
            y_3class[token_diff < 0] = 0
            y_3class[token_diff > 0] = 2
            y = y_3class
            
            # Log class distribution
            class_counts = np.bincount(y.flatten())
            total_samples = y.size
            logger.info(f"    {prediction_horizon}h horizon - 3-class directional (no threshold) distribution:")
            logger.info(f"      Class 0 (SELL): {class_counts[0]:,} ({100*class_counts[0]/total_samples:.1f}%)")
            logger.info(f"      Class 1 (HOLD): {class_counts[1]:,} ({100*class_counts[1]/total_samples:.1f}%)")
            logger.info(f"      Class 2 (BUY): {class_counts[2]:,} ({100*class_counts[2]/total_samples:.1f}%)")
        else:
            # Use raw token as target (256-class token prediction)
            y = y_1h
            
            # Log token distribution
            logger.info(f"    {prediction_horizon}h horizon - 256-class token prediction:")
            logger.info(f"      Min token: {y.min()}, Max token: {y.max()}, Mean token: {y.astype(np.float32).mean():.1f}")
            unique_tokens = len(np.unique(y))
            logger.info(f"      Unique tokens in targets: {unique_tokens}/256")
        
        # Verify no NaNs
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN values detected in sequences!")
        
        return X, y

