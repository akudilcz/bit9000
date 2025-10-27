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
from src.pipeline.schemas import ArtifactMetadata, TokenizeArtifact
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
    
    def run(self, tokenize_artifact: TokenizeArtifact = None):
        """
        Create sequences from tokenized data
        
        Process:
        1. Load tokenized data (timesteps × coins × channels)
        2. Create rolling windows:
           - Input: 24 hours × all coins × 2 channels (price + volume)
           - Target: 8 hours × target coin price only
        3. Save as PyTorch tensors
        
        Args:
            tokenize_artifact: TokenizeArtifact from step_05_tokenize (optional, will load from disk if not provided)
            
        Returns:
            SequencesArtifact
        """
        logger.info("="*70)
        logger.info("STEP 5: SEQUENCES - Creating rolling windows")
        logger.info("="*70)
        
        # Load tokenize artifact if not provided
        if tokenize_artifact is None:
            tokenize_artifact_data = self.artifact_io.read_json('artifacts/step_05_tokenize/tokenize_artifact.json')
            tokenize_artifact = TokenizeArtifact(**tokenize_artifact_data)
        
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
        train_X, train_y, train_trading = self._create_sequences(
            train_tokens, input_length, output_length, target_coin, num_channels, prediction_horizon
        )
        val_X, val_y, val_trading = self._create_sequences(
            val_tokens, input_length, output_length, target_coin, num_channels, prediction_horizon
        )
        
        logger.info(f"  Train: X={train_X.shape}, y={train_y.shape}, trading={train_trading.shape}")
        logger.info(f"  Val: X={val_X.shape}, y={val_y.shape}, trading={val_trading.shape}")
        
        # Save as PyTorch tensors
        logger.info("\n[3/3] Saving tensors...")
        block_dir = self.artifact_io.get_block_dir("step_06_sequences", clean=True)
        
        train_X_path = block_dir / "train_X.pt"
        train_y_path = block_dir / "train_y.pt"
        train_trading_path = block_dir / "train_trading.pt"
        val_X_path = block_dir / "val_X.pt"
        val_y_path = block_dir / "val_y.pt"
        val_trading_path = block_dir / "val_trading.pt"
        
        torch.save(torch.tensor(train_X, dtype=torch.long), train_X_path)
        torch.save(torch.tensor(train_y, dtype=torch.long), train_y_path)
        torch.save(torch.tensor(train_trading, dtype=torch.long), train_trading_path)
        torch.save(torch.tensor(val_X, dtype=torch.long), val_X_path)
        torch.save(torch.tensor(val_y, dtype=torch.long), val_y_path)
        torch.save(torch.tensor(val_trading, dtype=torch.long), val_trading_path)
        
        logger.info(f"  Saved train_X.pt: {train_X.shape}")
        logger.info(f"  Saved train_y.pt: {train_y.shape}")
        logger.info(f"  Saved train_trading.pt: {train_trading.shape}")
        logger.info(f"  Saved val_X.pt: {val_X.shape}")
        logger.info(f"  Saved val_y.pt: {val_y.shape}")
        logger.info(f"  Saved val_trading.pt: {val_trading.shape}")
        
        # Extract num_coins from config (not from shape since it's now flattened)
        num_coins = len(self.config['data']['coins'])
        
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
        Create rolling window sequences - DECODER-ONLY VERSION
        
        Flattens tokens into 1D sequences with special tokens marking timesteps and coins.
        Each sample is: [T0] [BTC] btc_data... [ETH] eth_data... ... [T1] ...
        
        For next-token prediction:
        - X[i] = flattened_sequence[:-1]  (input: all but last token)
        - y[i] = flattened_sequence[1:]   (target: shifted by 1)
        
        Args:
            tokens_df: DataFrame with token values (timesteps × coin_channels)
            input_length: Number of input timesteps (48)
            output_length: Not used in decoder-only version
            target_coin: Not used in decoder-only version
            num_channels: Number of channels per coin (19)
            
        Returns:
            (X, y) tuple of numpy arrays
            X shape: (num_samples, seq_len-1) - input sequences
            y shape: (num_samples, seq_len-1) - target sequences (shifted by 1)
        """
        # Get special token configuration
        special_tokens_cfg = self.config.get('special_tokens', {})
        TIMESTEP_TOKEN_START = special_tokens_cfg['timestep_range'][0]  # 21
        COIN_TOKEN_START = special_tokens_cfg['coin_range'][0]  # 69
        
        # Extract coin names from columns - look for coins from config
        coins_from_config = self.config['data']['coins']
        all_columns = list(tokens_df.columns)
        
        # Verify coins have all required channels (19 channels per coin)
        coin_names = []
        channels_list = ['price', 'volume', 'rsi', 'macd', 'bb_position', 'ema_9', 'ema_21', 'ema_50', 'ema_ratio',
                         'stochastic', 'williams_r', 'atr', 'adx', 'obv', 'volume_roc', 'vwap',
                         'price_momentum', 'support_resistance', 'volatility_regime']
        
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
            tokens_array[:, coin_idx, 9] = tokens_df[f"{coin}_stochastic"].values
            tokens_array[:, coin_idx, 10] = tokens_df[f"{coin}_williams_r"].values
            tokens_array[:, coin_idx, 11] = tokens_df[f"{coin}_atr"].values
            tokens_array[:, coin_idx, 12] = tokens_df[f"{coin}_adx"].values
            tokens_array[:, coin_idx, 13] = tokens_df[f"{coin}_obv"].values
            tokens_array[:, coin_idx, 14] = tokens_df[f"{coin}_volume_roc"].values
            tokens_array[:, coin_idx, 15] = tokens_df[f"{coin}_vwap"].values
            tokens_array[:, coin_idx, 16] = tokens_df[f"{coin}_price_momentum"].values
            tokens_array[:, coin_idx, 17] = tokens_df[f"{coin}_support_resistance"].values
            tokens_array[:, coin_idx, 18] = tokens_df[f"{coin}_volatility_regime"].values
        
        # Calculate number of valid windows
        # Each window is input_length timesteps
        # Stride controls overlap: 1 = every timestep, input_length = non-overlapping
        stride = self.config['sequences'].get('sequence_stride', 1)
        num_samples = (T - input_length) // stride + 1
        
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for sequences. Need {input_length} timesteps, "
                f"have {T}. Increase data range or decrease window sizes."
            )
        
        # Calculate sequence length per sample
        # Per timestep: 1 timestep token + num_coins × (1 coin token + num_channels data tokens)
        tokens_per_timestep = 1 + num_coins * (1 + num_channels)
        seq_len = input_length * tokens_per_timestep
        
        logger.info(f"    Creating {num_samples:,} windows from {T} timesteps...")
        logger.info(f"    Window stride: {stride} timesteps (overlap={input_length-stride} timesteps)")
        logger.info(f"    Flattened sequence structure:")
        logger.info(f"      - Tokens per timestep: {tokens_per_timestep} (1 timestep + {num_coins}×{1+num_channels} coins)")
        logger.info(f"      - Total sequence length: {seq_len} tokens ({input_length} timesteps)")
        logger.info(f"      - Order: [T0] [BTC] data... [ETH] data... [T1] [BTC] ...")
        
        # Pre-allocate arrays for flattened sequences
        all_sequences = []
        
        # Create flattened windows with stride
        for i in range(0, T - input_length + 1, stride):
            # Get window of tokens_array: (input_length, num_coins, num_channels)
            window = tokens_array[i:i+input_length, :, :]
            
            # Flatten with special tokens
            sequence = self._flatten_with_special_tokens(
                window, input_length, num_coins, num_channels,
                TIMESTEP_TOKEN_START, COIN_TOKEN_START
            )
            
            all_sequences.append(sequence)
        
        # Convert to numpy array: (num_samples, seq_len)
        all_sequences = np.array(all_sequences, dtype=np.int64)
        
        # Create training pairs for next-token prediction
        X = all_sequences[:, :-1]  # Input: all but last token
        y = all_sequences[:, 1:]   # Target: shifted by 1
        
        logger.info(f"    Next-token prediction:")
        logger.info(f"      Input X: {X.shape} (each position predicts next token)")
        logger.info(f"      Target y: {y.shape} (shifted by 1)")
        logger.info(f"      Token range: data=[0-20], timestep=[21-68], coin=[69-78]")
        
        # Create training pairs with stride of 201 tokens (1 hour)
        # Each training pair predicts the next hour of data (201 tokens = 1 timestep for all 10 coins)
        stride_tokens = 201  # 1 timestep token + 10 coins × 20 (1 coin token + 19 data channels)
        X = all_sequences[:, :-stride_tokens]  # Input: all but last 201 tokens (1 hour)
        y = all_sequences[:, stride_tokens:]    # Target: shifted by 201 tokens (predict next hour)
        
        logger.info(f"    Next-hour prediction (stride={stride_tokens} tokens):")
        logger.info(f"      Input X: {X.shape} (input sequences)")
        logger.info(f"      Target y: {y.shape} (predict next hour for all coins)")
        logger.info(f"      Each target predicts {stride_tokens} tokens = 1 hour of data")
        
        # Generate trading labels (BUY/HOLD/SELL) based on future XRP price
        logger.info(f"    Generating trading labels (BUY/HOLD/SELL)...")
        trading_labels = self._generate_trading_labels(
            tokens_array, coin_names, target_coin, input_length, num_samples, stride
        )
        logger.info(f"      Trading labels: {trading_labels.shape}")
        
        # Log distribution
        unique, counts = np.unique(trading_labels, return_counts=True)
        label_names = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
        for label, count in zip(unique, counts):
            pct = 100.0 * count / len(trading_labels)
            logger.info(f"        {label_names.get(label, label)}: {count} ({pct:.1f}%)")
        
        # Verify no NaNs
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN values detected in sequences!")
        
        return X, y, trading_labels
    
    def _flatten_with_special_tokens(self, tokens_3d: np.ndarray, timesteps: int, 
                                    coins: int, channels: int,
                                    timestep_token_start: int, 
                                    coin_token_start: int) -> np.ndarray:
        """
        Flatten 3D tokens (T, C, Ch) to 1D with special tokens.
        
        Structure: [T0] [BTC] btc_ch0 btc_ch1 ... [ETH] eth_ch0 ... [T1] ...
        
        Args:
            tokens_3d: (T, C, Ch) numpy array of data tokens
            timesteps: Number of timesteps (T)
            coins: Number of coins (C)
            channels: Number of channels per coin (Ch)
            timestep_token_start: Start index for timestep tokens (21)
            coin_token_start: Start index for coin tokens (69)
            
        Returns:
            1D sequence with special tokens
        """
        sequence = []
        
        for t in range(timesteps):
            # Add timestep token
            sequence.append(timestep_token_start + t)
            
            # Add all coins for this timestep
            for c in range(coins):
                # Add coin token
                sequence.append(coin_token_start + c)
                
                # Add all channel data tokens for this coin
                for ch in range(channels):
                    sequence.append(int(tokens_3d[t, c, ch]))
        
        return np.array(sequence, dtype=np.int64)
    
    def _generate_trading_labels(self, tokens_array: np.ndarray, coin_names: list,
                                 target_coin: str, input_length: int, 
                                 num_samples: int, stride: int) -> np.ndarray:
        """
        Generate trading labels (BUY/HOLD/SELL) based on future price movement.
        
        Strategy: 
        - Compare XRP price 8 hours after the window ends vs current price
        - Top 5% price increases → BUY (0)
        - Bottom 5% price decreases → SELL (2)
        - Middle 90% → HOLD (1)
        
        Args:
            tokens_array: (T, C, Ch) array of all tokens
            coin_names: List of coin names
            target_coin: Target coin for trading (e.g. 'XRP')
            input_length: Input window length (48)
            num_samples: Number of samples
            stride: Stride between samples
            
        Returns:
            trading_labels: (num_samples,) array of labels {0: BUY, 1: HOLD, 2: SELL}
        """
        target_coin_idx = coin_names.index(target_coin)
        future_horizon = 8  # Look 8 hours ahead
        T = tokens_array.shape[0]
        
        # Extract price changes for all valid samples
        price_changes = []
        valid_indices = []
        
        for i, start_idx in enumerate(range(0, T - input_length + 1, stride)):
            if i >= num_samples:
                break
            
            # Current price: last price in the input window
            current_idx = start_idx + input_length - 1
            # Future price: 8 hours after window ends
            future_idx = start_idx + input_length + future_horizon - 1
            
            # Check if future index is valid
            if future_idx < T:
                current_price_token = tokens_array[current_idx, target_coin_idx, 0]
                future_price_token = tokens_array[future_idx, target_coin_idx, 0]
                
                # Price change in token space (-20 to +20)
                change = int(future_price_token) - int(current_price_token)
                price_changes.append(change)
                valid_indices.append(i)
            else:
                # Not enough future data - default to HOLD
                price_changes.append(0)
                valid_indices.append(i)
        
        price_changes = np.array(price_changes)
        
        # Calculate percentile thresholds for 5% BUY and 5% SELL
        buy_threshold = np.percentile(price_changes, 95)  # Top 5%
        sell_threshold = np.percentile(price_changes, 5)  # Bottom 5%
        
        # Assign labels
        trading_labels = np.ones(len(price_changes), dtype=np.int64)  # Default: HOLD (1)
        trading_labels[price_changes >= buy_threshold] = 0  # BUY
        trading_labels[price_changes <= sell_threshold] = 2  # SELL
        
        return trading_labels

