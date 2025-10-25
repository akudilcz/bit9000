"""Step 4: Tokenize - Convert prices and volume to discrete token sequences

Philosophy: Simple price and volume movements → discrete tokens (256-bin quantization)
- 2 channels: price (log returns) and volume (log changes)
- Quantile-based binning ensures uniform distribution across 256 bins
- No engineered features, just raw movements
- Fit bin edges on train data only (no leakage)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt

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
    """Convert prices and volume to 256-bin token sequences using quantile-based binning"""
    
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
        
        # FIT: Calculate quantile-based bin edges on training data
        logger.info("\n[2/4] FIT: Calculating quantile-based bin edges on TRAINING data...")
        bin_edges = self._fit_bin_edges(train_df, coins)
        
        # Log bin edges info
        vocab_size = self.config['tokenization']['vocab_size']
        logger.info(f"  Using {vocab_size} bins (0-{vocab_size-1})")
        for coin in coins:
            if coin in bin_edges:
                price_edges = bin_edges[coin]['price']
                volume_edges = bin_edges[coin]['volume']
                logger.info(f"  {coin}:")
                logger.info(f"    Price:  {len(price_edges)} bin edges (min={price_edges.min():.6f}, max={price_edges.max():.6f})")
                logger.info(f"    Volume: {len(volume_edges)} bin edges (min={volume_edges.min():.6f}, max={volume_edges.max():.6f})")
        
        # TRANSFORM: Apply bin edges to train and val
        logger.info("\n[3/4] TRANSFORM: Tokenizing train and val data...")
        train_tokens = self._transform_to_tokens(train_df, coins, bin_edges)
        val_tokens = self._transform_to_tokens(val_df, coins, bin_edges)
        
        logger.info(f"  Train tokens: {train_tokens.shape} (columns: {list(train_tokens.columns)})")
        logger.info(f"  Val tokens: {val_tokens.shape}")
        
        # Verify distribution on training data
        token_distribution = self._compute_distribution(train_tokens, val_tokens)
        logger.info("\n  Token distribution (showing first 10 bins):")
        for token in range(min(10, vocab_size)):
            train_pct = token_distribution.get(token, {}).get('train', 0) * 100
            val_pct = token_distribution.get(token, {}).get('val', 0) * 100
            logger.info(f"    {token:3d}: Train={train_pct:5.1f}%, Val={val_pct:5.1f}%")
        
        # Show distribution stats
        train_ratios = [token_distribution.get(i, {}).get('train', 0) for i in range(vocab_size)]
        val_ratios = [token_distribution.get(i, {}).get('val', 0) for i in range(vocab_size)]
        expected_ratio = 1.0 / vocab_size
        
        logger.info(f"\n  Distribution stats:")
        logger.info(f"    Expected uniform ratio: {expected_ratio:.3%}")
        logger.info(f"    Train min/max ratio: {min(train_ratios):.3%} / {max(train_ratios):.3%}")
        logger.info(f"    Val min/max ratio: {min(val_ratios):.3%} / {max(val_ratios):.3%}")
        
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
        
        # Save bin edges (for inference)
        bin_edges_path = block_dir / "bin_edges.json"
        # Convert numpy arrays to lists for JSON serialization
        bin_edges_serializable = {}
        for coin, channels in bin_edges.items():
            bin_edges_serializable[coin] = {
                'price': channels['price'].tolist(),
                'volume': channels['volume'].tolist()
            }
        with open(bin_edges_path, 'w') as f:
            json.dump(bin_edges_serializable, f, indent=2)
        logger.info(f"  Saved bin edges: {bin_edges_path}")
        
        # Create token distribution visualization
        self._plot_token_distribution(token_distribution, vocab_size, block_dir)
        
        # Create artifact
        num_channels = 2  # price + volume
        artifact = TokenizeArtifact(
            train_path=train_path,
            val_path=val_path,
            train_shape=(train_tokens.shape[0], len(coins) * num_channels),
            val_shape=(val_tokens.shape[0], len(coins) * num_channels),
            thresholds_path=bin_edges_path,
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
    
    def _fit_bin_edges(self, train_df: pd.DataFrame, coins: list) -> Dict[str, Dict[str, np.ndarray]]:
        """
        FIT: Calculate quantile-based bin edges on training data for price and volume
        
        For each coin:
        - Price channel: Compute hourly log returns: r = log(close[t] / close[t-1])
        - Volume channel: Compute hourly log changes: v = log(volume[t] / volume[t-1])
        - For each channel, find 255 quantile thresholds to create 256 bins
        
        Args:
            train_df: Training data with COIN_close and COIN_volume columns
            coins: List of coin symbols
            
        Returns:
            Dictionary {coin: {price: bin_edges, volume: bin_edges}}
        """
        vocab_size = self.config['tokenization']['vocab_size']  # 256
        percentiles = self.config['tokenization']['percentiles']  # [1, 2, ..., 99]
        
        bin_edges = {}
        
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
            
            # Calculate quantile-based bin edges
            price_bin_edges = np.percentile(price_returns, percentiles)
            volume_bin_edges = np.percentile(volume_changes, percentiles)
            
            bin_edges[coin] = {
                'price': price_bin_edges,
                'volume': volume_bin_edges
            }
        
        return bin_edges
    
    def _transform_to_tokens(self, df: pd.DataFrame, coins: list, 
                            bin_edges: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
        """
        TRANSFORM: Convert prices and volume to tokens using fitted bin edges
        
        Token mapping (applied independently to each channel):
        - Use np.digitize to assign values to bins (0-255)
        - Values below min bin edge → bin 0
        - Values above max bin edge → bin 255
        
        Args:
            df: DataFrame with COIN_close and COIN_volume columns
            coins: List of coin symbols
            bin_edges: Fitted bin edges {coin: {price: bin_edges, volume: bin_edges}}
            
        Returns:
            DataFrame with columns: COIN_price, COIN_volume (token values 0-255)
        """
        tokens_dict = {}
        
        for coin in coins:
            close_col = f"{coin}_close"
            volume_col = f"{coin}_volume"
            
            if close_col not in df.columns or volume_col not in df.columns:
                logger.warning(f"  Missing columns for {coin}, skipping")
                continue
            
            if coin not in bin_edges:
                logger.warning(f"  No bin edges for {coin}, skipping")
                continue
            
            # PRICE CHANNEL
            prices = df[close_col]
            price_returns = np.log(prices / prices.shift(1))
            
            # Use np.digitize to assign to bins (0-255)
            price_bin_edges = bin_edges[coin]['price']
            price_tokens = np.digitize(price_returns, price_bin_edges)
            
            tokens_dict[f"{coin}_price"] = price_tokens
            
            # VOLUME CHANNEL
            volumes = df[volume_col]
            volume_changes = np.log(volumes / volumes.shift(1))
            volume_changes = volume_changes.replace([np.inf, -np.inf], np.nan)
            
            # Use np.digitize to assign to bins (0-255)
            volume_bin_edges = bin_edges[coin]['volume']
            volume_tokens = np.digitize(volume_changes, volume_bin_edges)
            
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
        vocab_size = self.config['tokenization']['vocab_size']
        
        def get_ratios(tokens_df):
            all_tokens = tokens_df.values.flatten()
            valid_tokens = all_tokens[~np.isnan(all_tokens)].astype(int)
            
            total = len(valid_tokens)
            if total == 0:
                return {i: 0.0 for i in range(vocab_size)}
            
            unique, counts = np.unique(valid_tokens, return_counts=True)
            ratios = {int(token): float(count / total) for token, count in zip(unique, counts)}
            
            # Ensure all tokens present (0-255)
            for token in range(vocab_size):
                if token not in ratios:
                    ratios[token] = 0.0
            
            return ratios
        
        train_ratios = get_ratios(train_tokens_df)
        val_ratios = get_ratios(val_tokens_df)
        
        # Return distribution for all 256 bins
        return {
            i: {'train': train_ratios[i], 'val': val_ratios[i]}
            for i in range(vocab_size)
        }
    
    def _plot_token_distribution(self, token_distribution: Dict[int, Dict[str, float]], 
                                 vocab_size: int, block_dir: Path):
        """
        Create a visualization of the token distribution
        
        Args:
            token_distribution: Dictionary {token: {train: ratio, val: ratio}}
            vocab_size: Number of tokens (256)
            block_dir: Directory to save the plot
        """
        plot_path = block_dir / "token_distribution.png"
        
        # Extract data
        tokens = list(range(vocab_size))
        train_percentages = [token_distribution.get(i, {}).get('train', 0) * 100 for i in tokens]
        val_percentages = [token_distribution.get(i, {}).get('val', 0) * 100 for i in tokens]
        expected_pct = (1.0 / vocab_size) * 100
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Bar chart of distribution
        bar_width = 0.8
        ax1.bar(tokens, train_percentages, width=bar_width, alpha=0.7, label='Train', color='#2E86AB')
        ax1.bar(tokens, val_percentages, width=bar_width, alpha=0.5, label='Val', color='#A23B72')
        ax1.axhline(y=expected_pct, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Expected Uniform ({expected_pct:.3f}%)', alpha=0.7)
        
        ax1.set_xlabel('Token ID (0-255)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Token Distribution: Train vs Val', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.set_xlim(-5, vocab_size + 5)
        
        # Add statistics text
        train_min, train_max = min(train_percentages), max(train_percentages)
        val_min, val_max = min(val_percentages), max(val_percentages)
        stats_text = (
            f'Train: min={train_min:.3f}%, max={train_max:.3f}%, range={train_max-train_min:.3f}%\n'
            f'Val: min={val_min:.3f}%, max={val_max:.3f}%, range={val_max-val_min:.3f}%'
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        # Plot 2: Histogram of percentage values
        ax2.hist(train_percentages, bins=30, alpha=0.7, label='Train', color='#2E86AB', edgecolor='black')
        ax2.hist(val_percentages, bins=30, alpha=0.5, label='Val', color='#A23B72', edgecolor='black')
        ax2.axvline(x=expected_pct, color='red', linestyle='--', linewidth=2, 
                   label=f'Expected ({expected_pct:.3f}%)', alpha=0.7)
        
        ax2.set_xlabel('Token Percentage (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency (# of tokens)', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Token Frequencies', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  Saved token distribution plot: {plot_path}")

