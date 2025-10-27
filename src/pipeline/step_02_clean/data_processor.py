
"""Data processing: tokenization, sequence creation, and splitting"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CryptoDataset(Dataset):
    """PyTorch Dataset for cryptocurrency data"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, config: dict = None):
        """
        Initialize dataset
        
        Args:
            sequences: Input sequences, shape (samples, seq_len, num_coins)
            targets: Target labels, shape (samples, num_coins)
            config: Configuration dictionary
        """
        config = config or {}
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class DataProcessor:
    """Process data for model training"""
    
    def __init__(self, config: Dict, bin_calculator):
        """
        Initialize data processor
        
        Args:
            config: Configuration dictionary
            bin_calculator: BinCalculator instance
        """
        self.config = config
        self.bin_calculator = bin_calculator
        self.model_config = config['model']
        self.training_config = config['training']
        
        self.sequence_length = self.model_config['sequence_length']
        self.num_coins = self.model_config['num_coins']
        
        self.scaler = StandardScaler()
    
    def create_sequences(self, features: pd.DataFrame, bins: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for transformer input (WITHOUT normalization)
        Normalization will be done per walk-forward split
        
        Args:
            features: DataFrame with engineered features
            bins: DataFrame with bin assignments
            
        Returns:
            Tuple of (sequences, targets)
            sequences: shape (samples, seq_len, num_features) - NOT NORMALIZED
            targets: shape (samples, num_coins)
        """
        logger.info(f"Creating sequences with length {self.sequence_length}")
        
        # Convert to numpy arrays
        features_array = features.values  # (timesteps, num_features)
        bins_array = bins.values  # (timesteps, num_coins)
        
        num_samples = len(features_array) - self.sequence_length
        num_features = features_array.shape[1]
        num_coins_actual = bins_array.shape[1]  # Infer from actual data
        
        # Handle edge case: not enough data for even one sequence
        if num_samples <= 0:
            logger.warning(f"Insufficient data for sequences: {len(features_array)} timesteps < {self.sequence_length} sequence_length")
            return np.zeros((0, self.sequence_length, num_features)), np.zeros((0, num_coins_actual), dtype=int)
        
        sequences = np.zeros((num_samples, self.sequence_length, num_features))
        targets = np.zeros((num_samples, num_coins_actual), dtype=int)
        
        for i in range(num_samples):
            # Input: UN-NORMALIZED features for last seq_length hours
            sequences[i] = features_array[i:i+self.sequence_length]
            
            # Target: bin labels for next hour
            targets[i] = bins_array[i+self.sequence_length]
        
        logger.info(f"Created {num_samples} sequences (UN-NORMALIZED)")
        logger.info(f"Sequences shape: {sequences.shape}")
        logger.info(f"Targets shape: {targets.shape}")
        logger.info(f"Features per timestep: {num_features} ({num_features//num_coins_actual} per coin × {num_coins_actual} coins)")
        
        return sequences, targets
    
    def create_walk_forward_splits(self, sequences: np.ndarray, targets: np.ndarray) -> list:
        """
        Create walk-forward validation splits with fixed rolling window
        
        Args:
            sequences: Input sequences
            targets: Target labels
            
        Returns:
            List of (train_data, val_data) tuples
        """
        # Read window sizes from config
        walk_forward_config = self.config.get('training', {}).get('walk_forward', {})
        train_window = walk_forward_config.get('train_window_hours', 720)
        val_window = walk_forward_config.get('val_window_hours', 168)
        step_size = walk_forward_config.get('step_size_hours', 168)
        
        splits = []
        total_samples = len(sequences)
        
        # Start after we have enough data for first training window
        for start_idx in range(0, total_samples - train_window - val_window, step_size):
            train_start = start_idx
            train_end = start_idx + train_window
            val_start = train_end
            val_end = val_start + val_window
            
            if val_end > total_samples:
                break
            
            train_X = sequences[train_start:train_end]
            train_y = targets[train_start:train_end]
            val_X = sequences[val_start:val_end]
            val_y = targets[val_start:val_end]
            
            splits.append(((train_X, train_y), (val_X, val_y)))
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        logger.info(f"  Train window: {train_window} samples ({train_window/24:.0f} days)")
        logger.info(f"  Val window: {val_window} samples ({val_window/24:.0f} days)")
        logger.info(f"  Step size: {step_size} samples ({step_size/24:.0f} days)")
        
        return splits
    
    def split_data(self, sequences: np.ndarray, targets: np.ndarray) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Split data into train, validation, and test sets
        (DEPRECATED - use create_walk_forward_splits for proper validation)
        
        Args:
            sequences: Input sequences
            targets: Target labels
            
        Returns:
            Tuple of ((train_X, train_y), (val_X, val_y), (test_X, test_y))
        """
        train_split = self.training_config['train_split']
        val_split = self.training_config['val_split']
        
        total_samples = len(sequences)
        train_end = int(total_samples * train_split)
        val_end = int(total_samples * (train_split + val_split))
        
        train_X = sequences[:train_end]
        train_y = targets[:train_end]
        
        val_X = sequences[train_end:val_end]
        val_y = targets[train_end:val_end]
        
        test_X = sequences[val_end:]
        test_y = targets[val_end:]
        
        logger.info(f"Train set: {len(train_X)} samples")
        logger.info(f"Validation set: {len(val_X)} samples")
        logger.info(f"Test set: {len(test_X)} samples")
        
        return (train_X, train_y), (val_X, val_y), (test_X, test_y)
    
    def create_dataloaders(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders
        
        Args:
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            test_data: Test data (X, y)
            batch_size: Batch size (if None, use from config)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.training_config['batch_size']
        
        train_dataset = CryptoDataset(*train_data)
        val_dataset = CryptoDataset(*val_data)
        test_dataset = CryptoDataset(*test_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Created dataloaders with batch size {batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def process_all(self, price_data: pd.DataFrame) -> Dict:
        """
        Complete data processing pipeline with walk-forward validation
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Dictionary with walk-forward splits (no leakage)
        """
        logger.info("Starting walk-forward data processing pipeline")
        
        # Calculate features and bins using rolling price quartiles (NO LEAKAGE - shifted)
        features, bins = self.bin_calculator.calculate_rolling_price_bins(price_data)
        
        # Create ALL sequences first (without normalization)
        sequences, targets = self.create_sequences(features, bins)
        
        # Create walk-forward splits
        splits = self.create_walk_forward_splits(sequences, targets)
        
        # For each split, fit scaler on train data only (NO LEAKAGE)
        processed_splits = []
        for idx, ((train_X, train_y), (val_X, val_y)) in enumerate(splits):
            logger.info(f"\nProcessing split {idx + 1}/{len(splits)}")
            
            # Fit scaler on training data ONLY
            scaler = StandardScaler()
            train_X_normalized = scaler.fit_transform(
                train_X.reshape(-1, train_X.shape[-1])
            ).reshape(train_X.shape)
            
            # Transform validation data with fitted scaler
            val_X_normalized = scaler.transform(
                val_X.reshape(-1, val_X.shape[-1])
            ).reshape(val_X.shape)
            
            processed_splits.append({
                'train_X': train_X_normalized,
                'train_y': train_y,
                'val_X': val_X_normalized,
                'val_y': val_y,
                'scaler': scaler
            })
            
            logger.info(f"  Train: {train_X_normalized.shape}, Val: {val_X_normalized.shape}")
        
        # Save the last split's scaler for inference (if we have splits)
        processed_dir = self.config['data']['processed_dir']
        os.makedirs(processed_dir, exist_ok=True)
        
        if processed_splits:
            import pickle
            with open(os.path.join(processed_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(processed_splits[-1]['scaler'], f)
            logger.info(f"Saved last split's scaler to {processed_dir}")
        else:
            logger.warning("No walk-forward splits created - insufficient data")
        
        logger.info(f"\nCreated {len(processed_splits)} walk-forward splits (NO LEAKAGE)")
        
        return {
            'features': features,
            'bins': bins,
            'splits': processed_splits,
            'num_splits': len(processed_splits)
        }
    
    def load_processed_data(self) -> Dict:
        """
        Load previously processed data
        
        Returns:
            Dictionary with processed data and dataloaders
        """
        processed_dir = self.config['data']['processed_dir']
        
        train_X = np.load(os.path.join(processed_dir, 'train_X.npy'))
        train_y = np.load(os.path.join(processed_dir, 'train_y.npy'))
        val_X = np.load(os.path.join(processed_dir, 'val_X.npy'))
        val_y = np.load(os.path.join(processed_dir, 'val_y.npy'))
        test_X = np.load(os.path.join(processed_dir, 'test_X.npy'))
        test_y = np.load(os.path.join(processed_dir, 'test_y.npy'))
        
        train_data = (train_X, train_y)
        val_data = (val_X, val_y)
        test_data = (test_X, test_y)
        
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_data, val_data, test_data
        )
        
        logger.info(f"Loaded processed data from {processed_dir}")
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }

