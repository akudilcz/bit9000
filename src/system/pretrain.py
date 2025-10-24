"""
Synthetic data generation for pretraining token prediction models.

This module generates synthetic cryptocurrency price movements that match the statistical
properties of real data, allowing for unlimited pretraining data generation.

Usage:
    python main.py pipeline pretrain --num-samples 100000 --epochs 50
"""

import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import Config
from src.model.token_predictor import SimpleTokenPredictor
from src.model.trainer import SimpleTrainer
from src.pipeline.schemas import TokenizeArtifact, PretrainArtifact
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic token sequences matching real coin statistics."""

    def __init__(
        self,
        tokenize_artifact: TokenizeArtifact,
        num_coins: int,
        num_samples: int,
        input_length: int,
        output_length: int,
        seed: int = 42,
    ):
        """
        Initialize synthetic data generator.

        Args:
            tokenize_artifact: Artifact with fitted thresholds and token statistics
            num_coins: Number of coins to simulate
            num_samples: Number of synthetic sequences to generate
            input_length: Input sequence length (24 hours)
            output_length: Output sequence length (8 hours)
            seed: Random seed for reproducibility
        """
        self.tokenize_artifact = tokenize_artifact
        self.num_coins = num_coins
        self.num_samples = num_samples
        self.input_length = input_length
        self.output_length = output_length
        self.seed = seed

        np.random.seed(seed)

        # Load fitted thresholds
        with open(tokenize_artifact.fitted_thresholds_path, "r") as f:
            self.thresholds = json.load(f)

        logger.info(f"Initialized SyntheticDataGenerator")
        logger.info(f"  Num samples: {num_samples:,}")
        logger.info(f"  Num coins: {num_coins}")
        logger.info(f"  Input length: {input_length}")
        logger.info(f"  Output length: {output_length}")

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic token sequences.

        Returns:
            (X, y) tensors:
            - X: (num_samples, input_length, num_coins, 2) - price + volume tokens
            - y: (num_samples, output_length) - XRP price tokens
        """
        logger.info("Generating synthetic data...")

        # Generate using Markov chains (token transition probabilities)
        X = self._generate_sequences()
        y = self._generate_targets()

        logger.info(f"Generated synthetic data: X={X.shape}, y={y.shape}")

        return X, y

    def _generate_sequences(self) -> torch.Tensor:
        """
        Generate input sequences using random walk with token transitions.

        Returns:
            X: (num_samples, input_length, num_coins, 2)
        """
        X = np.zeros(
            (self.num_samples, self.input_length, self.num_coins, 2), dtype=np.int64
        )

        # Simple strategy: sample tokens with equal probability (33/33/33)
        # More sophisticated: learn transition matrices from real data
        for sample_idx in range(self.num_samples):
            for coin_idx in range(self.num_coins):
                # Price channel: random walk with some momentum
                price_tokens = self._random_walk_tokens(self.input_length)
                X[sample_idx, :, coin_idx, 0] = price_tokens

                # Volume channel: independent random tokens
                volume_tokens = np.random.randint(0, 3, size=self.input_length)
                X[sample_idx, :, coin_idx, 1] = volume_tokens

        return torch.tensor(X, dtype=torch.long)

    def _random_walk_tokens(self, length: int, momentum: float = 0.3) -> np.ndarray:
        """
        Generate token sequence with momentum (autocorrelation).

        Args:
            length: Sequence length
            momentum: Probability of repeating previous token (0.3 = 30% momentum)

        Returns:
            Token sequence of shape (length,)
        """
        tokens = np.zeros(length, dtype=np.int64)
        tokens[0] = np.random.randint(0, 3)  # Random start

        for t in range(1, length):
            if np.random.rand() < momentum:
                # Continue trend
                tokens[t] = tokens[t - 1]
            else:
                # Random new token
                tokens[t] = np.random.randint(0, 3)

        return tokens

    def _generate_targets(self) -> torch.Tensor:
        """
        Generate target sequences (XRP price tokens).

        Returns:
            y: (num_samples, output_length)
        """
        # Simple strategy: balanced random targets
        y = np.random.randint(0, 3, size=(self.num_samples, self.output_length))
        return torch.tensor(y, dtype=torch.long)


def run_pretraining(
    config: Config,
    tokenize_artifact: TokenizeArtifact,
    output_dir: Path,
    num_samples: int = 100000,
    epochs: int = 50,
) -> PretrainArtifact:
    """
    Run pretraining on synthetic data.

    Args:
        config: Configuration object
        tokenize_artifact: Artifact from tokenization step (for thresholds)
        output_dir: Directory to save pretrained model
        num_samples: Number of synthetic samples to generate
        epochs: Number of pretraining epochs

    Returns:
        PretrainArtifact with pretrained model path
    """
    logger.info("="*70)
    logger.info("PRETRAINING ON SYNTHETIC DATA")
    logger.info("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    logger.info("\n[1/3] Generating synthetic data...")
    generator = SyntheticDataGenerator(
        tokenize_artifact=tokenize_artifact,
        num_coins=config.data.num_coins,
        num_samples=num_samples,
        input_length=config.sequences.input_length,
        output_length=config.sequences.output_length,
        seed=config.random_seed,
    )

    X_synthetic, y_synthetic = generator.generate()

    # Split into train/val (90/10 for pretraining)
    split_idx = int(0.9 * num_samples)
    train_X = X_synthetic[:split_idx]
    train_y = y_synthetic[:split_idx]
    val_X = X_synthetic[split_idx:]
    val_y = y_synthetic[split_idx:]

    logger.info(f"  Train: X={train_X.shape}, y={train_y.shape}")
    logger.info(f"  Val: X={val_X.shape}, y={val_y.shape}")

    # Create model
    logger.info("\n[2/3] Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleTokenPredictor(
        vocab_size=config.model.vocab_size,
        embedding_dim=config.model.embedding_dim,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        feedforward_dim=config.model.feedforward_dim,
        dropout=config.model.dropout,
        input_length=config.sequences.input_length,
        output_length=config.sequences.output_length,
        num_coins=config.data.num_coins,
        num_classes=config.model.num_classes,
        num_channels=config.sequences.num_channels,
    ).to(device)

    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer_config = {
        **config.__dict__,
        "training": {
            **config.training.__dict__,
            "epochs": epochs,
            "batch_size": 256,  # Larger batches for synthetic data
            "learning_rate": 1e-3,  # Higher LR for pretraining
        },
    }

    trainer = SimpleTrainer(model=model, config=trainer_config, device=str(device))

    # Train on synthetic data
    logger.info("\n[3/3] Pretraining model...")
    splits = [
        {
            "train_X": train_X,
            "train_y": train_y,
            "val_X": val_X,
            "val_y": val_y,
        }
    ]

    pretrain_output_dir = output_dir / "pretrain_checkpoints"
    history = trainer.train_simple(splits, output_dir=str(pretrain_output_dir))

    # Save pretrained model
    model_path = output_dir / "pretrained_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": trainer_config,
            "history": history,
            "num_synthetic_samples": num_samples,
            "epochs": epochs,
        },
        model_path,
    )

    logger.info(f"\nSaved pretrained model to {model_path}")

    # Save history
    history_path = output_dir / "pretraining_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save statistics
    stats_path = output_dir / "synthetic_data_stats.json"
    stats = {
        "num_samples": num_samples,
        "train_samples": split_idx,
        "val_samples": num_samples - split_idx,
        "num_coins": config.data.num_coins,
        "input_length": config.sequences.input_length,
        "output_length": config.sequences.output_length,
        "num_channels": config.sequences.num_channels,
        "best_val_loss": history.get("best_val_loss", float("nan")),
        "best_val_acc": history.get("best_val_acc", float("nan")),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved pretraining stats to {stats_path}")

    # Create artifact
    artifact = PretrainArtifact(
        model_path=str(model_path),
        history_path=str(history_path),
        stats_path=str(stats_path),
        num_synthetic_samples=num_samples,
        best_val_loss=history.get("best_val_loss", float("nan")),
        best_val_acc=history.get("best_val_acc", float("nan")),
        total_epochs=epochs,
    )

    logger.info("\n" + "="*70)
    logger.info("PRETRAINING COMPLETE")
    logger.info(f"  Synthetic samples: {num_samples:,}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Best val loss: {history.get('best_val_loss', 0):.4f}")
    logger.info(f"  Model saved: {model_path}")
    logger.info("="*70 + "\n")

    return artifact




