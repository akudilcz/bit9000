"""
Hyperparameter tuning using Optuna.

This module provides automated hyperparameter optimization for the token predictor model.
It uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for efficient Bayesian
optimization with automatic pruning of unpromising trials.

Usage:
    python main.py pipeline tune --num-trials 30 --epochs-per-trial 20
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.model.token_predictor import SimpleTokenPredictor
from src.model.trainer import Trainer
from src.pipeline.schemas import SequencesArtifact, TuneArtifact
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning for token predictor."""

    def __init__(
        self,
        config: Config,
        sequences_artifact: SequencesArtifact,
        output_dir: Path,
        num_trials: int = 30,
        epochs_per_trial: int = 20,
        timeout_hours: Optional[float] = None,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            config: Base configuration
            sequences_artifact: Artifact from sequence creation step
            output_dir: Directory to save tuning results
            num_trials: Number of optimization trials
            epochs_per_trial: Training epochs per trial (keep short for speed)
            timeout_hours: Optional timeout for entire tuning run
        """
        self.config = config
        self.sequences_artifact = sequences_artifact
        self.output_dir = output_dir
        self.num_trials = num_trials
        self.epochs_per_trial = epochs_per_trial
        self.timeout_seconds = int(timeout_hours * 3600) if timeout_hours else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load sequences
        logger.info("Loading sequences for tuning...")
        sequences_dir = Path(sequences_artifact.train_X_path).parent
        self.train_X = torch.load(sequences_dir / "train_X.pt")
        self.train_y = torch.load(sequences_dir / "train_y.pt")
        self.val_X = torch.load(sequences_dir / "val_X.pt")
        self.val_y = torch.load(sequences_dir / "val_y.pt")

        logger.info(f"Loaded train: X={self.train_X.shape}, y={self.train_y.shape}")
        logger.info(f"Loaded val: X={self.val_X.shape}, y={self.val_y.shape}")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define search space and sample hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled hyperparameters
        """
        # Training hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        dropout = trial.suggest_float("dropout", 0.05, 0.5)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        # Architecture hyperparameters
        embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        feedforward_dim = trial.suggest_categorical(
            "feedforward_dim", [128, 256, 512, 1024]
        )

        # Ensure num_heads divides hidden_dim (embedding_dim * num_coins)
        # For simplicity, we use embedding_dim directly
        # The model will aggregate coins, so d_model = embedding_dim * 2 (price + volume fusion)
        d_model = embedding_dim * 2  # After channel fusion

        # Warmup and scheduler
        warmup_epochs = trial.suggest_int("warmup_epochs", 2, 10)

        return {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "feedforward_dim": feedforward_dim,
            "warmup_epochs": warmup_epochs,
        }

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for a single trial.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss (to be minimized)
        """
        # Sample hyperparameters
        params = self._suggest_hyperparameters(trial)
        logger.info(f"Trial {trial.number}: Testing params {params}")

        try:
            # Create model with sampled hyperparameters
            model = SimpleTokenPredictor(
                vocab_size=self.config.model.vocab_size,
                embedding_dim=params["embedding_dim"],
                d_model=params["d_model"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                feedforward_dim=params["feedforward_dim"],
                dropout=params["dropout"],
                input_length=self.config.sequences.input_length,
                output_length=self.config.sequences.output_length,
                num_coins=self.sequences_artifact.num_coins,
                num_classes=self.config.model.num_classes,
                num_channels=self.config.sequences.num_channels,
            ).to(self.device)

            # Create data loaders
            train_dataset = TensorDataset(self.train_X, self.train_y)
            val_dataset = TensorDataset(self.val_X, self.val_y)

            train_loader = DataLoader(
                train_dataset,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False,
            )

            # Create optimizer with sampled learning rate and weight decay
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )

            # Loss function with label smoothing
            criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])

            # Simple training loop (no full trainer for speed)
            best_val_loss = float("inf")

            for epoch in range(self.epochs_per_trial):
                # Training
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    logits = model(X_batch, targets=y_batch)  # Teacher forcing

                    # Compute loss: (batch, output_length, num_classes) vs (batch, output_length)
                    loss = criterion(
                        logits.reshape(-1, self.config.model.num_classes),
                        y_batch.reshape(-1),
                    )

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params["max_grad_norm"]
                    )

                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        logits = model(X_batch, targets=y_batch)
                        loss = criterion(
                            logits.reshape(-1, self.config.model.num_classes),
                            y_batch.reshape(-1),
                        )
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Track best val loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                # Report intermediate value for pruning
                trial.report(val_loss, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    logger.info(
                        f"Trial {trial.number} pruned at epoch {epoch} (val_loss={val_loss:.4f})"
                    )
                    raise optuna.TrialPruned()

                logger.info(
                    f"Trial {trial.number} Epoch {epoch+1}/{self.epochs_per_trial}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            logger.info(
                f"Trial {trial.number} completed with best_val_loss={best_val_loss:.4f}"
            )
            return best_val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            raise

    def run(self) -> TuneArtifact:
        """
        Run hyperparameter tuning.

        Returns:
            TuneArtifact with best parameters and tuning history
        """
        logger.info(f"Starting hyperparameter tuning with {self.num_trials} trials")
        logger.info(f"Each trial will train for {self.epochs_per_trial} epochs")

        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.config.random_seed),
            pruner=MedianPruner(
                n_startup_trials=5,  # Don't prune first 5 trials
                n_warmup_steps=5,  # Wait 5 epochs before pruning
            ),
        )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=self.num_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=True,
        )

        # Extract results
        best_params = study.best_params
        best_value = study.best_value
        best_trial_number = study.best_trial.number

        logger.info(f"Tuning completed!")
        logger.info(f"Best trial: {best_trial_number}")
        logger.info(f"Best validation loss: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Save results
        results_path = self.output_dir / "tuning_results.json"
        results = {
            "best_trial": best_trial_number,
            "best_val_loss": best_value,
            "best_params": best_params,
            "num_trials": len(study.trials),
            "num_completed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "num_pruned_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "timestamp": datetime.now().isoformat(),
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved tuning results to {results_path}")

        # Save all trials history
        trials_path = self.output_dir / "trials_history.json"
        trials_data = []
        for trial in study.trials:
            trials_data.append(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": trial.params,
                    "duration_seconds": (
                        trial.duration.total_seconds() if trial.duration else None
                    ),
                }
            )

        with open(trials_path, "w") as f:
            json.dump(trials_data, f, indent=2)

        logger.info(f"Saved trials history to {trials_path}")

        # Create visualization
        self._create_visualizations(study)

        # Create artifact
        artifact = TuneArtifact(
            best_params=best_params,
            best_val_loss=best_value,
            best_trial=best_trial_number,
            num_trials=len(study.trials),
            results_path=str(results_path),
            trials_path=str(trials_path),
            plots_dir=str(self.output_dir),
        )

        return artifact

    def _create_visualizations(self, study: optuna.Study) -> None:
        """Create diagnostic plots for tuning results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")

            # 1. Optimization history
            fig, ax = plt.subplots(figsize=(10, 6))
            trials = [t for t in study.trials if t.value is not None]
            trial_numbers = [t.number for t in trials]
            values = [t.value for t in trials]

            ax.plot(trial_numbers, values, marker="o", alpha=0.6)
            ax.axhline(study.best_value, color="r", linestyle="--", label="Best")
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Validation Loss")
            ax.set_title("Hyperparameter Optimization History")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "tuning_history.png", dpi=150)
            plt.close()

            # 2. Parameter importance
            try:
                importances = optuna.importance.get_param_importances(study)
                fig, ax = plt.subplots(figsize=(10, 6))

                params = list(importances.keys())
                importance_values = list(importances.values())

                ax.barh(params, importance_values)
                ax.set_xlabel("Importance")
                ax.set_title("Hyperparameter Importance")
                ax.grid(True, alpha=0.3, axis="x")

                plt.tight_layout()
                plt.savefig(self.output_dir / "param_importance.png", dpi=150)
                plt.close()

            except Exception as e:
                logger.warning(f"Could not create parameter importance plot: {e}")

            logger.info(f"Saved tuning visualizations to {self.output_dir}")

        except ImportError as e:
            logger.warning(f"Could not create visualizations: {e}")


def run_tuning(
    config: Config,
    sequences_artifact: SequencesArtifact,
    output_dir: Path,
    num_trials: int = 30,
    epochs_per_trial: int = 20,
    timeout_hours: Optional[float] = None,
) -> TuneArtifact:
    """
    Run hyperparameter tuning.

    Args:
        config: Configuration object
        sequences_artifact: Artifact from sequence creation step
        output_dir: Directory to save tuning results
        num_trials: Number of optimization trials
        epochs_per_trial: Training epochs per trial
        timeout_hours: Optional timeout for tuning

    Returns:
        TuneArtifact with best parameters
    """
    tuner = HyperparameterTuner(
        config=config,
        sequences_artifact=sequences_artifact,
        output_dir=output_dir,
        num_trials=num_trials,
        epochs_per_trial=epochs_per_trial,
        timeout_hours=timeout_hours,
    )

    return tuner.run()




