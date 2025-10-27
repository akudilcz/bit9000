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
import copy

import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset

from src.model import create_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning for token predictor."""

    def __init__(
        self,
        config: dict,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        val_X: torch.Tensor,
        val_y: torch.Tensor,
        output_dir: Path,
        num_trials: int = 30,
        epochs_per_trial: int = 50,
        timeout_hours: Optional[float] = None,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            config: Base configuration dict
            train_X: Training input sequences (N, 24, 10, 2)
            train_y: Training targets (N,) - single next-hour token
            val_X: Validation input sequences
            val_y: Validation targets
            output_dir: Directory to save tuning results
            num_trials: Number of optimization trials
            epochs_per_trial: Training epochs per trial (keep short for speed)
            timeout_hours: Optional timeout for entire tuning run
        """
        self.config = config
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.output_dir = output_dir
        self.num_trials = num_trials
        self.epochs_per_trial = epochs_per_trial
        self.timeout_seconds = int(timeout_hours * 3600) if timeout_hours else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loaded train: X={self.train_X.shape}, y={self.train_y.shape}")
        logger.info(f"Loaded val: X={self.val_X.shape}, y={self.val_y.shape}")
        
        # Handle both old (N, 8) and new (N,) target formats
        # New design: single next-hour token (N,)
        # Old design: 8-hour output (N, 8)
        if len(self.train_y.shape) > 1 and self.train_y.shape[1] > 1:
            logger.info(f"Detected old sequence format (N, {self.train_y.shape[1]}). Taking first token only.")
            self.train_y = self.train_y[:, 0]  # Take first token (next hour only)
            self.val_y = self.val_y[:, 0]
            logger.info(f"Updated targets: train_y={self.train_y.shape}, val_y={self.val_y.shape}")

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
        # Training hyperparameters - optimized for 3-class problem
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)  # Smaller range for 3-class
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)  # Higher dropout for regularization
        label_smoothing = trial.suggest_float("label_smoothing", 0.01, 0.1)  # Reduced for 3-class
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.5)  # Tighter clipping
        gaussian_noise = trial.suggest_float("gaussian_noise", 0.0, 0.1)  # Reduced noise
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])  # Larger batches for stability

        # Architecture hyperparameters - optimized for 3-class problem
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])  # Direct d_model selection
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 3)  # Fewer layers
        num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 3)  # Fewer layers
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])  # Reasonable head counts
        feedforward_dim = trial.suggest_categorical("feedforward_dim", [128, 256, 512])  # Smaller FF

        # Ensure d_model is divisible by num_heads
        while d_model % num_heads != 0:
            d_model += 1

        # Warmup epochs - shorter for 3-class problem
        warmup_epochs = trial.suggest_int("warmup_epochs", 3, 10)

        return {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size,
            "d_model": d_model,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "num_heads": num_heads,
            "feedforward_dim": feedforward_dim,
            "warmup_epochs": warmup_epochs,
            "gaussian_noise": gaussian_noise,
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
            # Create a copy of config and update with trial hyperparameters
            trial_config = copy.deepcopy(self.config)
            trial_config['model']['d_model'] = params['d_model']
            trial_config['model']['nhead'] = params['num_heads']
            trial_config['model']['num_encoder_layers'] = params['num_encoder_layers']
            trial_config['model']['num_decoder_layers'] = params['num_decoder_layers']
            trial_config['model']['dim_feedforward'] = params['feedforward_dim']
            trial_config['model']['dropout'] = params['dropout']
            trial_config['training']['learning_rate'] = params['learning_rate']
            trial_config['training']['weight_decay'] = params['weight_decay']
            trial_config['training']['label_smoothing'] = params['label_smoothing']
            trial_config['training']['max_grad_norm'] = params['max_grad_norm']
            trial_config['training']['gaussian_noise'] = params['gaussian_noise']
            trial_config['training']['batch_size'] = params['batch_size']
            trial_config['training']['warmup_epochs'] = params['warmup_epochs']

            # Create model with sampled hyperparameters
            model = create_model(trial_config).to(self.device)

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

            # Loss function - use standard cross entropy
            criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])

            # Simple training loop (no full trainer for speed)
            best_val_loss = float('inf')  # Initialize to infinity for loss minimization
            epochs_without_improvement = 0
            early_stopping_patience = 10
            min_delta = 0.000001

            for epoch in range(self.epochs_per_trial):
                # Training
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Squeeze y_batch if it has shape (N, 1) -> (N,)
                    if len(y_batch.shape) > 1:
                        y_batch = y_batch.squeeze(-1)

                    optimizer.zero_grad()
                    
                    # Forward pass: model outputs dict with logits
                    outputs = model(X_batch)
                    
                    # Handle different model architectures
                    if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                        # V4 format: single-horizon binary classification
                        logits = outputs['horizon_1h']['logits']  # (batch, 2)
                    else:
                        # V1/V2/V3 format
                        logits = outputs if not isinstance(outputs, dict) else outputs['logits']

                    # Compute loss
                    loss = criterion(logits, y_batch)

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params["max_grad_norm"]
                    )

                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation - use accuracy (label-smoothing agnostic)
                model.eval()
                val_accuracy = 0.0
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        # Squeeze y_batch if it has shape (N, 1) -> (N,)
                        if len(y_batch.shape) > 1:
                            y_batch = y_batch.squeeze(-1)

                        outputs = model(X_batch)
                        
                        # Handle different model architectures
                        if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                            # V4 format: single-horizon binary classification
                            logits = outputs['horizon_1h']['logits']  # (batch, 2)
                        else:
                            # V1/V2/V3 format
                            logits = outputs if not isinstance(outputs, dict) else outputs['logits']
                        
                        # Compute accuracy (label smoothing doesn't affect this)
                        preds = torch.argmax(logits, dim=1)
                        batch_acc = (preds == y_batch).float().mean().item()
                        val_accuracy += batch_acc
                        
                        # Also track loss for logging
                        loss = criterion(logits, y_batch)
                        val_loss += loss.item()

                val_accuracy /= len(val_loader)
                val_loss /= len(val_loader)

                # Track best val loss (for early stopping)
                if val_loss < best_val_loss - min_delta:  # Reuse variable for best metric
                    best_val_loss = val_loss
                    epochs_without_improvement = 0  # Reset counter
                else:
                    epochs_without_improvement += 1

                # Report validation loss for pruning and optimization (minimize loss)
                trial.report(val_loss, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    logger.info(
                        f"Trial {trial.number} pruned at epoch {epoch} (val_accuracy={val_accuracy:.4f})"
                    )
                    raise optuna.TrialPruned()

                logger.info(
                    f"Trial {trial.number} Epoch {epoch+1}/{self.epochs_per_trial}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.8f}"
                )
                print(
                    f"Trial {trial.number} Epoch {epoch+1}/{self.epochs_per_trial}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.8f}"
                )

                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs, val_acc={val_accuracy:.8f})")
                    break

            logger.info(
                f"Trial {trial.number} completed with best_val_loss={best_val_loss:.8f}"
            )
            print(f"[DONE] Trial {trial.number} completed with best_val_loss={best_val_loss:.8f}")
            return best_val_loss  # Return validation loss (Optuna minimizes)

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            raise

    def run(self) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.

        Returns:
            Dictionary with best parameters and tuning results
        """
        logger.info(f"Starting hyperparameter tuning with {self.num_trials} trials")
        logger.info(f"Each trial will train for {self.epochs_per_trial} epochs")

        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=5,  # Don't prune first 5 trials
                n_warmup_steps=3,  # Wait 3 epochs before pruning
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

        return results

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
            logger.warning(f"Could not create visualizations (missing matplotlib/seaborn): {e}")


def run_tuning(
    config: dict,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    output_dir: Path,
    num_trials: int = 30,
    epochs_per_trial: int = 50,
    timeout_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning.

    Args:
        config: Configuration dict
        train_X: Training input sequences
        train_y: Training target tokens
        val_X: Validation input sequences
        val_y: Validation target tokens
        output_dir: Directory to save tuning results
        num_trials: Number of optimization trials
        epochs_per_trial: Training epochs per trial
        timeout_hours: Optional timeout for tuning

    Returns:
        Dictionary with tuning results and best parameters
    """
    tuner = HyperparameterTuner(
        config=config,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        output_dir=output_dir,
        num_trials=num_trials,
        epochs_per_trial=epochs_per_trial,
        timeout_hours=timeout_hours,
    )

    return tuner.run()




