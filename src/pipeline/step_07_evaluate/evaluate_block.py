"""Step 7: Evaluate - Validate model quality

Philosophy: Understand prediction quality over horizon
- Per-hour accuracy (hour 1 vs hour 8 degradation)
- Sequence accuracy (all 8 correct)
- Confusion matrices per hour
- Baseline comparisons (persistence, random)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import json

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import ArtifactMetadata
from src.model.token_predictor import SimpleTokenPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvalArtifact:
    """Artifact for evaluation results"""
    def __init__(self, results_path: Path, per_hour_accuracy: list,
                 sequence_accuracy: float, confusion_matrices_dir: Path,
                 baseline_results: dict, metadata: ArtifactMetadata):
        self.results_path = results_path
        self.per_hour_accuracy = per_hour_accuracy
        self.sequence_accuracy = sequence_accuracy
        self.confusion_matrices_dir = confusion_matrices_dir
        self.baseline_results = baseline_results
        self.metadata = metadata
    
    def model_dump(self, mode='json'):
        return {
            'results_path': str(self.results_path),
            'per_hour_accuracy': self.per_hour_accuracy,
            'sequence_accuracy': self.sequence_accuracy,
            'confusion_matrices_dir': str(self.confusion_matrices_dir),
            'baseline_results': self.baseline_results,
            'metadata': self.metadata.model_dump(mode=mode)
        }


class EvaluateBlock(PipelineBlock):
    """Evaluate trained model on validation data"""
    
    def run(self, train_artifact, sequences_artifact):
        """
        Evaluate model
        
        Args:
            train_artifact: TrainedModelArtifact from step_06_train
            sequences_artifact: SequencesArtifact from step_05_sequences
            
        Returns:
            EvalArtifact
        """
        logger.info("="*70)
        logger.info("STEP 7: EVALUATE - Validating model quality")
        logger.info("="*70)
        
        device = self.config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logger.info("\n[1/5] Loading trained model...")
        model = SimpleTokenPredictor(self.config)
        checkpoint = torch.load(train_artifact.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info(f"  Model loaded from: {train_artifact.model_path}")
        
        # Load validation data
        logger.info("\n[2/5] Loading validation data...")
        val_X = torch.load(sequences_artifact.val_X_path)
        val_y = torch.load(sequences_artifact.val_y_path)
        logger.info(f"  Val X: {val_X.shape}")
        logger.info(f"  Val y: {val_y.shape}")
        
        # Generate predictions
        logger.info("\n[3/5] Generating predictions...")
        predictions = self._predict_batch(model, val_X, device, batch_size=256)
        logger.info(f"  Predictions: {predictions.shape}")
        
        # Compute metrics
        logger.info("\n[4/5] Computing metrics...")
        
        # Per-hour accuracy
        per_hour_acc = self._compute_per_hour_accuracy(predictions, val_y.numpy())
        logger.info("\n  Per-hour accuracy:")
        for hour, acc in enumerate(per_hour_acc, 1):
            logger.info(f"    Hour {hour}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Sequence accuracy (all 8 correct)
        sequence_acc = self._compute_sequence_accuracy(predictions, val_y.numpy())
        logger.info(f"\n  Sequence accuracy (all 8 correct): {sequence_acc:.4f} ({sequence_acc*100:.2f}%)")
        
        # Baseline comparisons
        baseline_results = self._compute_baselines(val_X.numpy(), val_y.numpy())
        logger.info("\n  Baseline comparisons:")
        logger.info(f"    Random (33.3%): {baseline_results['random']:.4f}")
        logger.info(f"    Persistence: {baseline_results['persistence']:.4f}")
        logger.info(f"    Model: {np.mean(per_hour_acc):.4f}")
        
        # Save results
        logger.info("\n[5/5] Saving results...")
        block_dir = self.artifact_io.get_block_dir("step_07_evaluate", clean=True)
        
        # Save confusion matrices
        cm_dir = block_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        self._save_confusion_matrices(predictions, val_y.numpy(), cm_dir)
        logger.info(f"  Saved confusion matrices: {cm_dir}")
        
        # Save per-hour accuracy plot
        self._plot_per_hour_accuracy(per_hour_acc, block_dir / "per_hour_accuracy.png")
        logger.info(f"  Saved per-hour accuracy plot")
        
        # Save results JSON
        results = {
            'per_hour_accuracy': per_hour_acc,
            'sequence_accuracy': float(sequence_acc),
            'mean_accuracy': float(np.mean(per_hour_acc)),
            'baseline_results': baseline_results,
            'num_samples': int(val_y.shape[0]),
            'output_length': int(val_y.shape[1])
        }
        
        results_path = block_dir / "eval_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Saved results: {results_path}")
        
        # Create artifact
        artifact = EvalArtifact(
            results_path=results_path,
            per_hour_accuracy=per_hour_acc,
            sequence_accuracy=float(sequence_acc),
            confusion_matrices_dir=cm_dir,
            baseline_results=baseline_results,
            metadata=self.create_metadata(
                upstream_inputs={
                    "model": str(train_artifact.model_path),
                    "val_sequences": str(sequences_artifact.val_X_path)
                }
            )
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_07_evaluate",
            artifact_name="eval_artifact"
        )
        
        logger.info("\n" + "="*70)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"  Mean accuracy: {np.mean(per_hour_acc):.4f} ({np.mean(per_hour_acc)*100:.2f}%)")
        logger.info(f"  Hour 1 accuracy: {per_hour_acc[0]:.4f}")
        logger.info(f"  Hour 8 accuracy: {per_hour_acc[7]:.4f}")
        logger.info(f"  Sequence accuracy: {sequence_acc:.4f}")
        logger.info("="*70 + "\n")
        
        return artifact
    
    def _predict_batch(self, model, X, device, batch_size=256):
        """Generate predictions autoregressively in batches"""
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device)
                # Use autoregressive generation (not teacher forcing)
                preds = model.generate(X_batch, max_len=self.config['sequences']['output_length'])
                predictions.append(preds.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def _compute_per_hour_accuracy(self, predictions, targets):
        """Compute accuracy for each hour"""
        output_length = predictions.shape[1]
        per_hour_acc = []
        
        for hour in range(output_length):
            correct = (predictions[:, hour] == targets[:, hour]).sum()
            acc = correct / len(predictions)
            per_hour_acc.append(float(acc))
        
        return per_hour_acc
    
    def _compute_sequence_accuracy(self, predictions, targets):
        """Compute accuracy where all timesteps are correct"""
        all_correct = (predictions == targets).all(axis=1)
        acc = all_correct.sum() / len(predictions)
        return float(acc)
    
    def _compute_baselines(self, X, y):
        """Compute baseline predictions
        
        Args:
            X: Input tensor (N, input_length, num_coins, num_channels)
            y: Target tensor (N, output_length)
        """
        # Random baseline (uniform 0/1/2)
        random_preds = np.random.randint(0, 3, size=y.shape)
        random_acc = (random_preds == y).mean()
        
        # Persistence baseline: repeat last price token from target coin
        # X shape: (N, 24, num_coins, 2) where channel 0 = price
        # Assume XRP is the last coin (or look it up from config)
        target_coin = self.config['data'].get('target_coin', 'XRP')
        
        # Get last timestep, all coins, price channel (0)
        last_tokens = X[:, -1, :, 0]  # (N, num_coins)
        
        # Use last coin (assuming XRP is last) - in production, look up index
        target_coin_idx = -1  # TODO: Look up actual target coin index
        persistence_token = last_tokens[:, target_coin_idx]  # (N,)
        
        # Repeat for all 8 hours
        persistence_preds = np.repeat(persistence_token[:, np.newaxis], y.shape[1], axis=1)
        persistence_acc = (persistence_preds == y).mean()
        
        return {
            'random': float(random_acc),
            'persistence': float(persistence_acc)
        }
    
    def _save_confusion_matrices(self, predictions, targets, output_dir):
        """Save confusion matrix for each hour"""
        output_length = predictions.shape[1]
        
        for hour in range(output_length):
            # Compute confusion matrix
            preds_hour = predictions[:, hour]
            targets_hour = targets[:, hour]
            
            cm = np.zeros((3, 3), dtype=int)
            for true_label in range(3):
                for pred_label in range(3):
                    cm[true_label, pred_label] = ((targets_hour == true_label) & (preds_hour == pred_label)).sum()
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['down', 'steady', 'up'],
                       yticklabels=['down', 'steady', 'up'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - Hour {hour+1}')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"cm_hour_{hour+1}.png", dpi=100, bbox_inches='tight')
            plt.close()
    
    def _plot_per_hour_accuracy(self, per_hour_acc, output_path):
        """Plot per-hour accuracy"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        hours = list(range(1, len(per_hour_acc) + 1))
        ax.plot(hours, per_hour_acc, marker='o', linewidth=2, markersize=8)
        ax.axhline(y=0.333, color='r', linestyle='--', label='Random baseline (33.3%)')
        
        ax.set_xlabel('Prediction Hour', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Hour Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(hours)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

