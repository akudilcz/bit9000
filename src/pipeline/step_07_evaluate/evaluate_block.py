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
from src.model import create_model
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
        model = create_model(self.config)
        checkpoint = torch.load(train_artifact.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Load calibrated threshold if available
        calibrated_threshold = checkpoint.get('calibrated_threshold', None)
        if calibrated_threshold is not None:
            logger.info(f"  Using calibrated threshold: {calibrated_threshold:.4f}")
        else:
            logger.info(f"  No calibrated threshold found, will use config default")
        
        logger.info(f"  Model loaded from: {train_artifact.model_path}")
        
        # Load validation data
        logger.info("\n[2/5] Loading validation data...")
        val_X = torch.load(sequences_artifact.val_X_path, weights_only=False)
        val_y = torch.load(sequences_artifact.val_y_path, weights_only=False)
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
        
        # Save XRP price chart with buy signals
        self._plot_xrp_with_buy_signals(model, val_X, val_y, sequences_artifact, device, block_dir, calibrated_threshold)
        logger.info(f"  Saved XRP price chart with buy signals")
        
        # Save results JSON
        results = {
            'per_hour_accuracy': per_hour_acc,
            'sequence_accuracy': float(sequence_acc),
            'mean_accuracy': float(np.mean(per_hour_acc)),
            'baseline_results': baseline_results,
            'num_samples': int(val_y.shape[0]),
            'output_length': 1  # Single horizon binary classification
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
        if len(per_hour_acc) > 1:
            logger.info(f"  Hour 8 accuracy: {per_hour_acc[7]:.4f}")
        logger.info(f"  Sequence accuracy: {sequence_acc:.4f}")
        logger.info("="*70 + "\n")
        
        return artifact
    
    def _predict_batch(self, model, X, device, batch_size=256):
        """Generate predictions in batches (binary classification: BUY/NO-BUY)"""
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device)
                outputs = model(X_batch)
                
                # For binary classification, extract class predictions
                if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                    logits = outputs['horizon_1h']['logits']  # (B, 2)
                else:
                    logits = outputs  # (B, 2)
                
                # Get class predictions (0=NO-BUY, 1=BUY)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()  # (B,)
                predictions.append(preds)
        
        return np.concatenate(predictions, axis=0)
    
    def _compute_per_hour_accuracy(self, predictions, targets):
        """Compute accuracy for binary classification"""
        # For single-horizon binary, both predictions and targets are (N,)
        correct = (predictions == targets).sum()
        acc = correct / len(predictions)
        return [float(acc)]  # Return as list for compatibility
    
    def _compute_sequence_accuracy(self, predictions, targets):
        """Compute accuracy for binary classification (single horizon)"""
        # For single-horizon, same as per-hour accuracy
        correct = (predictions == targets).sum()
        acc = correct / len(predictions)
        return float(acc)
    
    def _compute_baselines(self, X, y):
        """Compute baseline predictions for binary classification
        
        Args:
            X: Input tensor (N, input_length, num_coins, num_channels)
            y: Target tensor (N,) for binary classification
        """
        # Random baseline (50% chance of each class)
        random_preds = np.random.randint(0, 2, size=y.shape)
        random_acc = (random_preds == y).mean()
        
        # Persistence baseline: always predict NO-BUY (0)
        # This is the most common class in imbalanced binary classification
        persistence_preds = np.zeros_like(y)
        persistence_acc = (persistence_preds == y).mean()
        
        return {
            'random': float(random_acc),
            'persistence': float(persistence_acc)
        }
    
    def _save_confusion_matrices(self, predictions, targets, output_dir):
        """Save confusion matrix for binary classification"""
        # Compute confusion matrix (2x2 for binary: NO-BUY vs BUY)
        cm = np.zeros((2, 2), dtype=int)
        for true_label in range(2):
            for pred_label in range(2):
                cm[true_label, pred_label] = ((targets == true_label) & (predictions == pred_label)).sum()
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['NO-BUY', 'BUY'],
                   yticklabels=['NO-BUY', 'BUY'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix - Binary BUY/NO-BUY Classification')
        
        plt.tight_layout()
        plt.savefig(output_dir / "cm_binary.png", dpi=100, bbox_inches='tight')
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

    def _plot_xrp_with_buy_signals(self, model, val_X, val_y, sequences_artifact, device, output_dir, calibrated_threshold=None):
        """
        Plot XRP price with buy signal overlays on validation data.
        High-resolution chart showing where the model would recommend buying.
        
        Args:
            calibrated_threshold: Optional calibrated threshold from training. If None, uses config.
        """
        logger.info("\n  Generating XRP price chart with buy signals...")
        
        # Load bin edges from tokenize artifacts
        from pathlib import Path
        tokenize_dir = Path("artifacts/step_04_tokenize")
        bin_edges_path = tokenize_dir / "bin_edges.json"
        
        if not bin_edges_path.exists():
            logger.warning("  Could not find bin_edges.json, skipping XRP chart")
            return
        
        with open(bin_edges_path, 'r') as f:
            bin_edges_data = json.load(f)
        
        # Get XRP price bins
        target_coin = sequences_artifact.target_coin
        if target_coin not in bin_edges_data:
            logger.warning(f"  Target coin {target_coin} not found in bin edges, skipping XRP chart")
            return
        
        price_bins = np.array(bin_edges_data[target_coin]['price'])
        
        # Get predictions with probabilities
        model.eval()
        buy_probs = []
        buy_signals = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(val_X), batch_size):
                X_batch = val_X[i:i+batch_size].to(device)
                outputs = model(X_batch)
                
                # Extract buy probabilities (class 1 from binary classification)
                if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                    logits = outputs['horizon_1h']['logits']  # (B, 2)
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(BUY)
                else:
                    probs = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                
                buy_probs.extend(probs)
        
        buy_probs = np.array(buy_probs)
        
        # Use calibrated threshold from training if available, otherwise fall back to config
        if calibrated_threshold is not None:
            decision_threshold = calibrated_threshold
            logger.info(f"    Using calibrated threshold from training: {decision_threshold:.4f}")
        else:
            decision_threshold = self.config['inference'].get('single_threshold', 0.5)
            logger.info(f"    Using config threshold: {decision_threshold:.4f}")
        
        buy_signals = (buy_probs >= decision_threshold).astype(int)
        
        # Decode target coin prices (XRP is last coin, channel 0 is price)
        target_coin_idx = -1  # XRP is last coin
        xrp_current_tokens = val_X[:, -1, target_coin_idx, 0].numpy()  # (N,)
        xrp_future_tokens = val_y.numpy()  # (N,) for single-horizon
        
        # Convert tokens to prices using bin edges
        def token_to_price(tokens, bins):
            """Convert token indices to price values"""
            prices = []
            for token in tokens:
                if 0 <= token < len(bins):
                    prices.append(bins[int(token)])
                else:
                    prices.append(bins[0])  # Default to first bin
            return np.array(prices)
        
        xrp_current_prices = token_to_price(xrp_current_tokens, price_bins)
        xrp_future_prices = token_to_price(xrp_future_tokens, price_bins)
        
        # Calculate price changes
        price_changes = xrp_future_prices - xrp_current_prices
        is_up = price_changes > 0
        
        # Create high-resolution chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), dpi=150)
        
        # Subsample for clarity (plot every 10th point)
        sample_indices = np.arange(0, len(xrp_current_prices), 10)
        
        # Plot 1: Price chart with buy signals
        ax1.plot(sample_indices, xrp_current_prices[sample_indices], 
                label='Current XRP Price', linewidth=2, color='steelblue', alpha=0.8)
        ax1.scatter(sample_indices, xrp_current_prices[sample_indices], 
                   s=30, color='steelblue', alpha=0.5, zorder=2)
        
        # Overlay buy signals
        buy_mask = buy_signals[sample_indices] == 1
        buy_indices = sample_indices[buy_mask]
        ax1.scatter(buy_indices, xrp_current_prices[buy_indices], 
                   s=200, color='lime', marker='^', edgecolors='darkgreen', linewidth=2,
                   label='BUY Signal', zorder=5)
        
        # Color background by correctness
        for i in range(len(sample_indices) - 1):
            idx = sample_indices[i]
            next_idx = sample_indices[i+1]
            
            if buy_signals[idx]:
                # If buy signal issued, color green if correct, red if wrong
                color = 'lightgreen' if is_up[idx] else 'lightcoral'
                alpha = 0.1
                ax1.axvspan(i, i+1, alpha=alpha, color=color)
        
        ax1.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
        ax1.set_ylabel('XRP Price ($)', fontsize=14, fontweight='bold')
        ax1.set_title('XRP Validation Period with Model BUY Signals', 
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Buy probability with threshold
        ax2.plot(sample_indices, buy_probs[sample_indices], 
                label='BUY Probability', linewidth=2, color='steelblue', alpha=0.8)
        ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        ax2.fill_between(sample_indices, 0, 1, where=(buy_signals[sample_indices]==1),
                        alpha=0.2, color='lime', label='BUY Region')
        
        ax2.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Probability', fontsize=14, fontweight='bold')
        ax2.set_title('Model Confidence for BUY Decisions', fontsize=16, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=12, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = output_dir / "xrp_chart_with_buy_signals.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"    Saved high-resolution chart: {output_path}")
        
        # Generate statistics
        num_buys = buy_signals.sum()
        num_correct_buys = (buy_signals * is_up).sum()
        buy_precision = num_correct_buys / num_buys if num_buys > 0 else 0
        buy_rate = num_buys / len(buy_signals)
        
        logger.info(f"    BUY Signal Statistics:")
        logger.info(f"      Total signals: {num_buys}")
        logger.info(f"      Correct (price went up): {num_correct_buys}")
        logger.info(f"      Precision: {buy_precision:.1%}")
        logger.info(f"      Signal rate: {buy_rate:.1%}")
        
        # Save stats
        stats = {
            'total_buy_signals': int(num_buys),
            'correct_buy_signals': int(num_correct_buys),
            'buy_precision': float(buy_precision),
            'buy_signal_rate': float(buy_rate),
            'chart_path': str(output_path)
        }
        
        stats_path = output_dir / "buy_signal_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"    Saved statistics: {stats_path}")

