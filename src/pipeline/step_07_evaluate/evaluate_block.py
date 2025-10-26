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
        
        # Load actual cleaned data with real prices
        from pathlib import Path
        split_dir = Path("artifacts/step_03_split")
        val_data_path = split_dir / "val_clean.parquet"
        
        if not val_data_path.exists():
            logger.warning("  Could not find val_clean.parquet, skipping XRP chart")
            return
        
        val_df = pd.read_parquet(val_data_path)
        
        # Get target coin (XRP) actual prices
        target_coin = sequences_artifact.target_coin
        price_col = f"{target_coin}_close"  # Use close price
        
        if price_col not in val_df.columns:
            logger.warning(f"  Price column {price_col} not found in validation data, skipping XRP chart")
            return
        
        # Extract XRP prices and timestamps from validation data
        xrp_prices = val_df[price_col].values
        timestamps = val_df.index if isinstance(val_df.index, pd.DatetimeIndex) else pd.to_datetime(val_df['timestamp']) if 'timestamp' in val_df.columns else np.arange(len(val_df))
        
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
        
        # Now we need to align sequences with actual data
        # Sequences are created with input_length window + prediction_horizon offset
        input_length = self.config['sequences']['input_length']
        prediction_horizon = self.config['sequences'].get('prediction_horizon', 1)
        
        # Each sequence looks at [i:i+input_length] and predicts at [i+input_length+prediction_horizon-1]
        # So the "current price" corresponds to position [i+input_length-1] in the original data
        # The number of sequences = len(val_df) - (input_length + prediction_horizon - 1)
        
        # Align actual prices with sequence predictions
        # The current price for sequence i is at position (input_length - 1) + i in the validation data
        sequence_indices = np.arange(len(val_X)) + input_length - 1
        
        # Ensure we don't exceed the validation data bounds
        valid_mask = sequence_indices < len(xrp_prices)
        sequence_indices = sequence_indices[valid_mask]
        buy_probs = buy_probs[valid_mask]
        buy_signals = buy_signals[valid_mask]
        
        # Get current and future prices
        xrp_current_prices = xrp_prices[sequence_indices]
        future_indices = sequence_indices + prediction_horizon
        future_indices = np.clip(future_indices, 0, len(xrp_prices) - 1)
        xrp_future_prices = xrp_prices[future_indices]
        
        # Calculate price changes
        price_changes = xrp_future_prices - xrp_current_prices
        is_up = price_changes > 0
        
        # Get aligned timestamps
        aligned_timestamps = timestamps[sequence_indices] if hasattr(timestamps, '__getitem__') else sequence_indices
        
        # Create high-resolution chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), dpi=150)
        
        # Plot full price history (no subsampling)
        x_axis = aligned_timestamps if isinstance(aligned_timestamps, pd.DatetimeIndex) else np.arange(len(xrp_current_prices))
        
        # Plot 1: Actual XRP price with buy signals as markers on the line
        ax1.plot(x_axis, xrp_current_prices, 
                label='XRP Price (Actual)', linewidth=2, color='steelblue', alpha=0.9, zorder=1)
        
        # Overlay buy signals directly on the price line
        buy_mask = buy_signals == 1
        if buy_mask.sum() > 0:
            buy_x = x_axis[buy_mask] if isinstance(x_axis, pd.DatetimeIndex) else x_axis[buy_mask]
            buy_y = xrp_current_prices[buy_mask]
            ax1.scatter(buy_x, buy_y, 
                       s=300, color='lime', marker='^', edgecolors='darkgreen', linewidth=2.5,
                       label=f'BUY Signal ({buy_mask.sum()} signals)', zorder=5)
        
        # Add subtle background shading for BUY correctness
        if isinstance(x_axis, pd.DatetimeIndex):
            # For time series, use axvspan with actual timestamps
            for i in range(len(buy_signals)):
                if buy_signals[i]:
                    color = 'lightgreen' if is_up[i] else 'lightcoral'
                    if i < len(x_axis) - 1:
                        ax1.axvspan(x_axis[i], x_axis[i+1], alpha=0.15, color=color, zorder=0)
        else:
            # For indices, use axvspan with indices
            for i in range(len(buy_signals)):
                if buy_signals[i]:
                    color = 'lightgreen' if is_up[i] else 'lightcoral'
                    ax1.axvspan(x_axis[i], x_axis[i]+1, alpha=0.15, color=color, zorder=0)
        
        ax1.set_xlabel('Time' if isinstance(x_axis, pd.DatetimeIndex) else 'Sample Index', 
                       fontsize=14, fontweight='bold')
        ax1.set_ylabel('XRP Price (USD)', fontsize=14, fontweight='bold')
        ax1.set_title(f'XRP Validation Period with Model BUY Signals ({prediction_horizon}h ahead prediction)', 
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis for datetime
        if isinstance(x_axis, pd.DatetimeIndex):
            import matplotlib.dates as mdates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Buy probability with threshold
        ax2.plot(x_axis, buy_probs, 
                label='BUY Probability', linewidth=2, color='steelblue', alpha=0.8)
        ax2.axhline(y=decision_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Decision Threshold ({decision_threshold:.4f})')
        ax2.fill_between(x_axis, 0, 1, where=(buy_signals==1),
                        alpha=0.2, color='lime', label='BUY Region')
        
        ax2.set_xlabel('Time' if isinstance(x_axis, pd.DatetimeIndex) else 'Sample Index', 
                       fontsize=14, fontweight='bold')
        ax2.set_ylabel('Probability', fontsize=14, fontweight='bold')
        ax2.set_title('Model Confidence for BUY Decisions', fontsize=16, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=12, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis for datetime
        if isinstance(x_axis, pd.DatetimeIndex):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
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

