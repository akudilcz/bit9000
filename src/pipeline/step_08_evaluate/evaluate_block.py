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
            train_artifact: TrainedModelArtifact from step_07_train
            sequences_artifact: SequencesArtifact from step_06_sequences
            
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
        
        # Load calibrated thresholds if available
        calibrated_threshold_buy = checkpoint.get('calibrated_threshold_buy', checkpoint.get('calibrated_threshold', None))
        calibrated_threshold_sell = checkpoint.get('calibrated_threshold_sell', None)
        
        if calibrated_threshold_buy is not None and calibrated_threshold_sell is not None:
            logger.info(f"  Using calibrated BUY threshold: {calibrated_threshold_buy:.4f}")
            logger.info(f"  Using calibrated SELL threshold: {calibrated_threshold_sell:.4f}")
        elif calibrated_threshold_buy is not None:
            logger.info(f"  Using calibrated threshold: {calibrated_threshold_buy:.4f}")
        else:
            logger.info(f"  No calibrated thresholds found, will use config defaults")
        
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
        block_dir = self.artifact_io.get_block_dir("step_08_evaluate", clean=True)
        
        # Save confusion matrices
        cm_dir = block_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        self._save_confusion_matrices(predictions, val_y.numpy(), cm_dir)
        logger.info(f"  Saved confusion matrices: {cm_dir}")
        
        # Save per-hour accuracy plot
        self._plot_per_hour_accuracy(per_hour_acc, block_dir / "per_hour_accuracy.png")
        logger.info(f"  Saved per-hour accuracy plot")
        
        # Save XRP price chart with buy and sell signals
        self._plot_xrp_with_trading_signals(model, val_X, val_y, sequences_artifact, device, block_dir, 
                                             calibrated_threshold_buy, calibrated_threshold_sell)
        logger.info(f"  Saved XRP price chart with trading signals")
        
        # Save results JSON
        results = {
            'per_hour_accuracy': per_hour_acc,
            'sequence_accuracy': float(sequence_acc),
            'mean_accuracy': float(np.mean(per_hour_acc)),
            'baseline_results': baseline_results,
            'num_samples': int(val_y.shape[0]),
            'output_length': 1  # Single horizon 3-class classification
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
            block_name="step_08_evaluate",
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
        """Generate predictions in batches (3-class: SELL/HOLD/BUY)"""
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device)
                outputs = model(X_batch)
                
                # Extract class predictions
                if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                    logits = outputs['horizon_1h']['logits']  # (B, num_classes)
                else:
                    logits = outputs  # (B, num_classes)
                
                # Get class predictions (0=SELL, 1=HOLD, 2=BUY)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()  # (B,)
                predictions.append(preds)
        
        return np.concatenate(predictions, axis=0)
    
    def _compute_per_hour_accuracy(self, predictions, targets):
        """Compute accuracy for 3-class classification"""
        # For single-horizon 3-class, both predictions and targets are (N,)
        correct = (predictions == targets).sum()
        acc = correct / len(predictions)
        return [float(acc)]  # Return as list for compatibility
    
    def _compute_sequence_accuracy(self, predictions, targets):
        """Compute accuracy for 3-class classification (single horizon)"""
        # For single-horizon, same as per-hour accuracy
        correct = (predictions == targets).sum()
        acc = correct / len(predictions)
        return float(acc)
    
    def _compute_baselines(self, X, y):
        """Compute baseline predictions for 3-class classification
        
        Args:
            X: Input tensor (N, input_length, num_coins, num_channels)
            y: Target tensor (N,) for 3-class classification (0=SELL, 1=HOLD, 2=BUY)
        """
        # Random baseline (33.3% chance of each class)
        random_preds = np.random.randint(0, 3, size=y.shape)
        random_acc = (random_preds == y).mean()
        
        # Persistence baseline: always predict HOLD (1)
        # This is typically the most common class in directional classification
        persistence_preds = np.ones_like(y)
        persistence_acc = (persistence_preds == y).mean()
        
        return {
            'random': float(random_acc),
            'persistence': float(persistence_acc)
        }
    
    def _save_confusion_matrices(self, predictions, targets, output_dir):
        """Save confusion matrix for 3-class classification"""
        # Compute confusion matrix (3x3: SELL/HOLD/BUY)
        cm = np.zeros((3, 3), dtype=int)
        for true_label in range(3):
            for pred_label in range(3):
                cm[true_label, pred_label] = ((targets == true_label) & (predictions == pred_label)).sum()
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['SELL', 'HOLD', 'BUY'],
                   yticklabels=['SELL', 'HOLD', 'BUY'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix - 3-Class Directional Classification')
        
        plt.tight_layout()
        plt.savefig(output_dir / "cm_3class.png", dpi=100, bbox_inches='tight')
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

    def _plot_xrp_with_trading_signals(self, model, val_X, val_y, sequences_artifact, device, output_dir, 
                                        calibrated_threshold_buy=None, calibrated_threshold_sell=None):
        """
        Plot XRP price with BUY and SELL signal overlays on validation data.
        High-resolution chart showing where the model would recommend buying or selling.
        
        Args:
            calibrated_threshold_buy: Optional calibrated threshold for BUY from training. If None, uses config.
            calibrated_threshold_sell: Optional calibrated threshold for SELL from training. If None, uses config.
        """
        logger.info("\n  Generating XRP price chart with trading signals...")
        
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
        
        # Get predictions with probabilities for 3-class (SELL, HOLD, BUY)
        model.eval()
        all_probs = []  # Will store (N, num_classes) logits
        all_tokens = []  # Will store (N,) predicted tokens
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(val_X), batch_size):
                X_batch = val_X[i:i+batch_size].to(device)
                outputs = model(X_batch)
                
                # Extract logits
                if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                    logits = outputs['horizon_1h']['logits']  # (B, 256)
                else:
                    logits = outputs
                
                all_probs.append(logits.cpu().numpy())
                # Get predicted tokens (argmax)
                pred_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
                all_tokens.append(pred_tokens)
        
        all_logits = np.concatenate(all_probs, axis=0)  # (N, 256)
        predicted_tokens = np.concatenate(all_tokens, axis=0)  # (N,)
        
        # Load tokenized validation data to get current tokens
        tokenize_artifact_path = self.artifact_io.base_dir / 'step_05_tokenize' / 'tokenize_artifact.json'
        with open(tokenize_artifact_path, 'r') as f:
            tokenize_artifact = json.load(f)
        val_tokens_df = pd.read_parquet(tokenize_artifact['val_path'])
        
        target_coin = self.config['data']['target_coin']
        price_col = f'{target_coin}_price'
        xrp_price_tokens = val_tokens_df[price_col].values
        
        # Align current tokens with sequences
        input_length = self.config['sequences']['input_length']
        prediction_horizon = self.config['sequences'].get('prediction_horizon', 1)
        num_sequences = len(xrp_price_tokens) - (input_length + prediction_horizon - 1)
        current_tokens = np.array([xrp_price_tokens[i + input_length - 1] for i in range(num_sequences)])
        
        # Calculate predicted token changes
        predicted_token_change = predicted_tokens.astype(np.float32) - current_tokens.astype(np.float32)
        
        # Load calibrated token change thresholds from training
        if isinstance(calibrated_threshold_buy, (int, float, np.number)):
            buy_token_change_threshold = float(calibrated_threshold_buy)
            logger.info(f"    Using calibrated BUY token change threshold from training: {buy_token_change_threshold:.1f}")
        else:
            buy_token_change_threshold = 0.0  # Default: any predicted increase
            logger.info(f"    Using default BUY token change threshold: {buy_token_change_threshold:.1f}")
        
        if isinstance(calibrated_threshold_sell, (int, float, np.number)):
            sell_token_change_threshold = float(calibrated_threshold_sell)
            logger.info(f"    Using calibrated SELL token change threshold from training: {sell_token_change_threshold:.1f}")
        else:
            sell_token_change_threshold = 0.0  # Default: any predicted decrease
            logger.info(f"    Using default SELL token change threshold: {sell_token_change_threshold:.1f}")
        
        # Generate BUY and SELL signals based on token change thresholds
        # BUY: predicted_token_change > threshold (we predict large positive movement)
        # SELL: predicted_token_change < threshold (we predict large negative movement)
        buy_signals = (predicted_token_change > buy_token_change_threshold).astype(int)
        sell_signals = (predicted_token_change < sell_token_change_threshold).astype(int)
        
        # Now we need to align sequences with actual data
        # Sequences are created with input_length window + prediction_horizon offset
        
        # Each sequence looks at [i:i+input_length] and predicts at [i+input_length+prediction_horizon-1]
        # So the "current price" corresponds to position [i+input_length-1] in the original data
        # The number of sequences = len(val_df) - (input_length + prediction_horizon - 1)
        
        # Align actual prices with sequence predictions
        # The current price for sequence i is at position (input_length - 1) + i in the validation data
        sequence_indices = np.arange(len(val_X)) + input_length - 1
        
        # Ensure we don't exceed the validation data bounds
        valid_mask = sequence_indices < len(xrp_prices)
        sequence_indices = sequence_indices[valid_mask]
        predicted_token_change = predicted_token_change[valid_mask]
        predicted_tokens = predicted_tokens[valid_mask]
        buy_signals = buy_signals[valid_mask]
        sell_signals = sell_signals[valid_mask]
        
        # Get current and future prices
        xrp_current_prices = xrp_prices[sequence_indices]
        future_indices = sequence_indices + prediction_horizon
        future_indices = np.clip(future_indices, 0, len(xrp_prices) - 1)
        xrp_future_prices = xrp_prices[future_indices]
        
        # Get current and future tokens (to match training calibration logic)
        xrp_current_tokens = xrp_price_tokens[sequence_indices]
        future_token_indices = sequence_indices + prediction_horizon
        future_token_indices = np.clip(future_token_indices, 0, len(xrp_price_tokens) - 1)
        xrp_future_tokens = xrp_price_tokens[future_token_indices]
        
        # Calculate actual token changes (same as training)
        actual_token_change = xrp_future_tokens.astype(np.float32) - xrp_current_tokens.astype(np.float32)
        
        # Determine correctness based on TOKEN changes (matching training calibration)
        is_up_token = actual_token_change > 0  # Future token > current token
        is_down_token = actual_token_change < 0  # Future token < current token
        
        # Also calculate price changes for visualization
        price_changes = xrp_future_prices - xrp_current_prices
        is_up_price = price_changes > 0
        is_down_price = price_changes < 0
        
        # Get aligned timestamps
        aligned_timestamps = timestamps[sequence_indices] if hasattr(timestamps, '__getitem__') else sequence_indices
        
        # Create high-resolution chart (8000x8000 pixels total)
        # 8000 pixels / 200 DPI = 40 inch figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 40), dpi=200)
        
        # Plot full price history (no subsampling)
        x_axis = aligned_timestamps if isinstance(aligned_timestamps, pd.DatetimeIndex) else np.arange(len(xrp_current_prices))
        
        # Plot 1: Actual XRP price with BUY and SELL signals as markers
        # First plot signals (so they appear below the line)
        buy_mask = buy_signals == 1
        sell_mask = sell_signals == 1
        
        if buy_mask.sum() > 0:
            buy_x = x_axis[buy_mask] if isinstance(x_axis, pd.DatetimeIndex) else x_axis[buy_mask]
            buy_y = xrp_current_prices[buy_mask]
            ax1.scatter(buy_x, buy_y, 
                       s=400, color='lime', marker='^', edgecolors='darkgreen', linewidth=3,
                       label=f'BUY Signal ({buy_mask.sum()} signals)', zorder=3)
        
        if sell_mask.sum() > 0:
            sell_x = x_axis[sell_mask] if isinstance(x_axis, pd.DatetimeIndex) else x_axis[sell_mask]
            sell_y = xrp_current_prices[sell_mask]
            ax1.scatter(sell_x, sell_y, 
                       s=400, color='red', marker='v', edgecolors='darkred', linewidth=3,
                       label=f'SELL Signal ({sell_mask.sum()} signals)', zorder=3)
        
        # Then plot price line on top (higher zorder)
        ax1.plot(x_axis, xrp_current_prices, 
                label='XRP Price (Actual)', linewidth=3, color='steelblue', alpha=0.95, zorder=5)
        
        # Add subtle background shading for signal correctness
        if isinstance(x_axis, pd.DatetimeIndex):
            # For time series, use axvspan with actual timestamps
            for i in range(len(buy_signals)):
                if buy_signals[i]:
                    color = 'lightgreen' if is_up_token[i] else 'lightcoral'
                    if i < len(x_axis) - 1:
                        ax1.axvspan(x_axis[i], x_axis[i+1], alpha=0.12, color=color, zorder=0)
                elif sell_signals[i]:
                    color = 'lightcoral' if is_down_token[i] else 'lightgreen'
                    if i < len(x_axis) - 1:
                        ax1.axvspan(x_axis[i], x_axis[i+1], alpha=0.12, color=color, zorder=0)
        else:
            # For indices, use axvspan with indices
            for i in range(len(buy_signals)):
                if buy_signals[i]:
                    color = 'lightgreen' if is_up_token[i] else 'lightcoral'
                    ax1.axvspan(x_axis[i], x_axis[i]+1, alpha=0.12, color=color, zorder=0)
                elif sell_signals[i]:
                    color = 'lightcoral' if is_down_token[i] else 'lightgreen'
                    ax1.axvspan(x_axis[i], x_axis[i]+1, alpha=0.12, color=color, zorder=0)
        
        ax1.set_xlabel('Time' if isinstance(x_axis, pd.DatetimeIndex) else 'Sample Index', 
                       fontsize=28, fontweight='bold')
        ax1.set_ylabel('XRP Price (USD)', fontsize=28, fontweight='bold')
        ax1.set_title(f'XRP Validation Period with BUY/SELL Signals ({prediction_horizon}h ahead prediction)', 
                     fontsize=32, fontweight='bold')
        ax1.legend(fontsize=24, loc='best', markerscale=1.5)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax1.tick_params(axis='both', which='major', labelsize=22)
        
        # Format x-axis for datetime
        if isinstance(x_axis, pd.DatetimeIndex):
            import matplotlib.dates as mdates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Predicted token changes with thresholds
        ax2.plot(x_axis, predicted_token_change, 
                label='Predicted Token Change', linewidth=3, color='steelblue', alpha=0.8)
        ax2.axhline(y=buy_token_change_threshold, color='green', linestyle='--', linewidth=3, 
                   label=f'BUY Threshold ({buy_token_change_threshold:.1f})')
        ax2.axhline(y=sell_token_change_threshold, color='red', linestyle='--', linewidth=3, 
                   label=f'SELL Threshold ({sell_token_change_threshold:.1f})')
        ax2.fill_between(x_axis, buy_token_change_threshold, predicted_token_change.max(),
                        where=(predicted_token_change > buy_token_change_threshold),
                        alpha=0.2, color='lime', label='BUY Region')
        ax2.fill_between(x_axis, predicted_token_change.min(), sell_token_change_threshold,
                        where=(predicted_token_change < sell_token_change_threshold),
                        alpha=0.2, color='lightcoral', label='SELL Region')
        
        ax2.set_xlabel('Time' if isinstance(x_axis, pd.DatetimeIndex) else 'Sample Index', 
                       fontsize=28, fontweight='bold')
        ax2.set_ylabel('Token Change', fontsize=28, fontweight='bold')
        ax2.set_title('Model Predicted Token Changes (Movement)', fontsize=32, fontweight='bold')
        ax2.legend(fontsize=24, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax2.tick_params(axis='both', which='major', labelsize=22)
        
        # Format x-axis for datetime
        if isinstance(x_axis, pd.DatetimeIndex):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = output_dir / "xrp_chart_with_trading_signals.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"    Saved high-resolution chart: {output_path}")
        
        # Generate statistics for BUY and SELL signals (using TOKEN-based correctness like training)
        num_buys = buy_signals.sum()
        num_correct_buys = (buy_signals * is_up_token).sum()
        buy_precision = num_correct_buys / num_buys if num_buys > 0 else 0
        buy_rate = num_buys / len(buy_signals)
        
        num_sells = sell_signals.sum()
        num_correct_sells = (sell_signals * is_down_token).sum()
        sell_precision = num_correct_sells / num_sells if num_sells > 0 else 0
        sell_rate = num_sells / len(sell_signals)
        
        logger.info(f"    Trading Signal Statistics (TOKEN-BASED, matching training):")
        logger.info(f"      BUY Signals: {num_buys} ({buy_rate:.1%} of samples)")
        logger.info(f"        Correct (future token > current token): {num_correct_buys}")
        logger.info(f"        Precision: {buy_precision:.1%}")
        logger.info(f"      SELL Signals: {num_sells} ({sell_rate:.1%} of samples)")
        logger.info(f"        Correct (future token < current token): {num_correct_sells}")
        logger.info(f"        Precision: {sell_precision:.1%}")
        
        # Save stats
        stats = {
            'total_buy_signals': int(num_buys),
            'correct_buy_signals': int(num_correct_buys),
            'buy_precision': float(buy_precision),
            'buy_signal_rate': float(buy_rate),
            'total_sell_signals': int(num_sells),
            'correct_sell_signals': int(num_correct_sells),
            'sell_precision': float(sell_precision),
            'sell_signal_rate': float(sell_rate),
            'chart_path': str(output_path)
        }
        
        stats_path = output_dir / "trading_signal_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"    Saved statistics: {stats_path}")

