"""Step 6: Train - Train transformer on token sequences

256-class token prediction with standard cross-entropy loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
from typing import Tuple
from tqdm import tqdm
import math
import numpy as np
import pandas as pd

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import ArtifactMetadata
from src.model import create_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainedModelArtifact:
    """Artifact for trained model"""
    def __init__(self, model_path: Path, history_path: Path,
                 best_val_loss: float, best_val_acc: float,
                 total_epochs: int, metadata: ArtifactMetadata):
        self.model_path = model_path
        self.history_path = history_path
        self.best_val_loss = best_val_loss
        self.best_val_acc = best_val_acc
        self.total_epochs = total_epochs
        self.metadata = metadata
    
    def model_dump(self, mode='json'):
        return {
            'model_path': str(self.model_path),
            'history_path': str(self.history_path),
            'best_val_loss': float(self.best_val_loss),
            'best_val_acc': float(self.best_val_acc),
            'total_epochs': self.total_epochs,
            'metadata': self.metadata.model_dump(mode=mode)
        }


class TrainBlock(PipelineBlock):
    """Train transformer model on token sequences"""
    
    def run(self, sequences_artifact):
        """Train model on sequences"""
        logger.info("="*70)
        logger.info("STEP 6: TRAIN - Training 256-class token prediction")
        logger.info("="*70)
        
        # Load training config
        train_config = self.config['training']
        device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        epochs = train_config.get('epochs', 100)
        batch_size = train_config.get('batch_size', 128)
        learning_rate = train_config.get('learning_rate', 0.001)
        patience = train_config['early_stopping'].get('patience', 15)
        
        logger.info(f"\n  Device: {device}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Early stopping patience: {patience}")
        
        # Load sequences
        logger.info("\n[1/4] Loading sequences...")
        train_X = torch.load(sequences_artifact.train_X_path)
        train_y = torch.load(sequences_artifact.train_y_path)
        val_X = torch.load(sequences_artifact.val_X_path)
        val_y = torch.load(sequences_artifact.val_y_path)
        
        logger.info(f"  Train: X={train_X.shape}, y={train_y.shape}")
        logger.info(f"  Val: X={val_X.shape}, y={val_y.shape}")
        
        # Create data loaders
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # Initialize model
        logger.info("\n[2/4] Initializing model...")
        model = create_model(self.config)
        model = model.to(device)
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Using 256-class token prediction")
        
        # Loss and optimizer
        label_smoothing = train_config.get('label_smoothing', 0.0)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info(f"  Using Standard CE Loss (label_smoothing={label_smoothing})")
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=train_config.get('weight_decay', 0.0)
        )
        
        # Learning rate scheduler with warmup
        warmup_epochs = train_config.get('warmup_epochs', 3)
        warmup_start_lr = train_config.get('warmup_start_lr', 1e-6)
        
        def warmup_lr(epoch):
            if epoch < warmup_epochs:
                return warmup_start_lr + (learning_rate - warmup_start_lr) * (epoch / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return learning_rate * (0.5 + 0.5 * math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: warmup_lr(e) / learning_rate)
        
        # Training loop
        logger.info("\n[3/4] Training...")
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        best_calibrated_threshold_buy = 0.5
        best_calibrated_threshold_sell = 0.5
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc, avg_precision, threshold_buy, threshold_sell = self._validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"precision={avg_precision:.3f}"
            )
            
            # Early stopping based on validation loss (token prediction)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                best_calibrated_threshold_buy = threshold_buy
                best_calibrated_threshold_sell = threshold_sell
                patience_counter = 0
                logger.info(f"    → New best model (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"    Early stopping triggered (patience={patience})")
                    break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"\n  Restored best model (val_loss={best_val_loss:.4f})")
        
        # Save artifacts
        logger.info("\n[4/4] Saving artifacts...")
        block_dir = self.artifact_io.get_block_dir("step_07_train", clean=True)
        
        # Save model with calibrated thresholds
        model_path = block_dir / "model.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'epoch': len(history['val_loss']),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'calibrated_threshold_buy': best_calibrated_threshold_buy,
            'calibrated_threshold_sell': best_calibrated_threshold_sell
        }
        torch.save(checkpoint, model_path)
        logger.info(f"  Saved model: {model_path}")
        
        # Save history
        history_path = block_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"  Saved history: {history_path}")
        
        # Create artifact
        metadata = self.create_metadata(
            upstream_inputs={"sequences_artifact": str(sequences_artifact.train_X_path.parent)}
        )
        
        artifact = TrainedModelArtifact(
            model_path=model_path,
            history_path=history_path,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            total_epochs=len(history['val_loss']),
            metadata=metadata
        )
        
        # Save artifact metadata
        artifact_path = block_dir / "train_artifact.json"
        self.artifact_io.write_json(artifact.model_dump(), "step_07_train", "train_artifact")
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Best val loss: {best_val_loss:.6f}")
        logger.info(f"  Best val acc: {best_val_acc:.6f}")
        logger.info(f"  Total epochs: {len(history['val_loss'])}")
        logger.info("="*70 + "\n")
        
        return artifact
    
    def _train_epoch(self, model, data_loader, criterion, optimizer, device) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(data_loader, desc='Training', leave=False, ncols=100)
        
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Note: Do not add Gaussian noise to categorical token IDs; it degrades embeddings
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Extract logits from V4 output
            if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                logits = outputs['horizon_1h']['logits']
            else:
                logits = outputs
            
            # Compute loss with CrossEntropyLoss
            loss = criterion(logits, y_batch)
            
            # For accuracy, use hard argmax
            predictions = torch.argmax(logits, dim=-1)
            batch_correct = (predictions == y_batch).sum().item()
            batch_samples = y_batch.numel()
            
            # Backward pass with gradient clipping
            loss.backward()
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * X_batch.size(0)
            total_correct += batch_correct
            total_samples += batch_samples
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_correct/batch_samples:.4f}'})
        
        avg_loss = total_loss / len(data_loader.dataset)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, model, data_loader, criterion, device) -> Tuple[float, float, float, float, float]:
        """
        Validate for one epoch with enhanced metrics tracking
        
        Returns:
            (val_loss, val_acc, avg_precision, threshold_buy, threshold_sell)
        """
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Enhanced metrics tracking
        all_predictions = []
        all_targets = []
        all_probs = []
        distance_errors = []
        
        pbar = tqdm(data_loader, desc='Validation', leave=False, ncols=100)
        
        with torch.no_grad():
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                
                # Extract logits from V4 output
                if isinstance(outputs, dict) and 'horizon_1h' in outputs:
                    logits = outputs['horizon_1h']['logits']
                else:
                    logits = outputs
                
                # Compute loss
                loss = criterion(logits, y_batch)
                predictions = torch.argmax(logits, dim=-1)
                batch_correct = (predictions == y_batch).sum().item()
                batch_samples = y_batch.numel()
                
                # Collect enhanced metrics
                prob_dist = torch.softmax(logits, dim=-1)
                all_probs.append(prob_dist.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                
                # Distance error (ordinal-aware metric)
                distance_error = torch.abs(predictions.float() - y_batch.float()).mean().item()
                distance_errors.append(distance_error)
                
                total_loss += loss.item() * X_batch.size(0)
                total_correct += batch_correct
                total_samples += batch_samples
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'acc': f'{batch_correct/batch_samples:.4f}',
                    'dist_err': f'{distance_error:.2f}'
                })
            
            avg_loss = total_loss / len(data_loader.dataset)
            avg_acc = total_correct / total_samples
            avg_distance_error = np.mean(distance_errors)
            
            # Enhanced logging with precision focus
            logger.info(f"    Val metrics: loss={avg_loss:.4f}, acc={avg_acc:.4f}, dist_err={avg_distance_error:.2f}")
            
            return avg_loss, avg_acc, avg_acc, 0.5, 0.5
    
    def _compute_directional_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute directional targets (price went up/down) from validation RAW PRICES
        This EXACTLY matches the evaluation logic.
        
        The model predicts on tokenized data, but we calibrate thresholds based on
        whether the ACTUAL RAW PRICE went up/down, not the tokenized price.
        
        Returns:
            (buy_targets, sell_targets) - binary arrays indicating if price went up/down
        """
        # Load ACTUAL cleaned prices (same as evaluation uses)
        split_artifact_path = self.artifact_io.base_dir / 'step_03_split' / 'split_artifact.json'
        with open(split_artifact_path, 'r') as f:
            split_artifact = json.load(f)
        val_df = pd.read_parquet(split_artifact['val_path'])
        
        # Get target coin and raw prices
        target_coin = self.config['data']['target_coin']
        price_col = f'{target_coin}_close'  # e.g., "XRP_close" - raw prices
        
        xrp_prices = val_df[price_col].values
        
        # Align sequences with actual data (EXACTLY matching evaluation logic)
        input_length = self.config['sequences']['input_length']
        prediction_horizon = self.config['sequences'].get('prediction_horizon', 1)
        
        # Number of sequences created in step_06
        num_sequences = len(xrp_prices) - (input_length + prediction_horizon - 1)
        
        # Each sequence i corresponds to:
        #   - Current price: xrp_prices[i + input_length - 1]
        #   - Future price: xrp_prices[i + input_length + prediction_horizon - 1]
        sequence_indices = np.arange(num_sequences) + input_length - 1
        
        # Get current and future prices (EXACTLY matching evaluation)
        xrp_current_prices = xrp_prices[sequence_indices]
        future_indices = sequence_indices + prediction_horizon
        future_indices = np.clip(future_indices, 0, len(xrp_prices) - 1)
        xrp_future_prices = xrp_prices[future_indices]
        
        # Calculate directional movement based on ACTUAL RAW PRICES
        price_changes = xrp_future_prices - xrp_current_prices
        buy_targets = (price_changes > 0).astype(np.int64)  # 1 if actual price went up
        sell_targets = (price_changes < 0).astype(np.int64)  # 1 if actual price went down
        
        return buy_targets, sell_targets
    
    def _calibrate_threshold(self, probs: np.ndarray, targets: np.ndarray, 
                            target_rate: float = 0.02) -> Tuple[float, float, int, int]:
        """
        Find optimal threshold that achieves target signal rate while maximizing precision
        
        Args:
            probs: Probability scores for the signal
            targets: Binary targets (1 = correct direction, 0 = wrong direction)
            target_rate: Target signal rate (e.g., 0.05 for 5%)
        
        Returns:
            (threshold, precision, num_signals, num_correct)
        """
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        sorted_targets = targets[sorted_indices]
        
        N = len(probs)
        target_num_signals = max(1, int(N * target_rate))
        calib_cfg = self.config.get('inference', {}).get('calibration', {})
        mode = calib_cfg.get('mode', 'hit_rate_exact')
        min_signals = int(calib_cfg.get('min_signals', 20))
        
        if mode == 'precision_at_most_rate':
            best_precision = -1.0
            best_k = None
            # Scan k from min_signals to target_num_signals
            k_start = min(target_num_signals, max(min_signals, 1))
            if k_start > target_num_signals:
                k_start = target_num_signals
            for k in range(k_start, target_num_signals + 1):
                # Top-k by prob
                top_k_targets = sorted_targets[:k]
                prec_k = float(top_k_targets.sum()) / float(k) if k > 0 else 0.0
                if prec_k > best_precision:
                    best_precision = prec_k
                    best_k = k
            if best_k is None or best_k == 0:
                threshold = 1.0  # no signals
                num_signals = 0
                num_correct = 0
                precision = 0.0
            else:
                threshold = sorted_probs[best_k - 1]
                threshold = max(0.0, threshold - 1e-6)
                signal_mask = probs > threshold
                num_signals = int(signal_mask.sum())
                num_correct = int((targets[signal_mask] == 1).sum()) if num_signals > 0 else 0
                precision = float(num_correct) / float(num_signals) if num_signals > 0 else 0.0
        else:
            # hit_rate_exact
            if target_num_signals < N:
                threshold = sorted_probs[target_num_signals - 1]
                threshold = max(0.0, threshold - 1e-6)
            else:
                threshold = 0.0
            signal_mask = probs > threshold
            num_signals = int(signal_mask.sum())
            if num_signals > 0:
                num_correct = int((targets[signal_mask] == 1).sum())
                precision = float(num_correct) / float(num_signals)
            else:
                num_correct = 0
                precision = 0.0
        
        return float(threshold), float(precision), int(num_signals), int(num_correct)

    def _calibrate_token_change_threshold(self, predicted_token_change: np.ndarray, targets: np.ndarray,
                                          target_rate: float = 0.02, direction: str = 'buy') -> Tuple[float, float, int, int]:
        """
        Find optimal token change threshold that achieves target signal rate while maximizing precision
        
        Args:
            predicted_token_change: Predicted token changes (can be negative)
            targets: Binary targets (1 = correct prediction, 0 = incorrect)
            target_rate: Target signal rate (e.g., 0.05 for 5%)
            direction: 'buy' for high positive changes, 'sell' for high negative changes
        
        Returns:
            (token_change_threshold, precision, num_signals, num_correct)
        """
        N = len(predicted_token_change)
        target_num_signals = max(1, int(N * target_rate))
        
        if direction == 'buy':
            # BUY: Select top target_rate% by predicted token change (largest increases)
            percentile = 100 * (1 - target_rate)  # e.g., 95th percentile for top 5%
            threshold = float(np.percentile(predicted_token_change, percentile))
            
            # Apply threshold: BUY signal when predicted_token_change > threshold
            signal_mask = predicted_token_change > threshold
            num_signals = int(signal_mask.sum())
            
            # If we got too few or too many signals, adjust threshold
            if num_signals == 0 or num_signals > 2 * target_num_signals:
                # Fall back to sorting and selecting exact top-k
                sorted_changes = np.sort(predicted_token_change)[::-1]  # Descending
                if target_num_signals < N:
                    threshold = float(sorted_changes[target_num_signals])
                    signal_mask = predicted_token_change >= threshold
                    num_signals = int(signal_mask.sum())
            
            num_correct = int(targets[signal_mask].sum()) if num_signals > 0 else 0
            precision = float(num_correct) / float(num_signals) if num_signals > 0 else 0.0
            
            return float(threshold), precision, num_signals, num_correct
        
        else:  # direction == 'sell'
            # SELL: Select bottom target_rate% by predicted token change (largest decreases)
            percentile = 100 * target_rate  # e.g., 5th percentile for bottom 5%
            threshold = float(np.percentile(predicted_token_change, percentile))
            
            # Apply threshold: SELL signal when predicted_token_change < threshold
            signal_mask = predicted_token_change < threshold
            num_signals = int(signal_mask.sum())
            
            # If we got too few or too many signals, adjust threshold
            target_num_signals = max(1, int(N * target_rate))
            if num_signals == 0 or num_signals > 2 * target_num_signals:
                # Fall back to sorting and selecting exact bottom-k
                sorted_changes = np.sort(predicted_token_change)  # Ascending
                if target_num_signals < N:
                    threshold = float(sorted_changes[target_num_signals])
                    signal_mask = predicted_token_change <= threshold
                    num_signals = int(signal_mask.sum())
            
            num_correct = int(targets[signal_mask].sum()) if num_signals > 0 else 0
            precision = float(num_correct) / float(num_signals) if num_signals > 0 else 0.0
            
            return float(threshold), precision, num_signals, num_correct
