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
from src.pipeline.schemas import ArtifactMetadata, SequencesArtifact
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
    
    def __init__(self, config: dict, artifact_io):
        super().__init__(config, artifact_io)
        self.diagnostics = {
            'batches': [],
            'epochs': [],
            'model_outputs': []
        }
    
    def run(self, sequences_artifact=None):
        """Train model on sequences"""
        logger.info("="*70)
        logger.info("STEP 6: TRAIN - Training 256-class token prediction")
        logger.info("="*70)
        
        # Load sequences artifact if not provided
        if sequences_artifact is None:
            sequences_artifact_data = self.artifact_io.read_json('artifacts/step_06_sequences/sequences_artifact.json')
            sequences_artifact = SequencesArtifact(**sequences_artifact_data)
        
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
        # Use ordinal loss to respect that tokens represent ordered returns
        from src.utils.ordinal_loss import SoftOrdinalCrossEntropyLoss
        label_smoothing = train_config.get('label_smoothing', 0.0)
        sigma = train_config.get('ordinal_sigma', 10.0)  # Controls spread of soft targets
        criterion = SoftOrdinalCrossEntropyLoss(num_classes=256, sigma=sigma, label_smoothing=label_smoothing)
        logger.info(f"  Using Soft Ordinal CE Loss (sigma={sigma}, label_smoothing={label_smoothing})")
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=train_config.get('weight_decay', 0.0)
        )
        
        # Learning rate scheduler with warmup
        warmup_epochs = min(train_config.get('warmup_epochs', 3), max(1, epochs - 1))  # Ensure warmup < total epochs
        warmup_start_lr = train_config.get('warmup_start_lr', 1e-6)
        
        def warmup_lr(epoch):
            if epoch < warmup_epochs:
                return warmup_start_lr + (learning_rate - warmup_start_lr) * (epoch / warmup_epochs)
            else:
                remaining_epochs = epochs - warmup_epochs
                if remaining_epochs <= 0:
                    return learning_rate
                progress = (epoch - warmup_epochs) / remaining_epochs
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
            
            # Record epoch-level diagnostics
            self.diagnostics['epochs'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'avg_precision': avg_precision,
                'threshold_buy': threshold_buy,
                'threshold_sell': threshold_sell,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
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
        
        # Save diagnostics for post-training analysis
        diagnostics_path = block_dir / "diagnostics.json"
        with open(diagnostics_path, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)
        logger.info(f"  Saved diagnostics: {diagnostics_path}")
        
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
        
        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
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
            
            # Collect diagnostics every 50 batches
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    self.diagnostics['batches'].append({
                        'epoch': len(self.diagnostics['epochs']),
                        'batch': batch_idx,
                        'loss': loss.item(),
                        'acc': batch_correct/batch_samples,
                        'logits_stats': {
                            'min': float(logits.min()),
                            'max': float(logits.max()),
                            'mean': float(logits.mean()),
                            'std': float(logits.std())
                        },
                        'predictions': {
                            'unique_count': len(torch.unique(predictions)),
                            'most_common': int(torch.mode(predictions)[0]),
                            'pred_std': float(predictions.float().std())
                        },
                        'targets': {
                            'unique_count': len(torch.unique(y_batch)),
                            'mean': float(y_batch.float().mean()),
                            'std': float(y_batch.float().std())
                        }
                    })
            
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
        Validate for one epoch with enhanced metrics tracking and threshold calibration
        
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
            
            # Tokens represent log returns (already changes), not absolute prices
            # Token 127 ≈ 0% return, token >127 = positive return, token <127 = negative return
            all_probs_concat = np.concatenate(all_probs, axis=0)  # (samples, 256)
            all_targets_concat = np.concatenate(all_targets, axis=0)  # (samples,)
            
            # Get predicted tokens (argmax of logits)
            predicted_tokens = np.argmax(all_probs_concat, axis=1)
            actual_tokens = all_targets_concat
            
            # Target signal rate: 5%
            target_signal_rate = 0.05
            target_count = int(len(actual_tokens) * target_signal_rate)
            neutral_token = 127  # Middle token represents ~0% return
            
            # Diagnostics: show token distribution
            logger.info(f"    Actual return tokens - min: {actual_tokens.min():.1f}, max: {actual_tokens.max():.1f}, mean: {actual_tokens.mean():.1f}, std: {actual_tokens.std():.1f}")
            logger.info(f"    Predicted return tokens - min: {predicted_tokens.min():.1f}, max: {predicted_tokens.max():.1f}, mean: {predicted_tokens.mean():.1f}, std: {predicted_tokens.std():.1f}")
            logger.info(f"    Predicted unique tokens: {len(np.unique(predicted_tokens))}")
            
            # Show sample of first 5 predictions
            logger.info(f"    Sample predictions (first 5):")
            for i in range(min(5, len(predicted_tokens))):
                pred_direction = "UP" if predicted_tokens[i] > neutral_token else ("DOWN" if predicted_tokens[i] < neutral_token else "FLAT")
                actual_direction = "UP" if actual_tokens[i] > neutral_token else ("DOWN" if actual_tokens[i] < neutral_token else "FLAT")
                logger.info(f"      [{i}] pred={predicted_tokens[i]:.0f} ({pred_direction}), actual={actual_tokens[i]:.0f} ({actual_direction})")
            
            logger.info(f"    Calibrating thresholds for ~{target_signal_rate*100:.0f}% signal rate ({target_count} samples)")
            
            # ========== CALIBRATE BUY THRESHOLD ==========
            # Strategy: Find token threshold where predictions > threshold represent strong positive returns
            # Select top 5% of predicted tokens as BUY signals
            buy_percentile = 95  # Top 5% most positive predictions
            best_buy_threshold = np.percentile(predicted_tokens, buy_percentile)
            
            # Calculate precision: among top 5% predicted positive returns, how many were ACTUALLY positive?
            predicted_buy_mask = predicted_tokens > best_buy_threshold
            buy_signal_count = predicted_buy_mask.sum()
            buy_signal_rate = buy_signal_count / len(predicted_tokens)
            
            if buy_signal_count > 0:
                # Actual positives: actual return token > neutral (127)
                actual_positives = (actual_tokens[predicted_buy_mask] > neutral_token).sum()
                best_buy_precision = actual_positives / buy_signal_count
            else:
                actual_positives = 0
                best_buy_precision = 0.0
            
            logger.info(f"    BUY: token_threshold={best_buy_threshold:.1f}, signals={buy_signal_count} ({buy_signal_rate*100:.2f}%), correct={actual_positives}, precision={best_buy_precision*100:.2f}%")
            
            # ========== CALIBRATE SELL THRESHOLD ==========
            # Strategy: Find token threshold where predictions < threshold represent strong negative returns
            # Select bottom 5% of predicted tokens as SELL signals
            sell_percentile = 5  # Bottom 5% most negative predictions
            best_sell_threshold = np.percentile(predicted_tokens, sell_percentile)
            
            # Calculate precision: among bottom 5% predicted negative returns, how many were ACTUALLY negative?
            predicted_sell_mask = predicted_tokens < best_sell_threshold
            sell_signal_count = predicted_sell_mask.sum()
            sell_signal_rate = sell_signal_count / len(predicted_tokens)
            
            if sell_signal_count > 0:
                # Actual negatives: actual return token < neutral (127)
                actual_negatives = (actual_tokens[predicted_sell_mask] < neutral_token).sum()
                best_sell_precision = actual_negatives / sell_signal_count
            else:
                actual_negatives = 0
                best_sell_precision = 0.0
            
            logger.info(f"    SELL: token_threshold={best_sell_threshold:.1f}, signals={sell_signal_count} ({sell_signal_rate*100:.2f}%), correct={actual_negatives}, precision={best_sell_precision*100:.2f}%")
            
            return avg_loss, avg_acc, max(best_buy_precision, best_sell_precision), best_buy_threshold, best_sell_threshold
