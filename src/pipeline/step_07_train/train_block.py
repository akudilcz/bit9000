"""Step 6: Train - Train transformer on token sequences

Token prediction with ordinal cross-entropy loss
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
        num_classes = self.config['model']['num_classes']
        model_version = self.config['model'].get('version', 'v6')
        logger.info(f"STEP 6: TRAIN - Training {model_version} {num_classes}-token next-token prediction")
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
        
        # Load trading labels
        train_trading_path = sequences_artifact.train_X_path.parent / "train_trading.pt"
        val_trading_path = sequences_artifact.val_X_path.parent / "val_trading.pt"
        train_trading = torch.load(train_trading_path)
        val_trading = torch.load(val_trading_path)
        
        logger.info(f"  Train: X={train_X.shape}, y={train_y.shape}, trading={train_trading.shape}")
        logger.info(f"  Val: X={val_X.shape}, y={val_y.shape}, trading={val_trading.shape}")
        
        # Create data loaders (include trading labels)
        train_dataset = TensorDataset(train_X, train_y, train_trading)
        val_dataset = TensorDataset(val_X, val_y, val_trading)
        
        num_workers = train_config.get('num_workers', 0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # Initialize model
        logger.info("\n[2/4] Initializing model...")
        model = create_model(self.config)
        model = model.to(device)
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        num_classes = self.config['model']['num_classes']
        logger.info(f"  Using {num_classes}-class token prediction")
        
        # Loss and optimizer
        # For v6 decoder: use standard cross-entropy for next-token prediction
        model_version = self.config['model'].get('version', 'v6')
        if model_version != 'v6':
            raise ValueError(f"Only v6 model is supported, got {model_version}")
        
        # Standard cross-entropy for GPT-style next-token prediction
        label_smoothing = train_config.get('label_smoothing', 0.0)
        criterion_tokens = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info(f"  Using Cross-Entropy Loss for next-token (label_smoothing={label_smoothing})")
        
        # Trading signal loss with class weights (weight HOLD less since it's ~90%)
        trading_class_weights = train_config.get('trading_class_weights', [1.0, 0.1, 1.0])
        trading_weights = torch.tensor(trading_class_weights, device=device)
        criterion_trading = nn.CrossEntropyLoss(weight=trading_weights)
        logger.info(f"  Using Weighted Cross-Entropy Loss for trading (weights={trading_weights.tolist()})")
        
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
        
        # Pass criterion as tuple for v6 (for easier unpacking in train/val functions)
        criterion_tuple = (criterion_tokens, criterion_trading)
        
        # Training loop
        logger.info("\n[3/4] Training...")
        history = {
            'train_loss': [],
            'train_token_acc': [],
            'train_trading_acc': [],
            'val_loss': [],
            'val_token_acc': [],
            'val_trading_acc': []
        }
        
        best_val_loss = float('inf')
        best_calibrated_threshold_buy = 0.5
        best_calibrated_threshold_sell = 0.5
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_token_acc, train_trading_acc = self._train_epoch(model, train_loader, criterion_tuple, optimizer, device)
            
            # Validate
            val_result = self._validate_epoch(model, val_loader, criterion_tuple, device, model_version)
            val_loss = val_result['loss']
            val_token_acc = val_result['token_acc']
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_token_acc'].append(train_token_acc)
            history['train_trading_acc'].append(train_trading_acc)
            history['val_loss'].append(val_loss)
            history['val_token_acc'].append(val_token_acc)
            
            val_trading_acc = val_result.get('trading_acc', 0)
            history['val_trading_acc'].append(val_trading_acc)
            
            # Record epoch-level diagnostics
            epoch_diag = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_token_acc': train_token_acc,
                'train_trading_acc': train_trading_acc,
                'val_loss': val_loss,
                'val_token_acc': val_token_acc,
                'val_trading_acc': val_trading_acc,
                'val_trading_acc_buy': val_result.get('trading_acc_buy', 0),
                'val_trading_acc_hold': val_result.get('trading_acc_hold', 0),
                'val_trading_acc_sell': val_result.get('trading_acc_sell', 0),
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            self.diagnostics['epochs'].append(epoch_diag)
            
            # Log progress
            if model_version == 'v6':
                logger.info(
                    f"  Epoch {epoch+1:3d}/{epochs}: "
                    f"train_loss={train_loss:.4f}, tok_acc={train_token_acc:.3f}, trd_acc={train_trading_acc:.3f}, "
                    f"val_loss={val_loss:.4f}, tok_acc={val_token_acc:.3f}, trd_acc={val_trading_acc:.3f}"
                )
                # Log detailed trading signal metrics with confusion matrix style counts
                confusion_matrix = val_result.get('confusion_matrix', {})
                logger.info(
                    f"    Trading signals:"
                )
                for signal in ['BUY', 'SELL', 'HOLD']:
                    if signal in confusion_matrix:
                        pred = confusion_matrix[signal]['predicted']
                        true = confusion_matrix[signal]['true']
                        acc = val_result.get(f'trading_acc_{signal.lower()}', 0)
                        prec = val_result.get(f'trading_prec_{signal.lower()}', 0)
                        logger.info(
                            f"      {signal:4s}: predicted={pred:3d}/{true:3d} true | "
                            f"acc={acc:.3f} prec={prec:.3f}"
                        )
            else:
                logger.info(
                    f"  Epoch {epoch+1:3d}/{epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_token_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_token_acc:.4f}"
                )
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_token_acc
                best_model_state = model.state_dict().copy()
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
    
    def _train_epoch(self, model, data_loader, criterion, optimizer, device) -> Tuple[float, float, float]:
        """Train for one epoch with multi-task learning"""
        model.train()
        total_loss = 0
        total_token_correct = 0
        total_token_samples = 0
        total_trading_correct = 0
        total_trading_samples = 0
        
        pbar = tqdm(data_loader, desc='Training', leave=False, ncols=100)
        
        model_version = self.config['model'].get('version', 'v6')
        
        # Get criterion functions (for v6, criterion is actually a tuple)
        if model_version == 'v6':
            criterion_tokens, criterion_trading = criterion
        else:
            criterion_single = criterion
        
        # Timing profiling
        import time
        batch_times = []
        forward_times = []
        backward_times = []

        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            
            if model_version == 'v6':
                X_batch, y_batch, trading_batch = batch
            else:
                X_batch, y_batch = batch
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if model_version == 'v6':
                trading_batch = trading_batch.to(device)
            
            data_time = time.time() - batch_start
            
            optimizer.zero_grad()
            
            forward_start = time.time()
            
            if model_version == 'v6':
                # V6 decoder: multi-task learning
                outputs = model(X_batch, return_trading=True)
                logits = outputs['logits']  # (B, L-1, vocab_size)
                trading_logits = outputs['trading']  # (B, 3)
                
                # Next-token prediction loss
                token_loss = criterion_tokens(
                    logits.reshape(-1, logits.size(-1)),
                    y_batch.reshape(-1)
                )
                
                # Trading signal loss
                trading_loss = criterion_trading(trading_logits, trading_batch)
                
                # Combined loss (weight trading loss lower since next-token is primary)
                trading_weight = self.config['training'].get('trading_loss_weight', 0.3)
                loss = token_loss + trading_weight * trading_loss
            else:
                # For v5, use a single criterion
                outputs = model(X_batch)
                logits = outputs['logits']
                loss = criterion_single(logits, y_batch)
            
            forward_time = time.time() - forward_start
            
            # Accuracies
            token_predictions = torch.argmax(logits, dim=-1)
            batch_token_correct = (token_predictions == y_batch).sum().item()
            batch_token_samples = y_batch.numel()
            
            trading_predictions = torch.argmax(trading_logits, dim=-1)
            batch_trading_correct = (trading_predictions == trading_batch).sum().item()
            batch_trading_samples = trading_batch.numel()
            
            total_token_correct += batch_token_correct
            total_token_samples += batch_token_samples
            total_trading_correct += batch_trading_correct
            total_trading_samples += batch_trading_samples
            
            # Collect diagnostics every 50 batches
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    diag = {
                        'epoch': len(self.diagnostics['epochs']),
                        'batch': batch_idx,
                        'loss': loss.item(),
                        'token_acc': batch_token_correct/batch_token_samples,
                        'trading_acc': batch_trading_correct/batch_trading_samples,
                        'token_loss': token_loss.item(),
                        'trading_loss': trading_loss.item(),
                        'data_time': data_time,
                        'forward_time': forward_time
                    }
                    self.diagnostics['batches'].append(diag)
            
            # Backward pass with gradient clipping
            loss.backward()
            backward_start = time.time()
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            backward_time = time.time() - backward_start
            
            # Track times
            batch_times.append(time.time() - batch_start)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            
            # Log profiling every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_batch = sum(batch_times[-100:]) / len(batch_times[-100:])
                avg_forward = sum(forward_times[-100:]) / len(forward_times[-100:])
                avg_backward = sum(backward_times[-100:]) / len(backward_times[-100:])
                logger.info(
                    f"    Profiling (batches {batch_idx-99}-{batch_idx}): "
                    f"batch={avg_batch*1000:.1f}ms, forward={avg_forward*1000:.1f}ms, "
                    f"backward={avg_backward*1000:.1f}ms"
                )
            
            # Accumulate metrics
            total_loss += loss.item() * X_batch.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'tok_acc': f'{batch_token_correct/batch_token_samples:.3f}',
                'trd_acc': f'{batch_trading_correct/batch_trading_samples:.3f}'
            })
        
        avg_loss = total_loss / len(data_loader.dataset)
        avg_token_acc = total_token_correct / total_token_samples if total_token_samples > 0 else 0
        avg_trading_acc = total_trading_correct / total_trading_samples
        return avg_loss, avg_token_acc, avg_trading_acc
    
    def _validate_epoch(self, model, data_loader, criterion, device, model_version='v6') -> dict:
        """
        Validate for one epoch with enhanced metrics tracking
        Returns dict with metrics for both v5 and v6 models
        """
        model.eval()
        total_loss = 0
        total_token_correct = 0
        total_token_samples = 0
        total_trading_correct = 0
        total_trading_samples = 0
        
        # Get criterion functions
        criterion_tokens, criterion_trading = criterion
        
        # For trading signal confusion matrix
        all_trading_preds = []
        all_trading_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                X_batch, y_batch, trading_batch = batch
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                trading_batch = trading_batch.to(device)
                
                # V6 decoder: multi-task learning
                outputs = model(X_batch, return_trading=True)
                logits = outputs['logits']
                trading_logits = outputs['trading']
                
                # Losses
                token_loss = criterion_tokens(
                    logits.reshape(-1, logits.size(-1)),
                    y_batch.reshape(-1)
                )
                trading_loss = criterion_trading(trading_logits, trading_batch)
                trading_weight = self.config['training'].get('trading_loss_weight', 0.3)
                loss = token_loss + trading_weight * trading_loss
                
                # Token accuracy
                token_predictions = torch.argmax(logits, dim=-1)
                batch_token_correct = (token_predictions == y_batch).sum().item()
                batch_token_samples = y_batch.numel()
                
                # Trading accuracy
                trading_predictions = torch.argmax(trading_logits, dim=-1)
                batch_trading_correct = (trading_predictions == trading_batch).sum().item()
                batch_trading_samples = trading_batch.numel()
                
                all_trading_preds.extend(trading_predictions.cpu().numpy())
                all_trading_targets.extend(trading_batch.cpu().numpy())
                
                total_token_correct += batch_token_correct
                total_token_samples += batch_token_samples
                total_trading_correct += batch_trading_correct
                total_trading_samples += batch_trading_samples
                
                total_loss += loss.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(data_loader.dataset)
        avg_token_acc = total_token_correct / total_token_samples if total_token_samples > 0 else 0
        avg_trading_acc = total_trading_correct / total_trading_samples
        
        result = {
            'loss': avg_loss,
            'token_acc': avg_token_acc,
            'trading_acc': avg_trading_acc,
        }
        
        # Compute per-class trading accuracy
        all_trading_preds = np.array(all_trading_preds)
        all_trading_targets = np.array(all_trading_targets)
        
        for label, name in [(0, 'BUY'), (1, 'HOLD'), (2, 'SELL')]:
            mask = all_trading_targets == label
            if mask.sum() > 0:
                class_acc = (all_trading_preds[mask] == label).sum() / mask.sum()
                result[f'trading_acc_{name.lower()}'] = class_acc
        
        if model_version == 'v6':
            avg_trading_acc = total_trading_correct / total_trading_samples
            result['trading_acc'] = avg_trading_acc
            
            # Compute per-class trading accuracy and precision/recall
            all_trading_preds = np.array(all_trading_preds)
            all_trading_targets = np.array(all_trading_targets)
            
            # Confusion matrix style counts
            confusion_matrix = {}
            for label, name in [(0, 'BUY'), (1, 'HOLD'), (2, 'SELL')]:
                mask = all_trading_targets == label
                if mask.sum() > 0:
                    # True instances of this class
                    true_count = mask.sum()
                    # Correct predictions of this class
                    correct_count = (all_trading_preds[mask] == label).sum()
                    confusion_matrix[name] = {
                        'predicted': correct_count,
                        'true': true_count
                    }
                    
                    # Accuracy: correct predictions / total instances of this class
                    class_acc = correct_count / true_count
                    result[f'trading_acc_{name.lower()}'] = class_acc
                    
                    # Precision: correct predictions of this class / total predictions of this class
                    pred_mask = all_trading_preds == label
                    if pred_mask.sum() > 0:
                        precision = (all_trading_preds[pred_mask] == all_trading_targets[pred_mask]).sum() / pred_mask.sum()
                        result[f'trading_prec_{name.lower()}'] = precision
                    else:
                        result[f'trading_prec_{name.lower()}'] = 0.0
                    
                    # Recall: same as accuracy for this class
                    result[f'trading_recall_{name.lower()}'] = class_acc
                else:
                    confusion_matrix[name] = {'predicted': 0, 'true': 0}
                    result[f'trading_acc_{name.lower()}'] = 0.0
                    result[f'trading_prec_{name.lower()}'] = 0.0
                    result[f'trading_recall_{name.lower()}'] = 0.0
            
            result['confusion_matrix'] = confusion_matrix
            
            # Validation check: ensure all label counts add up
            total_true_count = sum(cm['true'] for cm in confusion_matrix.values())
            expected_total = total_trading_samples
            assert total_true_count == expected_total, \
                f"Trading label validation failed: {total_true_count} != {expected_total}"
        
        return result

