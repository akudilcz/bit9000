"""Step 6: Train - Train transformer on token sequences

Philosophy: Simple supervised learning
- Loss: Multi-step cross-entropy (sum over 8 hours)
- Metric: Per-step accuracy
- Early stopping on validation loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List
from tqdm import tqdm
import math

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
        """
        Train model on sequences
        
        Args:
            sequences_artifact: SequencesArtifact from step_05_sequences
            
        Returns:
            TrainedModelArtifact
        """
        logger.info("="*70)
        logger.info("STEP 6: TRAIN - Training token predictor")
        logger.info("="*70)
        
        # Load training config
        train_config = self.config['training']
        device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        epochs = train_config.get('epochs', 100)
        batch_size = train_config.get('batch_size', 128)
        learning_rate = train_config.get('learning_rate', 0.0001)
        patience = train_config['early_stopping'].get('patience', 10)
        
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
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # Initialize model
        logger.info("\n[2/4] Initializing model...")
        model = create_model(self.config)
        model = model.to(device)
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()  # Multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Check if using binary classification
        binary_classification = self.config['model'].get('binary_classification', False)
        multi_horizon_enabled = self.config['model'].get('multi_horizon_enabled', True)
        if binary_classification:
            pos_weight = self.config['training'].get('pos_weight', 5.7)
            # For binary classification with 2-class output, use weighted CrossEntropyLoss
            # Create class weights: [1.0, pos_weight] to upweight class 1 (BUY)
            class_weights = torch.tensor([1.0, pos_weight], device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"  Using binary classification with class weights=[1.0, {pos_weight}], multi_horizon={multi_horizon_enabled}")
        else:
            logger.info(f"  Using multi-class classification ({self.config['model'].get('num_classes', 256)} classes)")
        
        # Learning rate scheduler with warmup
        warmup_epochs = train_config.get('warmup_epochs', 5)
        warmup_start_lr = train_config.get('warmup_start_lr', 1e-6)
        
        # Create warmup scheduler followed by cosine annealing
        def warmup_lr(epoch):
            if epoch < warmup_epochs:
                # Linear warmup from warmup_start_lr to learning_rate over warmup_epochs
                return warmup_start_lr + (learning_rate - warmup_start_lr) * (epoch / warmup_epochs)
            else:
                # Cosine annealing after warmup
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
        if binary_classification:
            history['buy_precision'] = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_result = self._validate_epoch(
                model, val_loader, criterion, device
            )

            if binary_classification:
                val_loss, val_acc, buy_precision = val_result
            else:
                val_loss, val_acc = val_result
                buy_precision = None
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if binary_classification:
                history['buy_precision'].append(buy_precision)

            # Log progress
            log_msg = (
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            if binary_classification:
                log_msg += f", buy_precision={buy_precision:.3f}"
            logger.info(log_msg)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
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
        block_dir = self.artifact_io.get_block_dir("step_06_train", clean=True)
        
        # Save model
        model_path = block_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'epoch': len(history['val_loss']),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc
        }, model_path)
        logger.info(f"  Saved model: {model_path}")
        
        # Save history
        history_path = block_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"  Saved history: {history_path}")
        
        # Create artifact
        artifact = TrainedModelArtifact(
            model_path=model_path,
            history_path=history_path,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            total_epochs=len(history['val_loss']),
            metadata=self.create_metadata(
                upstream_inputs={
                    "train_sequences": str(sequences_artifact.train_X_path),
                    "val_sequences": str(sequences_artifact.val_X_path)
                }
            )
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_06_train",
            artifact_name="train_artifact"
        )
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Best val loss: {best_val_loss:.6f}")
        logger.info(f"  Best val acc: {best_val_acc:.6f}")
        logger.info(f"  Total epochs: {artifact.total_epochs}")
        logger.info("="*70 + "\n")
        
        return artifact
    
    def _train_epoch(self, model, data_loader, criterion, optimizer, device):
        """Train for one epoch with multi-horizon prediction support"""
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Check if model is multi-horizon (V4) and binary classification
        is_multi_horizon = hasattr(model, 'horizon_heads')
        binary_classification = self.config['model'].get('binary_classification', False)
        multi_horizon_enabled = self.config['model'].get('multi_horizon_enabled', True)
        
        # Create progress bar
        pbar = tqdm(data_loader, desc='Training', leave=False, ncols=100)
        
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Add Gaussian noise to input during training
            if model.training:
                noise_scale = self.config['training'].get('gaussian_noise', 0.0)
                if noise_scale > 0:
                    vocab_size = self.config['model']['vocab_size']
                    X_float = X_batch.float()
                    noise = torch.randn_like(X_float) * noise_scale
                    X_noisy = X_float + noise
                    X_batch_noisy = torch.clamp(torch.round(X_noisy), 0, vocab_size - 1).long()
                else:
                    X_batch_noisy = X_batch
            else:
                X_batch_noisy = X_batch
            
            optimizer.zero_grad()
            outputs = model(X_batch_noisy)
            
            # Handle multi-horizon (V4) vs single horizon (V1/V2/V3)
            if is_multi_horizon and isinstance(outputs, dict) and 'horizon_1h' in outputs and multi_horizon_enabled:
                # Multi-horizon V4: weight by horizon difficulty
                # Shorter horizons are typically harder to predict
                horizon_weights = [1.0, 0.8, 0.6, 0.4]  # 1h: hardest, 8h: easiest
                loss = 0
                all_correct = 0
                all_samples = 0
                
                # y_batch shape: (B, 4) for 4 horizons
                for idx, horizon in enumerate(['horizon_1h', 'horizon_2h', 'horizon_4h', 'horizon_8h']):
                    logits = outputs[horizon]['logits']  # (B, num_classes or 2)
                    y_horizon = y_batch[:, idx]  # (B,)
                    
                    # Compute loss for this horizon with weight
                    if binary_classification:
                        # Binary classification with 2 classes: use CrossEntropyLoss
                        # logits shape: (B, 2), y_horizon shape: (B,)
                        horizon_loss = criterion(logits, y_horizon.long()) * horizon_weights[idx]
                        predictions = torch.argmax(logits, dim=-1)
                    else:
                        horizon_loss = criterion(logits, y_horizon) * horizon_weights[idx]
                        predictions = torch.argmax(logits, dim=-1)
                    
                    loss += horizon_loss
                    
                    # Metrics
                    all_correct += (predictions == y_horizon).sum().item()
                    all_samples += y_horizon.numel()
                
                # Normalize by sum of weights (not 4.0)
                loss = loss / sum(horizon_weights)
                batch_acc = all_correct / all_samples
            elif is_multi_horizon and isinstance(outputs, dict) and 'horizon_1h' in outputs and not multi_horizon_enabled:
                # Single-horizon V4: only 1h head
                logits = outputs['horizon_1h']['logits']  # (B, 2 or num_classes)
                y_target = y_batch if y_batch.dim() == 1 else y_batch.squeeze(-1)
                loss = criterion(logits, y_target.long())
                predictions = torch.argmax(logits, dim=-1)
                batch_correct = (predictions == y_target).sum().item()
                batch_samples = y_target.numel()
                batch_acc = batch_correct / batch_samples
                total_correct += batch_correct
                total_samples += batch_samples
                
            elif isinstance(outputs, dict):
                # Single horizon V3
                logits = outputs['logits']
                logits_flat = logits
                if len(y_batch.shape) > 1:
                    y_flat = y_batch.squeeze(-1)
                else:
                    y_flat = y_batch
                loss = criterion(logits_flat, y_flat)
                predictions = torch.argmax(logits_flat, dim=-1)
                batch_correct = (predictions == y_flat).sum().item()
                batch_samples = y_flat.numel()
                batch_acc = batch_correct / batch_samples
                total_correct += batch_correct
                total_samples += batch_samples
            else:
                # V1/V2
                logits = outputs
                if len(logits.shape) == 3:
                    B, T, C = logits.shape
                    logits_flat = logits.view(B * T, C)
                    y_flat = y_batch.view(B * T)
                else:
                    logits_flat = logits
                    y_flat = y_batch.squeeze(-1) if len(y_batch.shape) > 1 else y_batch
                loss = criterion(logits_flat, y_flat)
                predictions = torch.argmax(logits_flat, dim=-1)
                batch_correct = (predictions == y_flat).sum().item()
                batch_samples = y_flat.numel()
                batch_acc = batch_correct / batch_samples
                total_correct += batch_correct
                total_samples += batch_samples

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # For multi-horizon only, track metrics (single-horizon already tracked in its branch)
            if is_multi_horizon and isinstance(outputs, dict) and 'horizon_1h' in outputs and multi_horizon_enabled:
                total_correct += all_correct
                total_samples += all_samples
            
            total_loss += loss.item() * X_batch.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.4f}'})
        
        avg_loss = total_loss / len(data_loader.dataset)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, model, data_loader, criterion, device):
        """Validate for one epoch with multi-horizon support"""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Check if model is multi-horizon (V4) and binary classification
        is_multi_horizon = hasattr(model, 'horizon_heads')
        binary_classification = self.config['model'].get('binary_classification', False)

        # For binary classification, track BUY precision metrics
        if binary_classification:
            buy_predictions = []  # List of all BUY predictions (0 or 1)
            buy_targets = []      # Corresponding ground truth labels

        # Create progress bar
        pbar = tqdm(data_loader, desc='Validation', leave=False, ncols=100)
        
        with torch.no_grad():
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                
                # Handle multi-horizon (V4) vs single horizon (V1/V2/V3)
                if is_multi_horizon and isinstance(outputs, dict) and 'horizon_1h' in outputs and multi_horizon_enabled:
                    # Multi-horizon V4: weight by horizon difficulty
                    horizon_weights = [1.0, 0.8, 0.6, 0.4]  # 1h: hardest, 8h: easiest
                    loss = 0
                    all_correct = 0
                    all_samples = 0

                    # y_batch shape: (B, 4) for 4 horizons
                    for idx, horizon in enumerate(['horizon_1h', 'horizon_2h', 'horizon_4h', 'horizon_8h']):
                        logits = outputs[horizon]['logits']
                        y_horizon = y_batch[:, idx]

                        if binary_classification:
                            # Binary classification with 2 classes
                            horizon_loss = criterion(logits, y_horizon.long()) * horizon_weights[idx]
                            predictions = torch.argmax(logits, dim=-1)
                        else:
                            horizon_loss = criterion(logits, y_horizon) * horizon_weights[idx]
                            predictions = torch.argmax(logits, dim=-1)

                        loss += horizon_loss
                        all_correct += (predictions == y_horizon).sum().item()
                        all_samples += y_horizon.numel()

                    loss = loss / sum(horizon_weights)
                    batch_acc = all_correct / all_samples
                    total_correct += all_correct
                    total_samples += all_samples

                elif is_multi_horizon and isinstance(outputs, dict) and 'horizon_1h' in outputs and not multi_horizon_enabled:
                    # Single-horizon V4: only 1h head
                    logits = outputs['horizon_1h']['logits']
                    y_target = y_batch if y_batch.dim() == 1 else y_batch.squeeze(-1)
                    loss = criterion(logits, y_target.long())
                    predictions = torch.argmax(logits, dim=-1)
                    batch_correct = (predictions == y_target).sum().item()
                    batch_samples = y_target.numel()
                    batch_acc = batch_correct / batch_samples
                    total_correct += batch_correct
                    total_samples += batch_samples

                    # For binary classification, compute BUY precision using threshold
                    if binary_classification:
                        # Convert logits to probabilities for class 1 (BUY)
                        prob_dist = torch.softmax(logits, dim=-1)
                        buy_probs = prob_dist[:, 1]
                        threshold = self.config['inference'].get('single_threshold', 0.70)
                        buy_signals = (buy_probs > threshold).long()

                        # Collect for precision calculation
                        buy_predictions.extend(buy_signals.detach().cpu().numpy())
                        buy_targets.extend(y_target.detach().cpu().numpy())
                    
                elif isinstance(outputs, dict):
                    # Single horizon V3
                    logits = outputs['logits']
                    logits_flat = logits
                    if len(y_batch.shape) > 1:
                        y_flat = y_batch.squeeze(-1)
                    else:
                        y_flat = y_batch
                    loss = criterion(logits_flat, y_flat)
                    predictions = torch.argmax(logits_flat, dim=-1)
                    batch_correct = (predictions == y_flat).sum().item()
                    batch_samples = y_flat.numel()
                    batch_acc = batch_correct / batch_samples
                    total_correct += batch_correct
                    total_samples += batch_samples
                else:
                    # V1/V2
                    logits = outputs
                    if len(logits.shape) == 3:
                        B, T, C = logits.shape
                        logits_flat = logits.view(B * T, C)
                        y_flat = y_batch.view(B * T)
                    else:
                        logits_flat = logits
                        y_flat = y_batch.squeeze(-1) if len(y_batch.shape) > 1 else y_batch
                    loss = criterion(logits_flat, y_flat)
                    predictions = torch.argmax(logits_flat, dim=-1)
                    batch_correct = (predictions == y_flat).sum().item()
                    batch_samples = y_flat.numel()
                    batch_acc = batch_correct / batch_samples
                    total_correct += batch_correct
                    total_samples += batch_samples
                
                total_loss += loss.item() * X_batch.size(0)

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.4f}'})

        avg_loss = total_loss / len(data_loader.dataset)
        avg_acc = total_correct / total_samples

        # Calculate BUY precision for binary classification
        if binary_classification:
            import numpy as np
            buy_predictions = np.array(buy_predictions)
            buy_targets = np.array(buy_targets)

            # BUY precision: TP / (TP + FP) = correct BUY predictions / total BUY predictions
            buy_pred_mask = buy_predictions == 1
            if buy_pred_mask.sum() > 0:  # Avoid division by zero
                buy_precision = (buy_targets[buy_pred_mask] == 1).mean()
            else:
                buy_precision = 0.0

            buy_signal_rate = buy_pred_mask.mean()

            logger.info(f"  BUY Precision: {buy_precision:.3f} ({buy_pred_mask.sum()}/{len(buy_predictions)} signals = {buy_signal_rate:.3f} rate)")

            return avg_loss, avg_acc, buy_precision

        return avg_loss, avg_acc
    
    def inference_vote(self, outputs: Dict, voting_strategy: str = 'weighted', 
                      threshold: float = 0.70, min_agree: int = 3) -> torch.Tensor:
        """
        Multi-horizon voting for binary BUY signal generation
        
        Combines predictions from 4 horizons (1h, 2h, 4h, 8h) using different voting strategies
        
        Args:
            outputs: dict with 'horizon_1h', 'horizon_2h', 'horizon_4h', 'horizon_8h'
                    Each contains logits (B, 2) for binary classification
            voting_strategy: 'strict' (AND), 'majority', 'weighted', 'confidence_and'
            threshold: Confidence threshold for weighted/confidence_and strategies
            min_agree: Minimum horizons agreeing for majority/confidence_and
            
        Returns:
            BUY_signals: (B,) binary predictions (0 or 1)
        """
        # Get probabilities for class 1 (BUY) for each horizon
        probs = {}
        preds = {}
        for horizon in ['horizon_1h', 'horizon_2h', 'horizon_4h', 'horizon_8h']:
            logits = outputs[horizon]['logits']  # (B, 2) for binary classification
            # Use softmax for multi-class (even if 2 classes)
            prob_dist = torch.softmax(logits, dim=-1)
            probs[horizon] = prob_dist[:, 1]  # Probability of class 1 (BUY)
            preds[horizon] = (probs[horizon] > 0.5).float()
        
        if voting_strategy == 'strict':
            # ALL horizons must predict BUY
            result = (preds['horizon_1h'] * preds['horizon_2h'] * 
                     preds['horizon_4h'] * preds['horizon_8h'])
        
        elif voting_strategy == 'majority':
            # 3+ out of 4 must agree on BUY
            vote_sum = (preds['horizon_1h'] + preds['horizon_2h'] + 
                       preds['horizon_4h'] + preds['horizon_8h'])
            result = (vote_sum >= min_agree).float()
        
        elif voting_strategy == 'weighted':
            # Weighted average by horizon importance
            weights = torch.tensor([1.0, 0.8, 0.6, 0.4], device=probs['horizon_1h'].device)
            avg_prob = (weights[0] * probs['horizon_1h'] + 
                       weights[1] * probs['horizon_2h'] +
                       weights[2] * probs['horizon_4h'] +
                       weights[3] * probs['horizon_8h']) / weights.sum()
            result = (avg_prob > threshold).float()
        
        elif voting_strategy == 'confidence_and':
            # Majority voting + high average confidence
            vote_sum = (preds['horizon_1h'] + preds['horizon_2h'] + 
                       preds['horizon_4h'] + preds['horizon_8h'])
            avg_prob = (probs['horizon_1h'] + probs['horizon_2h'] + 
                       probs['horizon_4h'] + probs['horizon_8h']) / 4.0
            result = ((vote_sum >= min_agree) & (avg_prob > threshold)).float()
        
        else:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")
        
        return result

