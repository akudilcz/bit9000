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

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import ArtifactMetadata
from src.model.token_predictor import SimpleTokenPredictor
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
        model = SimpleTokenPredictor(self.config)
        model = model.to(device)
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()  # Multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        logger.info("\n[3/4] Training...")
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            
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
        """Train for one epoch using teacher forcing with correct shifting"""
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Add Gaussian noise to input during training (helps escape local minima)
            # Noise std is 0.1 * input std to avoid overwhelming the signal
            if model.training:
                noise = torch.randn_like(X_batch) * 0.05  # Small noise: 5% of input scale
                X_batch_noisy = X_batch + noise
            else:
                X_batch_noisy = X_batch
            
            optimizer.zero_grad()
            # Forward pass with correct teacher forcing (model handles shifting)
            logits = model(X_batch_noisy, targets=y_batch)  # (B, vocab_size) for single-step prediction

            # Handle both old (B, T, C) and new (B, C) output shapes
            if len(logits.shape) == 3:
                # Old format: (B, T, C) - reshape for loss
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                y_flat = y_batch.view(B * T)
            else:
                # New format: (B, vocab_size) for single-step prediction
                logits_flat = logits  # (B, vocab_size)
                # Handle both (B,) and (B, 1) target shapes
                if len(y_batch.shape) > 1:
                    y_flat = y_batch.squeeze(-1)  # (B,)
                else:
                    y_flat = y_batch  # Already (B,)

            # Compute loss
            loss = criterion(logits_flat, y_flat)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            predictions = torch.argmax(logits_flat, dim=-1)
            total_loss += loss.item() * X_batch.size(0)
            total_correct += (predictions == y_flat).sum().item()
            total_samples += y_flat.numel()
        
        avg_loss = total_loss / len(data_loader.dataset)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, model, data_loader, criterion, device):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass to get logits
                logits = model(X_batch, targets=y_batch)  # (B, vocab_size) for single-step
                
                # Handle both old (B, T, C) and new (B, C) output shapes
                if len(logits.shape) == 3:
                    # Old format: (B, T, C)
                    B, T, C = logits.shape
                    logits_flat = logits.view(B * T, C)
                    y_flat = y_batch.view(B * T)
                else:
                    # New format: (B, vocab_size) for single-step prediction
                    logits_flat = logits  # (B, vocab_size)
                    # Handle both (B,) and (B, 1) target shapes
                    if len(y_batch.shape) > 1:
                        y_flat = y_batch.squeeze(-1)  # (B,)
                    else:
                        y_flat = y_batch  # Already (B,)
                
                # Compute loss
                loss = criterion(logits_flat, y_flat)
                
                # Metrics
                predictions = torch.argmax(logits_flat, dim=-1)
                total_loss += loss.item() * X_batch.size(0)
                total_correct += (predictions == y_flat).sum().item()
                total_samples += y_flat.numel()
            
            avg_loss = total_loss / len(data_loader.dataset)
            avg_acc = total_correct / total_samples
            
            return avg_loss, avg_acc

