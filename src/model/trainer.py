import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict
from torch.utils.data import TensorDataset, DataLoader, Subset

# Mock logger for demonstration
class MockLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
logger = MockLogger()

# Optimize PyTorch for faster training
torch.set_float32_matmul_precision('high')  # Use high precision (faster than medium)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels
    torch.backends.cudnn.enabled = True


# --- Custom Callback (Simplified/Optional) ---
# Keeping this only if console logging is mandatory, but removing redundant metric extraction.
class ConsoleLogger(Callback):
    """Custom callback to log validation scores after each epoch"""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        # Safely extract values from metrics logged at epoch end
        def safe_item(key):
            value = metrics.get(key)
            if value is not None:
                if hasattr(value, 'item'):
                    return value.item()
                return float(value)
            return float('nan')
        
        if trainer.current_epoch % 1 == 0:
            logger.info(f"Epoch {trainer.current_epoch}: Train Loss: {safe_item('train_loss'):.6f}, Train Acc: {safe_item('train_acc'):.6f} | Val Loss: {safe_item('val_loss'):.6f}, Val Acc: {safe_item('val_acc'):.6f}")


class BestMetricsTracker(Callback):
    """Track best validation metrics during training"""
    
    def __init__(self):
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        def safe_item(key):
            value = metrics.get(key)
            if value is not None:
                if hasattr(value, 'item'):
                    return value.item()
                return float(value)
            return float('nan')
        
        val_loss = safe_item('val_loss')
        val_acc = safe_item('val_acc')
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = trainer.current_epoch
            # Avoid non-ASCII symbols in Windows console
            logger.info(f"  NEW BEST: Val Loss: {self.best_val_loss:.6f}, Val Acc: {self.best_val_acc:.6f} (epoch {self.best_epoch})")


class WarmupCallback(Callback):
    """Custom callback to handle learning rate warmup"""
    
    def __init__(self, warmup_epochs: int, start_lr: float, target_lr: float):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch_count = 0
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Adjust learning rate at the start of each epoch during warmup"""
        if self.current_epoch_count < self.warmup_epochs:
            # Linear warmup: gradually increase LR from start_lr to target_lr
            progress = self.current_epoch_count / self.warmup_epochs
            current_lr = self.start_lr + (self.target_lr - self.start_lr) * progress
            
            # Update all optimizer param groups
            for param_group in trainer.optimizers[0].param_groups:
                param_group['lr'] = current_lr
            
            logger.info(f"Warmup epoch {self.current_epoch_count + 1}/{self.warmup_epochs}: LR = {current_lr:.2e}")
        
        self.current_epoch_count += 1


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss that penalizes based on distance from true class.
    
    For a 256-class problem, predicting 56 instead of 57 should be much better
    than predicting 10 or 200. This loss treats tokens as having an inherent order.
    
    Philosophy: Convert ordinal classification to cumulative probabilities.
    For K classes (0 to K-1), we learn K-1 cumulative probabilities:
    P(y ≤ 0), P(y ≤ 1), ..., P(y ≤ K-2)
    
    Then: P(y = k) = P(y ≤ k) - P(y ≤ k-1)
    """
    
    def __init__(self, num_classes: int, margin: float = 0.5):
        """
        Args:
            num_classes: Number of classes (256)
            margin: Soft margin for ordinal constraint (default: 0.5)
        """
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw model outputs
            targets: (batch,) target class indices (0 to num_classes-1)
        
        Returns:
            Scalar loss
        """
        batch_size = targets.size(0)
        device = logits.device
        
        # Convert logits to probabilities using log_sigmoid for numerical stability
        log_probs = torch.nn.functional.logsigmoid(logits)  # log(sigmoid(x))
        
        # Loss accumulation
        loss = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            
            # For classes before true_class: want P(y <= j) to be low
            # Minimize log(sigmoid(logit)) or maximize -log(sigmoid(logit))
            if true_class > 0:
                # These should be low probabilities, so penalize positive logits
                loss[i] += torch.sum(torch.clamp(logits[i, :true_class], min=-10.0))
            
            # For classes at or after true_class: want P(y <= j) to be high
            # Minimize -log(sigmoid(logit)) or maximize log(sigmoid(logit))
            if true_class < self.num_classes - 1:
                # These should be high probabilities, so penalize negative logits
                loss[i] += torch.sum(torch.clamp(-logits[i, true_class:], min=-10.0))
        
        return loss.mean()


class SmoothOrdinalLoss(nn.Module):
    """
    Smooth ordinal loss using soft labels based on distance.
    
    Creates soft target distribution where nearby classes get high probability
    and distant classes get low probability. Much simpler than ordinal regression.
    
    Example: If true class is 100, then:
    - Class 100 gets label 1.0
    - Class 99, 101 get label ~0.8
    - Class 98, 102 get label ~0.6
    - etc.
    """
    
    def __init__(self, num_classes: int, sigma: float = 5.0):
        """
        Args:
            num_classes: Number of classes (256)
            sigma: Standard deviation of Gaussian smoothing (wider = more smoothing)
                   Default 5.0 means distance of 5 tokens gets ~0.6 probability
        """
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw model outputs
            targets: (batch,) target class indices
        
        Returns:
            Scalar loss
        """
        batch_size = targets.size(0)
        device = logits.device
        
        # Create soft label distribution for each sample
        soft_targets = torch.zeros_like(logits)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            # Gaussian centered at true_class
            distances = torch.arange(self.num_classes, device=device, dtype=torch.float32) - true_class
            # Square the distance for stronger penalization of distant predictions
            soft_targets[i] = torch.exp(-distances.pow(4) / (2 * self.sigma ** 2))
            # Normalize
            soft_targets[i] = soft_targets[i] / soft_targets[i].sum()
        
        # Compute KL divergence between soft targets and model predictions
        log_probs = torch.log_softmax(logits, dim=1)
        loss = self.criterion(log_probs, soft_targets)
        
        return loss


class DistanceWeightedCrossEntropy(nn.Module):
    """
    Standard cross-entropy but penalize based on distance from true class.
    
    Loss = -log(p_true) * (1 + alpha * distance)
    
    This is simpler than ordinal regression but still rewards nearby predictions.
    """
    
    def __init__(self, num_classes: int, alpha: float = 0.01):
        """
        Args:
            num_classes: Number of classes (256)
            alpha: Distance weight (higher = more penalty for distance)
                   Default 0.01 means distance of 10 gets ~10% extra penalty
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,)
        
        Returns:
            Scalar loss
        """
        # Standard cross entropy
        log_probs = torch.log_softmax(logits, dim=1)
        ce_loss = torch.nn.functional.nll_loss(log_probs, targets, reduction='none')
        
        # Get predicted classes
        preds = torch.argmax(logits, dim=1)
        
        # Compute distance-based weights
        distances = torch.abs(preds.float() - targets.float())
        distance_weights = 1.0 + self.alpha * distances
        
        # Apply distance weights
        weighted_loss = ce_loss * distance_weights
        
        return weighted_loss.mean()


class CryptoLightningModule(LightningModule):
    """PyTorch Lightning module for crypto prediction"""
    
    def __init__(self, model: nn.Module, config: Dict):
        super().__init__()
        self.model = model
        self.config = config
        self.training_config = config['training']
        
        self.num_classes = config['model']['num_classes']
        
        # Determine loss function type
        loss_type = self.training_config.get('loss_type', 'cross_entropy')
        label_smoothing = self.training_config.get('label_smoothing', 0.0)
        
        # Initialize appropriate loss function
        if loss_type == 'smooth_ordinal':
            sigma = self.training_config.get('ordinal_sigma', 5.0)
            self.criterion = SmoothOrdinalLoss(num_classes=self.num_classes, sigma=sigma)
            logger.info(f"Using SmoothOrdinalLoss (sigma={sigma})")
            logger.info("  -> Nearby token predictions will be rewarded")
        
        elif loss_type == 'ordinal':
            margin = self.training_config.get('ordinal_margin', 0.5)
            self.criterion = OrdinalRegressionLoss(num_classes=self.num_classes, margin=margin)
            logger.info(f"Using OrdinalRegressionLoss (margin={margin})")
            logger.info("  -> Distance-aware ordinal constraints")
        
        elif loss_type == 'distance_weighted':
            alpha = self.training_config.get('distance_alpha', 0.01)
            self.criterion = DistanceWeightedCrossEntropy(num_classes=self.num_classes, alpha=alpha)
            logger.info(f"Using DistanceWeightedCrossEntropy (alpha={alpha})")
            logger.info("  -> Cross-entropy weighted by prediction distance")
        
        else:  # Default: cross_entropy
            self.criterion = nn.CrossEntropyLoss(
                weight=None,  # Will be set later if use_class_weights=True
                label_smoothing=label_smoothing
            )
            logger.info(f"Using CrossEntropyLoss (label_smoothing={label_smoothing})")
        
        self.monitor_metric = 'val_loss'
        
        # Save hyperparameters
        # Note: If 'model' is passed as an argument, it must be ignored or saved manually
        self.save_hyperparameters(ignore=['model'])
        
        logger.info(f"Initialized CryptoLightningModule")
        logger.info(f"  - Learning rate: {self.training_config['learning_rate']}")
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def _shared_step(self, batch, stage: str):
        sequences, targets = batch  # sequences: (batch, seq_len, features), targets: (batch, seq_len, 1) or (batch, seq_len)
        
        # Apply Gaussian noise augmentation during training only
        if stage == 'train' and self.training:
            noise_std = self.training_config.get('gaussian_noise', 0.0)
            if noise_std > 0:
                vocab_size = self.config['model']['vocab_size']
                
                # Add Gaussian noise to input sequences (already integers)
                # Convert to float, add noise, then round and clip
                sequences_float = sequences.float()
                noise = torch.randn_like(sequences_float) * noise_std
                sequences_noisy = sequences_float + noise
                
                # Round to nearest integer and clip to valid token range [0, vocab_size-1]
                sequences = torch.clamp(torch.round(sequences_noisy), 0, vocab_size - 1).long()
                
                # Double-check: ensure all values are in valid range
                assert sequences.min() >= 0 and sequences.max() < vocab_size, \
                    f"Noise produced invalid tokens: min={sequences.min()}, max={sequences.max()}, vocab_size={vocab_size}"
        
        # Forward pass
        outputs = self(sequences)  # V1/V2: (batch, num_coins, num_classes) or V3: dict
        
        # Targets are hard labels (no soft labels for MVP)
        # Handle different target shapes
        if targets.dim() == 2:
            # Hard labels with shape (batch, 1) or (batch, num_targets)
            target_labels = targets.squeeze(1) if targets.shape[1] == 1 else targets.squeeze()
            is_one_hot = False
        else:
            # (batch,) shape
            target_labels = targets
            is_one_hot = False
        
        # Check if V3 model (dict outputs) or V1/V2 (tensor outputs)
        is_v3 = isinstance(outputs, dict)
        
        if is_v3:
            # CryptoTransformerV3: dict with 'logits', 'regression', 'quantiles'
            logits = outputs['logits']  # (batch, num_classes)
            
            # Multi-task loss
            loss_cls = self.criterion(logits, target_labels.long())
            loss = loss_cls
            
            # Auxiliary losses (if enabled)
            multitask_cfg = self.training_config.get('multitask', {})
            if multitask_cfg.get('enabled', False):
                # Regression loss: predict expected token index
                if 'regression' in outputs:
                    # Compute expected token from target
                    expected_token = target_labels.float()  # (batch,)
                    regression_pred = outputs['regression'].squeeze(-1)  # (batch,)
                    
                    # Huber loss (robust to outliers)
                    loss_reg = nn.functional.huber_loss(regression_pred, expected_token, reduction='mean')
                    w_reg = multitask_cfg.get('w_huber', 0.3)
                    loss = loss + w_reg * loss_reg
                    self.log(f'{stage}_loss_reg', loss_reg, prog_bar=False)
                
                # Quantile loss
                if 'quantiles' in outputs:
                    quantiles = outputs['quantiles']  # (batch, 3)
                    expected_token = target_labels.float().unsqueeze(-1)  # (batch, 1)
                    
                    # Quantile loss for τ = [0.1, 0.5, 0.9]
                    taus = torch.tensor([0.1, 0.5, 0.9], device=quantiles.device).unsqueeze(0)  # (1, 3)
                    errors = expected_token - quantiles  # (batch, 3)
                    loss_q = torch.mean(torch.maximum(taus * errors, (taus - 1) * errors))
                    
                    w_q = multitask_cfg.get('w_quantile', 0.0)
                    if w_q > 0:
                        loss = loss + w_q * loss_q
                        self.log(f'{stage}_loss_quantile', loss_q, prog_bar=False)
                
                self.log(f'{stage}_loss_cls', loss_cls, prog_bar=False)
            
            preds = torch.argmax(logits, dim=1)
            acc_value = (preds == target_labels).float().mean()
            
            # Soft accuracy (within ±1 token)
            soft_acc = ((preds - target_labels).abs() <= 1).float().mean()
            self.log(f'{stage}_soft_acc', soft_acc, prog_bar=False)
            
            # MAE on token indices
            mae = (preds - target_labels).abs().float().mean()
            self.log(f'{stage}_mae', mae, prog_bar=False)
            
        else:
            # V1/V2: Original logic
            # Get number of target coins from config
            num_target_coins = self.config['model'].get('num_target_coins', self.config['model']['num_coins'])
            
            # For single target coin, use first coin's output
            if num_target_coins == 1:
                # outputs: (batch, 1, num_classes), target_labels: (batch,)
                logits = outputs[:, 0, :]  # (batch, num_classes)
                
                # Target is hard labels (no soft labels for MVP)
                loss = self.criterion(logits, target_labels.long())
                preds = torch.argmax(logits, dim=1)
                acc_value = (preds == target_labels).float().mean()
                
                # Calculate per-class accuracies
                for class_idx in range(self.num_classes):
                    mask = target_labels == class_idx
                    if mask.sum() > 0:
                        class_acc = (preds[mask] == target_labels[mask]).float().mean()
                        self.log(f'{stage}_acc_class_{class_idx}', class_acc, prog_bar=False)
                
                # Calculate actionable accuracy (down and up only, for 3-class system)
                if self.num_classes == 3:
                    actionable_mask = target_labels != 1  # Exclude steady class
                    if actionable_mask.sum() > 0:
                        actionable_acc = (preds[actionable_mask] == target_labels[actionable_mask]).float().mean()
                        self.log(f'{stage}_acc_actionable', actionable_acc, prog_bar=True)
            else:
                # For multiple target coins (future): flatten and expand targets
                logits = outputs.view(-1, self.num_classes)  # (batch*num_coins, num_classes)
                targets_expanded = target_labels.unsqueeze(1).expand(-1, num_target_coins).reshape(-1)
                loss = self.criterion(logits, targets_expanded.long())
                
                preds = torch.argmax(logits, dim=1)
                acc_value = (preds == targets_expanded).float().mean()
        
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', acc_value, prog_bar=True)
        
        return loss

    
    def training_step(self, batch, batch_idx):
        """Training step"""
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        return self._shared_step(batch, 'val')
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 1e-5)
        )
        
        # Ensure the monitor key is correctly set
        monitor_key = self.monitor_metric
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.training_config['scheduler']['factor'],
                patience=self.training_config['scheduler']['patience'],
                min_lr=self.training_config['scheduler']['min_lr'],
                verbose=False # Set to False to avoid excessive console output unless debugging
            ),
            'monitor': monitor_key,
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class SimpleTrainer:
    """Simple trainer using PyTorch Lightning with random sampling"""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.training_config = config['training']
        self.overfit_cfg = self.training_config.get('overfit_mode', {})
        
        # Create Lightning module (without class weights initially)
        self.lightning_module = CryptoLightningModule(model, config)
        
        logger.info("Initialized SimpleTrainer")
    
    def _compute_class_weights(self, train_y: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset
        
        Args:
            train_y: Training labels (N, 1) or (N,)
            
        Returns:
            Class weights tensor of shape (num_classes,)
        """
        import numpy as np
        
        # Flatten and get class counts
        labels_flat = train_y.flatten().numpy()
        unique, counts = np.unique(labels_flat, return_counts=True)
        
        # Calculate inverse frequency weights
        total = counts.sum()
        num_classes = self.config['model']['num_classes']
        weights = total / (num_classes * counts)
        
        # Create full weight tensor (in case some classes missing)
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        for cls, weight in zip(unique, weights):
            class_weights[int(cls)] = weight
        
        logger.info(f"Computed class weights: {class_weights.tolist()}")
        return class_weights
    
    def train_simple(self, splits: list, output_dir: str = None) -> dict:
        """
        Train with simple train/val split and standard random shuffling.
        
        Args:
            splits: List of data splits
            output_dir: Directory to save checkpoints and logs (default: artifacts/step_07_train)
        """
        
        if len(splits) != 1:
            raise ValueError(f"Expected 1 split for simple training, got {len(splits)}")
        
        split = splits[0]
        logger.info("Starting simple training with standard random shuffling.")
        
        # Create datasets - X should be LongTensor (integer tokens), y should be LongTensor (class labels)
        # Ensure inputs are correctly cast to LongTensor for embedding lookup
        # If inputs are already tensors (from .pt files), this is a no-op
        # Keep tensors on CPU - DataLoader will transfer to GPU batch-by-batch
        if isinstance(split['train_X'], torch.Tensor):
            train_X = split['train_X'].cpu()
            train_y = split['train_y'].cpu()
            val_X = split['val_X'].cpu()
            val_y = split['val_y'].cpu()
        else:
            train_X = torch.as_tensor(split['train_X'], dtype=torch.long)
            train_y = torch.as_tensor(split['train_y'], dtype=torch.long)
            val_X = torch.as_tensor(split['val_X'], dtype=torch.long)
            val_y = torch.as_tensor(split['val_y'], dtype=torch.long)
        
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        
        # Compute class weights/alpha if using focal loss or class-weighted CE
        loss_type = self.training_config.get('loss_type', 'cross_entropy')
        
        if loss_type == 'focal' and self.training_config.get('compute_focal_alpha', True):
            # Compute alpha weights for focal loss
            alpha = self._compute_class_weights(train_y)
            # Update focal loss with computed alpha
            self.lightning_module.criterion.alpha = alpha.to(self.device)
            logger.info(f"Updated Focal Loss with computed alpha: {alpha.tolist()}")
            
        elif loss_type == 'cross_entropy' and self.training_config.get('use_class_weights', False):
            # Check if custom class weights are provided in config
            custom_weights = self.training_config.get('class_weights', None)
            if custom_weights is not None:
                # Use custom weights from config
                class_weights = torch.tensor(custom_weights, dtype=torch.float32)
                logger.info(f"Using custom class weights from config: {class_weights.tolist()}")
            else:
                # Compute class weights from data
                class_weights = self._compute_class_weights(train_y)
                logger.info(f"Computed class weights from data: {class_weights.tolist()}")
            
            # Update criterion with weights
            self.lightning_module.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device),
                label_smoothing=self.training_config.get('label_smoothing', 0.0)
            )
            logger.info(f"Updated CrossEntropyLoss with class weights")
        
        # Apply train sampling if configured
        train_sample_pct = self.training_config.get('train_sample_pct', 1.0)
        if train_sample_pct < 1.0:
            n_samples = int(len(train_dataset) * train_sample_pct)
            logger.info(f"Using {train_sample_pct*100:.0f}% of training data ({n_samples} samples)")
            # Randomly sample indices without replacement
            import random
            sampled_indices = random.sample(range(len(train_dataset)), n_samples)
            train_dataset = Subset(train_dataset, sampled_indices)
        else:
            logger.info(f"Using full training dataset ({len(train_dataset)} samples)")
        
        # Create data loaders
        # pin_memory=True enables faster CPU->GPU transfer
        # num_workers > 0 enables parallel data loading on CPU
        num_workers = self.training_config.get('num_workers', 0)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Train loader: {len(train_loader)} batches")
        logger.info(f"Val loader: {len(val_loader)} batches")
        
        # Create trainer
        grad_clip = self.training_config.get('max_grad_norm', 1.0)
        if grad_clip is not None and grad_clip <= 0:
            grad_clip = None
        
        # Print model structure with input/output dimensions
        logger.info("=" * 80)
        logger.info("MODEL ARCHITECTURE")
        logger.info("=" * 80)
        logger.info(f"Model type: {self.config['model']['type']}")
        logger.info("")
        logger.info("INPUT DIMENSIONS:")
        logger.info(f"  - Batch size: {self.training_config['batch_size']}")
        logger.info(f"  - Sequence length (lookback): {self.config['model']['sequence_length']} hours")
        logger.info(f"  - Input coins: {self.config['model']['num_coins']}")
        logger.info(f"  - Features per coin: {self.config['model']['features_per_coin']}")
        logger.info(f"  - Total input features: {self.config['model']['num_coins'] * self.config['model']['features_per_coin']} dims")
        logger.info(f"  - Input tensor shape: ({self.training_config['batch_size']}, {self.config['model']['sequence_length']}, {self.config['model']['num_coins'] * self.config['model']['features_per_coin']})")
        logger.info("")
        logger.info("ARCHITECTURE:")
        logger.info(f"  - Embedding dimension: {self.config['model']['embedding_dim']}")
        logger.info(f"  - Vocabulary size: {self.config['model']['vocab_size']}")
        logger.info(f"  - Number of transformer layers: {self.config['model']['num_layers']}")
        logger.info(f"  - Attention heads: {self.config['model']['num_heads']}")
        logger.info(f"  - Feedforward dimension: {self.config['model']['feedforward_dim']}")
        logger.info(f"  - Dropout rate: {self.config['model']['dropout']}")
        logger.info("")
        logger.info("OUTPUT DIMENSIONS:")
        logger.info(f"  - Target coins to predict: {self.config['model'].get('num_target_coins', self.config['model']['num_coins'])}")
        logger.info(f"  - Output classes: {self.config['model']['num_classes']} (price movement bins)")
        logger.info(f"  - Output tensor shape: ({self.training_config['batch_size']}, {self.config['model'].get('num_target_coins', self.config['model']['num_coins'])}, {self.config['model']['num_classes']})")
        logger.info("")
        logger.info("TRAINING SETTINGS:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Batch size: {self.training_config['batch_size']}")
        logger.info(f"  - Learning rate: {self.training_config['learning_rate']}")
        logger.info(f"  - Weight decay: {self.training_config.get('weight_decay', 0)}")
        logger.info(f"  - Precision: {self.training_config.get('precision', 32)}-bit")
        logger.info(f"  - Gradient clipping: {grad_clip}")
        logger.info(f"  - Num workers: {num_workers}")
        logger.info("=" * 80)
        logger.info("")
        
        # Set up output directory for checkpoints and logs
        if output_dir is None:
            output_dir = "artifacts/step_07_train"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Callbacks
        cb_list = [ConsoleLogger(), BestMetricsTracker()] # Use the simplified logger
        
        # Add warmup callback if configured
        warmup_config = self.training_config.get('warmup', {})
        if warmup_config.get('epochs', 0) > 0:
            warmup_epochs = warmup_config['epochs']
            start_lr = warmup_config.get('start_lr', 1e-7)
            target_lr = self.training_config['learning_rate']
            cb_list.append(WarmupCallback(warmup_epochs, start_lr, target_lr))
            logger.info(f"Warmup enabled: {warmup_epochs} epochs, {start_lr:.2e} -> {target_lr:.2e}")
            logger.info("")
        
        # Add standard callbacks unless in overfit mode
        if not self.overfit_cfg.get('enabled', False):
            cb_list.append(EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping']['patience'],
                mode='min',
                verbose=False # Set verbose to False to rely on the ConsoleLogger/PL progress bar
            ))
            # ModelCheckpoint: Save the best model based on the monitored metric
            # Use output_dir for checkpoint directory
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            cb_list.append(ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor='val_loss',
                save_top_k=1,
                mode='min',
                filename='best_model-{epoch:02d}-{val_loss:.4f}',
                save_last=True # Save the last model as well
            ))
        
        # Determine accelerator configuration
        accelerator = 'gpu' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        
        trainer = Trainer(
            max_epochs=self.training_config['epochs'],
            accelerator=accelerator,
            devices=1,
            callbacks=cb_list,
            logger=False,  # Disable logger for speed (use progress bar instead)
            enable_progress_bar=True,
            gradient_clip_val=grad_clip,
            deterministic=False,
            enable_checkpointing=True,
            precision=self.training_config.get('precision', 32),
            val_check_interval=self.training_config.get('val_check_interval', 1.0),  # Validate once per epoch
            num_sanity_val_steps=0,  # Disable sanity check
            log_every_n_steps=10,  # Log every 10 batches
            default_root_dir=output_dir  # Set root directory for all outputs
        )
        
        # Overfit mode handling
        if self.overfit_cfg.get('enabled', False):
            n_samples = int(self.overfit_cfg.get('num_samples', 256))
            val_samples = min(64, len(val_dataset))
            
            # Rebuild loaders with truncated datasets
            train_loader = DataLoader(
                Subset(train_dataset, list(range(min(n_samples, len(train_dataset))))),
                batch_size=self.training_config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            val_loader = DataLoader(
                Subset(val_dataset, list(range(val_samples))),
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            logger.info("Starting training (overfit mode)...")
            trainer.fit(self.lightning_module, train_loader, val_loader)
            logger.info("Training completed (overfit mode)")
        else:
            logger.info("Starting training...")
            trainer.fit(self.lightning_module, train_loader, val_loader)
            logger.info("Training completed")
        
        # Get best metrics from tracker callback
        best_metrics_tracker = None
        for callback in trainer.callbacks:
            if isinstance(callback, BestMetricsTracker):
                best_metrics_tracker = callback
                break
        
        # Retrieve best model checkpoint if available
        best_model_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        
        # Load best model if checkpoint exists
        if best_model_path and best_metrics_tracker:
            logger.info(f"Loading best model from checkpoint: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            # Load into the lightning module (not just the model)
            # This handles the 'model.' prefix correctly
            self.lightning_module.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Best model loaded (from epoch {best_metrics_tracker.best_epoch})")
        
        # Create history dict with best metrics
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_loss': best_metrics_tracker.best_val_loss if best_metrics_tracker else float('nan'),
            'best_val_acc': best_metrics_tracker.best_val_acc if best_metrics_tracker else float('nan'),
            'best_epoch': best_metrics_tracker.best_epoch if best_metrics_tracker else -1,
            'best_model_path': best_model_path
        }
        
        logger.info("Simple training complete!")
        if best_metrics_tracker:
            logger.info(f"Best validation loss: {best_metrics_tracker.best_val_loss:.6f} (epoch {best_metrics_tracker.best_epoch})")
            logger.info(f"Best validation accuracy: {best_metrics_tracker.best_val_acc:.6f} (epoch {best_metrics_tracker.best_epoch})")
        if best_model_path:
            logger.info(f"Best model saved at: {best_model_path}")
        
        return history
    
    def get_model(self) -> nn.Module:
        """Get the trained model"""
        # If using checkpointing, it might be better to load the best model here
        return self.lightning_module.model