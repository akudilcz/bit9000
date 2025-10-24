# System Module - Pretraining & Tuning

This module contains utilities for pretraining and hyperparameter tuning of the token predictor model.

## Contents

- `pretrain.py` - Pretraining utilities for model initialization
- `tune.py` - Automated hyperparameter tuning using Optuna
- `README.md` - This file

## Hyperparameter Tuning

The `tune.py` module provides automated hyperparameter optimization using Optuna's TPE sampler with median pruning.

### Prerequisites

1. **Sequences Data**: Must have completed pipeline steps 0-5 (download, clean, split, tokenize, sequences)
2. **Dependencies**: Install requirements including optuna and pytorch-lightning
   ```bash
   pip install -r requirements.txt
   ```

### Running Tuning

```python
from pathlib import Path
import torch
import yaml
from src.system.tune import run_tuning

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load sequence tensors
sequences_dir = Path('artifacts/step_05_sequences')
train_X = torch.load(sequences_dir / 'train_X.pt')
train_y = torch.load(sequences_dir / 'train_y.pt')
val_X = torch.load(sequences_dir / 'val_X.pt')
val_y = torch.load(sequences_dir / 'val_y.pt')

# Run tuning
results = run_tuning(
    config=config,
    train_X=train_X,
    train_y=train_y,
    val_X=val_X,
    val_y=val_y,
    output_dir=Path('artifacts/tuning'),
    num_trials=30,
    epochs_per_trial=20,
    timeout_hours=None
)

print("Best parameters:", results['best_params'])
print("Best validation loss:", results['best_val_loss'])
```

### What Gets Tuned

The tuning explores:

**Training Hyperparameters**:
- `learning_rate`: 5e-6 to 5e-3 (log scale)
- `weight_decay`: 1e-7 to 1e-2 (log scale)
- `dropout`: 0.05 to 0.3
- `label_smoothing`: 0.0 to 0.1
- `max_grad_norm`: 0.5 to 2.0
- `batch_size`: {64, 128, 256}

**Architecture Hyperparameters**:
- `embedding_dim`: {32, 64, 128}
- `num_layers`: 2 to 6
- `num_heads`: {4, 8}
- `feedforward_dim`: {256, 512, 1024}
- `warmup_epochs`: 2 to 5

### Outputs

The tuning produces:
- `tuning_results.json` - Best parameters and metrics
- `trials_history.json` - Full trial data (state, loss, params, duration)
- `tuning_history.png` - Optimization history plot
- `param_importance.png` - Parameter importance analysis

### Configuration

Key tuning parameters in `tune.py`:

```python
num_trials = 30           # Number of trials to run
epochs_per_trial = 20     # Training epochs per trial (shorter = faster)
timeout_hours = None      # Optional timeout (None = unlimited)
```

Reduce `epochs_per_trial` for faster exploration (e.g., 10-15 epochs), or increase `num_trials` for more thorough search (e.g., 50-100).

### Example: Quick Tuning Run

```python
# Fast exploration: 20 trials × 10 epochs = ~2-4 hours on GPU
results = run_tuning(
    config=config,
    train_X=train_X,
    train_y=train_y,
    val_X=val_X,
    val_y=val_y,
    output_dir=Path('artifacts/tuning_quick'),
    num_trials=20,
    epochs_per_trial=10
)
```

### Example: Thorough Tuning Run

```python
# Thorough search: 50 trials × 30 epochs = ~8-12 hours on GPU
results = run_tuning(
    config=config,
    train_X=train_X,
    train_y=train_y,
    val_X=val_X,
    val_y=val_y,
    output_dir=Path('artifacts/tuning_thorough'),
    num_trials=50,
    epochs_per_trial=30,
    timeout_hours=12
)
```

## Model Specifications for Tuning

The tuned model is `SimpleTokenPredictor` with:
- **Vocabulary**: 256 bins (0-255)
- **Input**: 24 hours × 10 coins × 2 channels (price + volume)
- **Output**: Single next-hour token (256 classes)
- **Architecture**: Transformer decoder-only with causal masking
- **Training**: Teacher forcing on single next token
- **Inference**: Autoregressive generation for 8 hours

## Notes

- Tuning uses GPU if available (CUDA). Falls back to CPU automatically.
- Early stopping is built into tuning via MedianPruner (stops unpromising trials early).
- Best parameters are saved and can be used directly in `config.yaml` for final training.
- Validation loss is the optimization metric (lower = better).




