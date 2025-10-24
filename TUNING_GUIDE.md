# Hyperparameter Tuning Guide

This guide explains how to run hyperparameter tuning for the cryptocurrency price prediction model.

## Prerequisites

1. **Python 3.8+** with PyTorch and required dependencies
2. **Pipeline Data**: Must have completed pipeline steps 0-5 (to generate sequences)
3. **GPU (Optional but Recommended)**: CUDA-capable GPU for faster training

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Sequence Data

Before tuning, you need sequence data from the pipeline:

```bash
# Run complete pipeline (steps 0-5)
python main.py pipeline run-all

# Or run from an existing download
python main.py pipeline run-from-clean
```

This creates:
- `artifacts/step_05_sequences/train_X.pt` - Training input sequences (N, 24, 10, 2)
- `artifacts/step_05_sequences/train_y.pt` - Training targets (N,)
- `artifacts/step_05_sequences/val_X.pt` - Validation inputs
- `artifacts/step_05_sequences/val_y.pt` - Validation targets

### Step 2: Run Hyperparameter Tuning

```bash
# Quick tuning: 20 trials × 10 epochs (≈2-4 hours on GPU)
python main.py pipeline tune --num-trials 20 --epochs-per-trial 10

# Standard tuning: 30 trials × 20 epochs (≈6-12 hours on GPU)
python main.py pipeline tune --num-trials 30 --epochs-per-trial 20

# Thorough tuning: 50 trials × 30 epochs (≈12-24 hours on GPU)
python main.py pipeline tune --num-trials 50 --epochs-per-trial 30 --timeout-hours 24
```

## Output

Tuning results are saved to `artifacts/tuning/`:

```
artifacts/tuning/
├── tuning_results.json      # Best parameters and summary
├── trials_history.json      # Full trial data
├── tuning_history.png       # Optimization history plot
└── param_importance.png     # Parameter importance chart
```

### Best Parameters Example

```json
{
  "best_trial": 15,
  "best_val_loss": 4.123456,
  "best_params": {
    "learning_rate": 0.000234,
    "weight_decay": 1.5e-05,
    "dropout": 0.15,
    "label_smoothing": 0.05,
    "max_grad_norm": 1.2,
    "batch_size": 256,
    "embedding_dim": 128,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 8,
    "feedforward_dim": 1024,
    "warmup_epochs": 3
  }
}
```

## What Gets Tuned

### Training Hyperparameters
- **learning_rate**: 5e-6 to 5e-3 (logarithmic search)
- **weight_decay**: 1e-7 to 1e-2 (logarithmic search)
- **dropout**: 0.05 to 0.3
- **label_smoothing**: 0.0 to 0.1
- **max_grad_norm**: 0.5 to 2.0
- **batch_size**: {64, 128, 256}

### Architecture Hyperparameters
- **embedding_dim**: {32, 64, 128}
- **num_layers**: 2 to 6
- **num_heads**: {4, 8}
- **feedforward_dim**: {256, 512, 1024}
- **d_model**: automatically derived from embedding_dim (embedding_dim × 4)
- **warmup_epochs**: 2 to 5

## Understanding the Results

### Tuning History Plot
Shows validation loss for each trial. Best trials have lower loss.

### Parameter Importance Plot
Shows which hyperparameters have the most impact on model performance. Focus tuning on high-importance parameters.

### Trials History
Each trial record contains:
- **state**: "COMPLETE", "PRUNED", or "FAILED"
- **value**: validation loss (lower is better)
- **params**: all sampled hyperparameters
- **duration_seconds**: wall-clock time for the trial

## Using Tuned Parameters

After tuning, apply the best parameters to your config:

```yaml
# config.yaml
model:
  embedding_dim: 128  # From best_params
  num_layers: 4       # From best_params
  num_heads: 8        # From best_params
  feedforward_dim: 1024  # From best_params
  dropout: 0.15       # From best_params
  d_model: 512        # Automatically derived

training:
  learning_rate: 0.000234      # From best_params
  weight_decay: 1.5e-05        # From best_params
  label_smoothing: 0.05        # From best_params
  max_grad_norm: 1.2           # From best_params
  batch_size: 256              # From best_params
  warmup:
    epochs: 3                  # From best_params
```

Then train the final model:

```bash
python main.py pipeline train
```

## Advanced Usage

### Custom Search Space

Edit `src/system/tune.py` `_suggest_hyperparameters()` method to modify search ranges:

```python
def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
    # Tighter search (faster):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    
    # Broader search (slower):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
```

### Early Stopping During Tuning

Trials are automatically pruned if they're underperforming:
- First 5 trials run fully (warmup)
- After that, trials pruned if median loss worse than best
- This saves time by stopping bad trials early

### GPU/CPU Selection

Tuning automatically detects and uses:
- CUDA GPU if available (fast, ~8-12 hours for 30 trials)
- CPU fallback (slow, ~24-48 hours for 30 trials)

To force CPU:

```python
# Edit src/system/tune.py line 78:
self.device = torch.device("cpu")
```

## Performance Expectations

### Time Estimates

| Setup | 20 trials | 30 trials | 50 trials |
|-------|-----------|-----------|-----------|
| GPU (RTX 3080) | 2-3h | 5-8h | 8-12h |
| GPU (RTX 4090) | 1-2h | 2-4h | 4-6h |
| CPU (16-core) | 8-12h | 12-18h | 24-36h |

### Memory Requirements

- GPU: 8-12 GB VRAM (batch_size=256, embedding_dim=128)
- CPU: 16-32 GB RAM

## Troubleshooting

### "No sequences yet - need to run pipeline first"

Run the pipeline first:
```bash
python main.py pipeline run-all
```

### Out of Memory Errors

Reduce parameters:
```bash
# Reduce batch_size in tune search space
python main.py pipeline tune --num-trials 10 --epochs-per-trial 5
```

### Tuning Too Slow

1. Reduce `--epochs-per-trial` (10 instead of 20)
2. Reduce `--num-trials` (20 instead of 50)
3. Use GPU instead of CPU
4. Add `--timeout-hours N` to limit total time

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch:
pip install torch torchvision torchaudio pytorch-cuda=11.8 -f https://download.pytorch.org/whl/torch_stable.html
```

## Examples

### Example 1: Quick Exploration (2-4 hours)

```bash
python main.py pipeline tune --num-trials 20 --epochs-per-trial 10
```

Perfect for:
- Testing the tuning infrastructure
- Quick parameter exploration
- Development/debugging

### Example 2: Standard Production (6-12 hours)

```bash
python main.py pipeline tune --num-trials 30 --epochs-per-trial 20
```

Perfect for:
- Finding good hyperparameters
- Production use
- Typical ML workflow

### Example 3: Comprehensive Search (12-24 hours)

```bash
python main.py pipeline tune --num-trials 50 --epochs-per-trial 30 --timeout-hours 24
```

Perfect for:
- Maximum performance
- Final model tuning
- High-stakes applications

## Next Steps

1. Run tuning with appropriate settings
2. Review `tuning_results.json` for best parameters
3. Update `config.yaml` with best parameters
4. Run `python main.py pipeline train` with optimized config
5. Evaluate: `python main.py pipeline evaluate`
6. Predict: `python main.py pipeline inference`

## References

- Optuna Documentation: https://optuna.readthedocs.io/
- TPE Sampler: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
- PyTorch Optimization: https://pytorch.org/docs/stable/optim.html
