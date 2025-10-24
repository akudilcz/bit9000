# System Utilities

This folder contains system-level utilities for the token prediction pipeline.

## Files

### `tune.py`
Hyperparameter tuning using Optuna for automated parameter optimization.

**Features:**
- Bayesian optimization with TPE sampler
- Automatic early stopping for unpromising trials
- Efficient trial pruning (MedianPruner)
- Parameter importance visualization
- Search spaces for training, architecture, and augmentation hyperparameters

**Usage:**
```python
from src.system.tune import run_tuning

artifact = run_tuning(
    config=config,
    sequences_artifact=sequences_artifact,
    output_dir=Path("artifacts/step_06a_tune"),
    num_trials=30,
    epochs_per_trial=20
)
```

**Search Space:**
- Learning rate: 5e-6 to 5e-3 (log scale)
- Weight decay: 0.0 to 0.1
- Dropout: 0.05 to 0.5
- Label smoothing: 0.0 to 0.2
- Batch size: [64, 128, 256]
- Embedding dim: [32, 64, 128]
- Num layers: 2 to 6
- Num heads: [4, 8]
- Feedforward dim: [128, 256, 512, 1024]

**Outputs:**
- `tuning_results.json` - Best parameters and summary
- `trials_history.json` - All trials with their performance
- `tuning_history.png` - Optimization progress plot
- `param_importance.png` - Feature importance visualization

**Tuning Budget:**
- Quick: 10 trials × 10 epochs (~1-2 hours)
- Standard: 30 trials × 20 epochs (~6-12 hours)
- Deep: 50 trials × 50 epochs (~24+ hours)

---

### `pretrain.py`
Synthetic data generation for pretraining token prediction models.

**Features:**
- Generate synthetic price movements matching real coin statistics
- Preserve correlation structure between coins
- Create balanced token sequences
- Support for data augmentation during pretraining

**Purpose:**
- Pretrain model weights on unlimited synthetic data
- Improve initialization before fine-tuning on real data
- Test model architecture and training pipeline
- Generate training data when real data is limited

**Usage:**
```python
from src.system.pretrain import run_pretraining

artifact = run_pretraining(
    config=config,
    tokenize_artifact=tokenize_artifact,
    output_dir=Path("artifacts/step_06a_pretrain"),
    num_samples=100000,
    epochs=50
)
```

**Synthetic Data Generation:**
- Learns statistics from real tokenization thresholds
- Generates multi-coin sequences with realistic correlations
- Preserves volatility and momentum patterns
- Creates balanced class distributions

**Pretraining Strategy:**
1. Fit statistical models on real token sequences
2. Generate large synthetic dataset (100K+ samples)
3. Pretrain model on synthetic data
4. Fine-tune on real data (Step 6: Train)

**Outputs:**
- `pretrained_model.pt` - Model weights after pretraining
- `pretraining_history.json` - Loss curves and metrics
- `synthetic_data_stats.json` - Statistics of generated data
- `synthetic_samples.png` - Visualization of generated sequences

---

## Integration with Pipeline

These utilities are **optional** enhancements to the base pipeline:

### Without Tuning/Pretraining (default):
```
Step 5: Sequences → Step 6: Train → Step 7: Evaluate
```

### With Tuning:
```
Step 5: Sequences → Step 6A: Tune → Step 6: Train (best params) → Step 7: Evaluate
```

### With Pretraining:
```
Step 5: Sequences → Step 6A: Pretrain → Step 6: Train (finetune) → Step 7: Evaluate
```

### With Both:
```
Step 5: Sequences → Step 6A: Pretrain → Step 6B: Tune → Step 6: Train → Step 7: Evaluate
```

---

## CLI Integration

To run from command line:

```bash
# Hyperparameter tuning
python main.py pipeline tune --num-trials 30

# Pretraining with synthetic data
python main.py pipeline pretrain --num-samples 100000 --epochs 50

# Run full pipeline with tuning
python main.py pipeline run-all --with-tuning
```

---

## Notes

- Both utilities save all outputs to `artifacts/` directory
- Tuning results can be automatically applied to `config.yaml`
- Pretrained weights are loaded automatically in Step 6 if available
- All artifacts follow the same schema as other pipeline steps




