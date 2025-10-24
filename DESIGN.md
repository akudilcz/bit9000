# Simple Multi-Coin Token Predictor - Technical Specification

## Overview

**Objective**: Train a lightweight transformer decoder to predict 8-hour XRP price movements using 24 hours of multi-coin token sequences.

**Input**: `(24, 10, 2)` - 24 hours × 10 coins × 2 channels (price + volume)  
**Output**: `(8,)` - 8 hours of XRP price direction tokens  
**Vocabulary**: 3 tokens `{0: down, 1: steady, 2: up}`  
**Architecture**: Transformer decoder-only with causal masking

---

## Data Flow Summary

```
Step 0: Reset
  └─> Clean artifacts directory

Step 1: Download
  └─> OHLCV data (2020-2025, 10 coins, hourly)
      ├─ raw_data.parquet [T × (10 coins × 5 OHLCV)]
      └─ Visualizations: price trends, data quality

Step 2: Clean
  └─> Fill gaps, remove outliers
      ├─ clean_data.parquet [T × (10 coins × 5 OHLCV)]
      └─ Quality metrics: % filled, % outliers

Step 3: Split
  └─> Temporal split (80% train, 20% val)
      ├─ train_clean.parquet [T_train × (10 coins × 5)]
      └─ val_clean.parquet [T_val × (10 coins × 5)]

Step 4: Tokenize
  └─> Convert price/volume to 3-class tokens
      ├─ Compute log returns (price) and log changes (volume)
      ├─ Fit quantile thresholds (33rd, 67th percentile) on train
      ├─ Apply thresholds to train & val
      ├─ train_tokens.parquet [T_train × 20 cols: COIN_price, COIN_volume]
      ├─ val_tokens.parquet [T_val × 20 cols]
      └─ fitted_thresholds.json {coin: {price: (τ_low, τ_high), volume: (...)}}

Step 5: Sequences
  └─> Create rolling windows (24h input → 8h target)
      ├─ train_X.pt [(N_train, 24, 10, 2)] - all coins, 2 channels
      ├─ train_y.pt [(N_train, 8)] - XRP price only
      ├─ val_X.pt [(N_val, 24, 10, 2)]
      └─ val_y.pt [(N_val, 8)]

Step 6: Train
  └─> Transformer decoder with teacher forcing
      ├─ Embed price & volume tokens separately
      ├─ Fuse channels → d_model=128
      ├─ Aggregate coins (mean pooling)
      ├─ Apply positional encoding
      ├─ Transformer decoder (4 layers, causal mask)
      ├─ Output head → 8 steps × 3 classes
      ├─ model.pt (best checkpoint)
      └─ history.json (loss/accuracy curves)

Step 7: Evaluate
  └─> Autoregressive generation on validation set
      ├─ Per-hour accuracy (hours 1-8)
      ├─ Sequence accuracy (all 8 correct)
      ├─ Baseline comparison (persistence, random)
      └─ eval_results.json + confusion matrices

Step 8: Inference
  └─> Real-time prediction
      ├─ Fetch last 24h of OHLCV
      ├─ Tokenize with fitted thresholds
      ├─ Autoregressive generation (8 steps)
      └─ predictions.json {hour: 1-8, token, probabilities}
```

**Key Transformations**:
- OHLCV → Tokens: `log(price[t]/price[t-1])` → quantile thresholds → `{0,1,2}`
- Tokens → Sequences: sliding window (stride=1) → `(24h, 10 coins, 2 ch)` inputs
- Training: Teacher forcing with ground truth targets
- Inference: Autoregressive generation (no ground truth)

---

## Design Principles

1. **Simplicity**: Raw price/volume → tokens → predictions (no feature engineering)
2. **Balanced classes**: Quantile thresholds ensure ~33% distribution per class
3. **Multi-coin context**: BTC/ETH patterns inform XRP predictions
4. **Autoregressive**: Each hour predicted conditionally on previous predictions
5. **No data leakage**: Fit thresholds on training data only
6. **Decoder-only**: Causal masking prevents future information leakage

---

## Pipeline Steps

### Step 0-3: Data Foundation (Existing)
- **Step 0**: Reset artifacts
- **Step 1**: Download hourly OHLCV (2020-2025, 10 coins)
- **Step 2**: Clean data (fill gaps, remove outliers)
- **Step 3**: Temporal split (80% train, 20% val)

### Step 4: Tokenization

**Input**: Clean OHLCV data  
**Output**: Token DataFrames with 2 channels per coin

**Process**:
1. **Fit Phase** (training data only):
   - Compute log returns: `r_price = log(close[t] / close[t-1])`
   - Compute log changes: `r_volume = log(volume[t] / volume[t-1])`
   - Calculate 33rd and 67th percentiles per coin per channel
   - Save thresholds: `{coin: {price: (τ_low, τ_high), volume: (τ_low, τ_high)}}`

2. **Transform Phase** (train + val):
   - Apply fitted thresholds:
     - Token = 0 if `r ≤ τ_low`
     - Token = 1 if `τ_low < r ≤ τ_high`
     - Token = 2 if `r > τ_high`

**Artifacts**:
- `train_tokens.parquet`: columns like `BTC_price`, `BTC_volume`, etc.
- `val_tokens.parquet`
- `fitted_thresholds.json`
- Visualizations: token distribution, threshold heatmap

### Step 5: Sequence Creation

**Input**: Tokenized DataFrames  
**Output**: PyTorch tensors

**Process**:
- Create rolling windows (stride=1):
  - `X`: 24 consecutive hours × all coins × 2 channels
  - `y`: next 8 hours of XRP price tokens only
- Drop incomplete windows

**Tensor Shapes**:
- `train_X.pt`: `(N_train, 24, 10, 2)` dtype=long
- `train_y.pt`: `(N_train, 8)` dtype=long
- `val_X.pt`: `(N_val, 24, 10, 2)` dtype=long
- `val_y.pt`: `(N_val, 8)` dtype=long

### Step 6: Model Training

**Architecture**: Transformer Decoder-Only

**Components**:
1. **Token Embeddings**: Separate embeddings for price (3×64) and volume (3×64)
2. **Channel Fusion**: Concatenate and project to `d_model=128`
3. **Coin Aggregation**: Mean pooling across coins per timestep
4. **Positional Encoding**: Sinusoidal (max_len = 24+8 = 32)
5. **Transformer Decoder**: 4 layers, 4 heads, causal masking
6. **Output Head**: Linear projection to 3 classes

**Training**:
- **Loss**: Cross-entropy with teacher forcing
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Batch size**: 256
- **Gradient clipping**: max_norm=1.0
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)
- **Early stopping**: patience=20 epochs
- **Warmup**: 5 epochs (1e-7 → 1e-4)

**Artifacts**:
- `model.pt`: Best checkpoint (by val loss)
- `history.json`: Training metrics
- Visualizations: loss curves, accuracy curves

### Step 7: Evaluation

**Metrics**:
1. **Per-hour accuracy**: Accuracy at each of 8 prediction steps
2. **Sequence accuracy**: % where all 8 predictions correct
3. **Confusion matrices**: Per-hour (hours 1, 4, 8)
4. **Baseline comparisons**:
   - Persistence: repeat last token
   - Random: uniform distribution

**Target Performance**:
- Hour-1 accuracy > 40% (baseline: 33% random, ~38% persistence)
- Hour-8 accuracy > 35%
- Sequence accuracy > 1% (baseline: 0.015% random)

**Artifacts**:
- `eval_results.json`: All metrics
- Visualizations: accuracy decay, confusion matrices, baseline comparison

### Step 8: Inference

**Process**:
1. Fetch last 24+1 hours of OHLCV data
2. Tokenize using fitted thresholds (2 channels)
3. Create tensor: `(1, 24, 10, 2)`
4. Run `model.generate()` autoregressively for 8 steps
5. Return predictions with probabilities

**Output Format**:
```json
{
  "timestamp": "2025-10-23T12:00:00Z",
  "coin": "XRP",
  "horizon_hours": 8,
  "predictions": [
    {
      "hour": 1,
      "prediction": "up",
      "confidence": 0.57,
      "probabilities": {"down": 0.15, "steady": 0.28, "up": 0.57}
    },
    ...
  ]
}
```

---

## Model Specification

### SimpleTokenPredictor

**Parameters**:
- `vocab_size`: 3
- `embedding_dim`: 64
- `d_model`: 128 (embedding_dim × 2 after channel fusion)
- `num_heads`: 4
- `num_layers`: 4
- `feedforward_dim`: 256
- `dropout`: 0.1
- `input_length`: 24
- `output_length`: 8
- `num_coins`: 10
- `num_classes`: 3
- `num_channels`: 2

**Methods**:
- `forward(x, targets=None)`: Training with teacher forcing
- `generate(x, max_len)`: Autoregressive inference

**Model Size**: ~3.2M parameters ≈ 12.8 MB (fp32)

---

## Configuration (config.yaml)

```yaml
data:
  coins: [BTC, ETH, BNB, XRP, SOL, DOGE, ADA, AVAX, DOT, LTC]
  target_coin: XRP
  interval: 1h

split:
  train_ratio: 0.8
  temporal: true

tokenization:
  vocab_size: 3
  method: quantile
  percentiles: [33, 67]

sequences:
  input_length: 24
  output_length: 8
  num_channels: 2

model:
  type: SimpleTokenPredictor
  vocab_size: 3
  num_classes: 3
  num_coins: 10
  embedding_dim: 64
  d_model: 128
  num_heads: 4
  num_layers: 4
  feedforward_dim: 256
  dropout: 0.1

training:
  device: cuda
  epochs: 100
  batch_size: 256
  learning_rate: 0.0001
  weight_decay: 0.00001
  label_smoothing: 0.0
  max_grad_norm: 1.0
  early_stopping:
    patience: 20
  warmup:
    epochs: 5
    start_lr: 0.0000001
  scheduler:
    type: ReduceLROnPlateau
    factor: 0.5
    patience: 10
    min_lr: 0.000001
```

---

## Artifact Structure

```
artifacts/
├── step_04_tokenize/
│   ├── train_tokens.parquet
│   ├── val_tokens.parquet
│   ├── fitted_thresholds.json
│   └── tokenize_artifact.json
├── step_05_sequences/
│   ├── train_X.pt  # (N, 24, 10, 2)
│   ├── train_y.pt  # (N, 8)
│   ├── val_X.pt
│   ├── val_y.pt
│   └── sequences_artifact.json
├── step_06_train/
│   ├── model.pt
│   ├── history.json
│   └── train_artifact.json
├── step_07_evaluate/
│   ├── eval_results.json
│   └── evaluate_artifact.json
└── step_08_inference/
    ├── predictions_{timestamp}.json
    └── inference_artifact.json
```

---

## Data Schemas

### TokenizeArtifact
```python
{
  "train_path": str,
  "val_path": str,
  "train_shape": tuple,
  "val_shape": tuple,
  "thresholds_path": str,
  "token_distribution": {
    0: {"train": float, "val": float},
    1: {"train": float, "val": float},
    2: {"train": float, "val": float}
  }
}
```

### SequencesArtifact
```python
{
  "train_X_path": str,
  "train_y_path": str,
  "val_X_path": str,
  "val_y_path": str,
  "train_num_samples": int,
  "val_num_samples": int,
  "input_length": 24,
  "output_length": 8,
  "num_coins": 10,
  "num_channels": 2,
  "target_coin": "XRP"
}
```

### TrainedModelArtifact
```python
{
  "model_path": str,
  "history_path": str,
  "best_val_loss": float,
  "best_val_acc": float,
  "total_epochs": int
}
```

### EvalReportArtifact
```python
{
  "per_hour_accuracy": [float] * 8,
  "sequence_accuracy": float,
  "baseline_results": {
    "random": float,
    "persistence": float
  }
}
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 24-hour input | Captures daily cycles, trains faster than 48h |
| 2 channels | Price + volume captures strength & direction |
| 3-class vocab | Balanced, interpretable, simple |
| Quantile thresholds | Auto-balanced classes, coin-adaptive |
| Fit on train only | Prevents data leakage |
| Decoder-only | Causal masking prevents future leakage |
| Teacher forcing | Stable training, fast convergence |
| 8-hour horizon | Practical for trading, measurable |

---

## Success Criteria

### Pipeline Validation
- ✅ Tokenization produces ~33/33/33 distribution on train
- ✅ No data leakage (val distribution may differ)
- ✅ All tensor shapes correct

### Model Performance
- 🎯 Hour-1 accuracy > 40%
- 🎯 Hour-8 accuracy > 35%
- 🎯 Sequence accuracy > 1%
- 🎯 Beats all baselines by ≥2%

### Deployment
- ✅ Inference < 100ms
- ✅ Model size < 500MB
- ✅ Reproducible results

---

## Known Limitations

1. **Accuracy degrades over horizon**: 8-hour predictions inherently uncertain
2. **No uncertainty calibration**: Softmax probabilities ≠ true confidence
3. **Macro event blindness**: Model can't handle news/crashes outside training distribution
4. **Single-coin output**: Only predicts XRP
5. **Fixed 8-hour horizon**: Not adaptive to user preferences

---

## Computational Requirements

- **Training**: 1 GPU (RTX 3080+), ~2-4 hours for 100 epochs, 2-4 GB GPU memory
- **Inference**: CPU sufficient, <50ms per prediction
- **Storage**: <100 MB total (data + model)

---

## Implementation Notes

1. **Anti-leakage**: Thresholds must be fit on train data only, then applied to val/inference
2. **Teacher forcing**: During training, use ground truth targets as input to decoder
3. **Autoregressive generation**: During eval/inference, generate tokens one-by-one
4. **Causal masking**: Position i can only attend to positions ≤ i (prevents future leakage)
5. **Channel handling**: Separate embeddings for price/volume, then fuse
6. **Target extraction**: Only XRP price channel used for targets, not volume
7. **Temporal separation**: Input = hours [0-23], Target = hours [24-31] (no overlap)

---

## CLI Interface

```bash
# Full pipeline
python main.py pipeline run-all

# Individual steps
python main.py pipeline tokenize
python main.py pipeline sequences
python main.py pipeline train
python main.py pipeline evaluate
python main.py pipeline inference
```

---

## Testing Requirements

1. **Unit tests**: Each pipeline block, model forward/generate
2. **Integration tests**: End-to-end pipeline execution
3. **Shape validation**: All tensor dimensions correct
4. **Leakage detection**: Val thresholds ≠ train thresholds
5. **Reproducibility**: Same seed → same results
