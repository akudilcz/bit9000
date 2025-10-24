# Simple Multi-Coin Token Predictor

A lightweight transformer model that predicts XRP price movements (down/steady/up) over the next 8 hours using multi-coin context.

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Run all steps (download → train → evaluate)
python main.py pipeline run-all

# Run from clean step (skips reset and download - saves time!)
python main.py pipeline run-from-clean

# Or run individual steps:
python main.py pipeline reset       # Clean artifacts
python main.py pipeline download    # Fetch data (2020-2025)
python main.py pipeline clean       # Clean and validate
python main.py pipeline split       # Train/val split (80/20)
python main.py pipeline tokenize    # Convert prices → tokens
python main.py pipeline sequences   # Create 48h → 8h windows
python main.py pipeline train       # Train transformer
python main.py pipeline evaluate    # Measure accuracy
python main.py pipeline inference   # Predict next 8 hours
```

### 3. Check Results
```bash
# Training metrics
cat artifacts/step_06_train/history.json

# Evaluation metrics
cat artifacts/step_07_evaluate/eval_results.json

# Latest predictions
ls -lt artifacts/step_08_inference/
```

## Pipeline Overview

**Philosophy**: Simplicity over complexity
- No feature engineering, no complex preprocessing
- Just raw price movements → discrete tokens → predictions

```
Step 1: Download   → Fetch hourly OHLCV (10 coins, 2020-2025)
Step 2: Clean      → Fill gaps, validate quality
Step 3: Split      → Temporal train/val split (80/20)
Step 4: Tokenize   → Convert prices to tokens (down=0, steady=1, up=2)
Step 5: Sequences  → Create 48h input → 8h output windows
Step 6: Train      → Train transformer on token sequences
Step 7: Evaluate   → Per-hour accuracy, confusion matrices
Step 8: Inference  → Predict next 8 hours
```

## Model Architecture

**Input**: 48 tokens × 10 coins (2 days of hourly movements)
**Output**: 8 tokens (next 8 hours of XRP price direction)
**Vocabulary**: 3 tokens {down=0, steady=1, up=2}

### Architecture
- Token Embedding (3 → d_model)
- Coin Aggregation (mean pooling across coins)
- Positional Encoding (sinusoidal)
- Transformer Encoder (4 layers, 8 heads)
- Prediction Head (8 hours × 3 classes)

### Configuration
Edit `config.yaml` to adjust:
- `sequences.input_length`: 48 hours (default)
- `sequences.output_length`: 8 hours (default)
- `model.d_model`: 256 (model dimension)
- `model.nhead`: 8 (attention heads)
- `model.num_layers`: 4 (transformer layers)
- `training.epochs`: 100
- `training.batch_size`: 128
- `training.learning_rate`: 0.0001

## Expected Performance

| Metric | Target | Random Baseline |
|--------|--------|-----------------|
| Hour 1 Accuracy | ~45% | 33% |
| Hour 8 Accuracy | ~37% | 33% |
| Mean Accuracy | ~40% | 33% |
| Sequence Accuracy | ~1-2% | 0.015% |

## Design Principles

1. **Simplicity**: Raw movements → tokens → predictions
2. **Balanced classes**: Quantile thresholds (33/33/33 distribution)
3. **Multi-coin context**: BTC/ETH patterns inform XRP predictions
4. **No data leakage**: Thresholds fit on training data only
5. **Short horizon**: 8 hours is predictable and actionable

## Artifacts

All outputs saved to `artifacts/`:
- `step_04_tokenize/`: Tokens + thresholds JSON
- `step_05_sequences/`: PyTorch tensors (train_X.pt, etc.)
- `step_06_train/`: Model checkpoint + training history
- `step_07_evaluate/`: Metrics + confusion matrices
- `step_08_inference/`: Prediction JSON with probabilities

## Development

### Project Structure
```
Bit3/
├── main.py                    # CLI entry point
├── config.yaml                # Configuration
├── DESIGN.md                  # Design document
├── IMPLEMENTATION.md          # Implementation details
├── src/
│   ├── pipeline/
│   │   ├── step_04_tokenize/  # Tokenization
│   │   ├── step_05_sequences/ # Sequence creation
│   │   ├── step_06_train/     # Training
│   │   ├── step_07_evaluate/  # Evaluation
│   │   └── step_08_inference/ # Inference
│   └── model/
│       └── token_predictor.py # Simple token predictor
└── artifacts/                 # Pipeline outputs
```

### Testing
```bash
# Test imports
python -c "from src.model.token_predictor import SimpleTokenPredictor; print('OK')"

# Test tokenization only
python main.py pipeline reset
python main.py pipeline download --end-date 2021-01-01  # Small dataset
python main.py pipeline clean
python main.py pipeline split
python main.py pipeline tokenize

# Check tokenization results
cat artifacts/step_04_tokenize/tokenize_artifact.json
```

## Documentation

- **DESIGN.md**: Complete design philosophy and architecture
- **IMPLEMENTATION.md**: Implementation details and decisions
- **This README**: Quick start and usage guide

## License

MIT

