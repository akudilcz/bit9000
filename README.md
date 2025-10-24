# Simple Multi-Coin Token Predictor

A lightweight transformer model that predicts XRP price movements over the next 8 hours using multi-coin context. Uses 256-bin continuous price quantization and autoregressive generation.

## ✅ System Status

**Latest Results (2025-10-24):**
- ✅ **Training complete** - Best validation loss: **1.0759** (54 epochs)
- ✅ **Model**: 471,808 parameters (2 layers, d_model=128, embedding_dim=32)
- ✅ **Evaluation**: **42.03% accuracy** (beats persistence baseline 35.83%)
- ✅ **Inference**: Working - Real-time predictions with probabilities
- ✅ **Hyperparameter tuning**: 40 trials completed (best val_loss=1.0776)

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Run all steps (download → train → evaluate)
python main.py pipeline run-all

# Or run individual steps:
python main.py pipeline reset       # Clean artifacts
python main.py pipeline download    # Fetch OHLCV data
python main.py pipeline clean       # Clean and validate
python main.py pipeline split       # Train/val split
python main.py pipeline tokenize    # Convert prices → 256-bin tokens
python main.py pipeline sequences   # Create input/output windows
python main.py pipeline train       # Train transformer
python main.py pipeline evaluate    # Measure accuracy
python main.py pipeline inference   # Predict next 8 hours
```

### 3. Hyperparameter Tuning
```bash
# Quick tune (2 trials, 2 epochs each, ~1 min)
python main.py pipeline tune --num-trials 2 --epochs-per-trial 2

# Full tune (40 trials, 20 epochs each, ~90 min)
python main.py pipeline tune --num-trials 40 --epochs-per-trial 20
```

### 4. Check Results
```bash
# Training metrics
cat artifacts/step_06_train/history.json

# Evaluation metrics
cat artifacts/step_07_evaluate/eval_results.json

# Latest predictions
cat artifacts/step_08_inference/predictions_*.json
```

## Pipeline Overview

The model uses a 9-step pipeline:

1. **Reset**: Clear previous artifacts
2. **Download**: Fetch hourly OHLCV data (10 coins, 2020-2025)
3. **Clean**: Fill gaps and validate data quality
4. **Split**: Temporal train/val split (80/20)
5. **Tokenize**: Convert prices to 256-bin tokens (quantile-based, uniform distribution)
6. **Sequences**: Create rolling windows (24h input → 1h output, autoregressively generated to 8h)
7. **Train**: Train transformer decoder on token sequences
8. **Evaluate**: Compute accuracy and baseline comparisons
9. **Inference**: Real-time 8-hour XRP price predictions (autoregressive generation)

## Architecture

**Model**: `SimpleTokenPredictor` (Transformer Decoder-Only)
- **Input**: 24 hours × 10 coins × 2 channels (price + volume)
- **Output**: 1 hour next-token prediction (256 classes), autoregressively generated to 8 hours
- **Layers**: 2 transformer decoder layers
- **Heads**: 4 attention heads
- **Model dim**: 128
- **Embedding dim**: 32
- **Feedforward dim**: 256
- **Parameters**: 471,808

**Tokenization**: 256-bin quantization
- Bins: 0-255 representing continuous price range
- Method: Quantile-based (uniform distribution across bins)
- Input: Price and volume for each coin

## Performance

| Metric | Value |
|--------|-------|
| **Best val loss** | 1.0759 |
| **Evaluation accuracy** | 42.03% |
| **Persistence baseline** | 35.83% |
| **Random baseline** | 34.51% |
| **Model improvement** | +6.2% over persistence |
| **Training time** | ~1 min (54 epochs) |

## Artifacts

Pipeline outputs are saved to `artifacts/`:
- `step_00_reset/`: Artifact metadata from reset
- `step_01_download/`: Raw OHLCV data (parquet files + visualizations)
- `step_02_clean/`: Cleaned OHLCV data (gaps filled, quality metrics)
- `step_03_split/`: Train/validation split data (temporal split at 80/20)
- `step_04_tokenize/`: Tokenized sequences + fitted 256-bin thresholds
- `step_05_sequences/`: PyTorch tensor sequences (train_X.pt, train_y.pt, val_X.pt, val_y.pt)
- `step_06_train/`: Trained model checkpoint + training history and loss curves
- `step_07_evaluate/`: Evaluation metrics, accuracy plots, confusion matrices
- `step_08_inference/`: Latest predictions with probabilities and confidence scores
- `tuning/`: Hyperparameter tuning results and visualizations

## Development

### Project Structure
```
bit9000/
├── main.py                           # CLI entry point
├── config.yaml                       # Configuration file
├── test_config.yaml                  # Test configuration
├── requirements.txt                  # Dependencies
├── pyproject.toml                    # Project metadata
├── DESIGN.md                         # Technical design document
├── README.md                         # This file
│
├── src/
│   ├── model/
│   │   ├── token_predictor.py       # SimpleTokenPredictor architecture
│   │   └── trainer.py               # Training logic
│   │
│   ├── pipeline/
│   │   ├── base.py                  # BasePipelineBlock (abstract)
│   │   ├── io.py                    # Artifact I/O utilities
│   │   ├── schemas.py               # Data schemas (Pydantic)
│   │   │
│   │   ├── step_00_reset/           # Clear artifacts
│   │   ├── step_01_download/        # Fetch OHLCV data
│   │   ├── step_02_clean/           # Data cleaning
│   │   ├── step_03_split/           # Train/val split
│   │   ├── step_04_tokenize/        # Price → tokens
│   │   ├── step_05_sequences/       # Create rolling windows
│   │   ├── step_06_train/           # Model training
│   │   ├── step_07_evaluate/        # Validation & metrics
│   │   └── step_08_inference/       # Real-time prediction
│   │
│   ├── system/
│   │   ├── pretrain.py              # Pretraining utilities
│   │   ├── tune.py                  # Fine-tuning utilities
│   │   └── README.md                # System documentation
│   │
│   └── utils/
│       ├── config_validator.py      # Config validation
│       ├── logger.py                # Logging utilities
│       ├── metrics.py               # Performance metrics
│       └── plot_utils.py            # Visualization utilities
│
├── tests/
│   ├── conftest.py                  # Pytest fixtures
│   ├── model/                       # Model tests
│   ├── pipeline/                    # Pipeline tests (unit + integration)
│   ├── utils/                       # Utility tests
│   └── data/                        # Test fixtures
│
└── artifacts/                       # Pipeline outputs (gitignored)
    ├── step_00_reset/ through step_08_inference/
    ├── checkpoints/                 # Model checkpoints
    └── lightning_logs/              # Training logs
```

### Testing
```bash
# Test imports
python -c "from src.model.token_predictor import SimpleTokenPredictor; print('OK')"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## Documentation

- **DESIGN.md**: Complete technical design, architecture, and data flow
- **This README**: Quick start and project overview

## License

MIT

