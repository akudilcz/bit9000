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

# Or run individual steps:
python main.py pipeline reset       # Clean artifacts
python main.py pipeline download    # Fetch OHLCV data
python main.py pipeline clean       # Clean and validate
python main.py pipeline split       # Train/val split
python main.py pipeline tokenize    # Convert prices → tokens
python main.py pipeline sequences   # Create input/output windows
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

The model uses a simple 8-step pipeline:

1. **Reset**: Clear previous artifacts
2. **Download**: Fetch hourly OHLCV data (10 coins, 2020-2025)
3. **Clean**: Fill gaps and validate data quality
4. **Split**: Temporal train/val split (80/20)
5. **Tokenize**: Convert prices to 3-class tokens (down=0, steady=1, up=2)
6. **Sequences**: Create rolling windows (24h input → 8h output)
7. **Train**: Train transformer decoder on token sequences
8. **Evaluate**: Compute accuracy and baseline comparisons
9. **Inference**: Real-time 8-hour XRP price predictions

## Artifacts

Pipeline outputs are saved to `artifacts/`:
- `step_00_reset/`: Artifact metadata from reset
- `step_01_download/`: Raw OHLCV data (parquet files + visualizations)
- `step_02_clean/`: Cleaned OHLCV data (gaps filled, quality metrics)
- `step_03_split/`: Train/validation split data (temporal split at 80/20)
- `step_04_tokenize/`: Tokenized sequences + fitted quantile thresholds
- `step_05_sequences/`: PyTorch tensor sequences (train_X.pt, train_y.pt, val_X.pt, val_y.pt)
- `step_06_train/`: Trained model checkpoint + training history and loss curves
- `step_07_evaluate/`: Evaluation metrics, accuracy per hour, confusion matrices
- `step_08_inference/`: Latest predictions with probabilities and confidence scores

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

