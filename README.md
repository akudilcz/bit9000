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
│   │   ├── step_00_reset/
│   │   │   └── reset_block.py       # Clear artifacts
│   │   ├── step_01_download/
│   │   │   ├── data_collector.py    # Fetch OHLCV data
│   │   │   └── download_block.py    # Download pipeline step
│   │   ├── step_02_clean/
│   │   │   ├── clean_block.py       # Data cleaning
│   │   │   └── data_processor.py    # Preprocessing utilities
│   │   ├── step_03_split/
│   │   │   └── split_block.py       # Train/val split
│   │   ├── step_04_tokenize/
│   │   │   └── tokenize_block.py    # Price → tokens
│   │   ├── step_05_sequences/
│   │   │   └── sequence_block.py    # Create rolling windows
│   │   ├── step_06_train/
│   │   │   └── train_block.py       # Model training
│   │   ├── step_07_evaluate/
│   │   │   └── evaluate_block.py    # Validation & metrics
│   │   └── step_08_inference/
│   │       └── inference_block.py   # Real-time prediction
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
│   ├── conftest.py                  # Pytest fixtures & configuration
│   │
│   ├── model/
│   │   └── test_token_predictor.py  # Model unit tests
│   │
│   ├── pipeline/
│   │   ├── unit/                    # Unit tests
│   │   │   ├── test_artifact_io.py
│   │   │   ├── test_clean_block.py
│   │   │   ├── test_download_block.py
│   │   │   ├── test_pipeline_base.py
│   │   │   ├── test_schemas_validation.py
│   │   │   └── quality/
│   │   │       └── test_clean_quality.py
│   │   ├── clean/                   # Integration tests
│   │   │   ├── test_clean_block.py
│   │   │   └── test_clean_quality.py
│   │   ├── download/
│   │   │   └── test_download_block.py
│   │   ├── sequences/
│   │   │   └── test_sequence_block.py
│   │   ├── tokenize/
│   │   │   └── test_tokenize_block.py
│   │   └── integration/             # End-to-end tests
│   │       (reserved for full pipeline tests)
│   │
│   ├── utils/
│   │   ├── test_logger.py
│   │   ├── test_logger_edge_cases.py
│   │   └── test_metrics.py
│   │
│   └── data/                        # Test fixtures
│
└── artifacts/                       # Pipeline outputs (gitignored)
    ├── step_00_reset/
    ├── step_01_download/
    ├── step_02_clean/
    ├── step_03_split/
    ├── step_04_tokenize/           # tokens + thresholds
    ├── step_05_sequences/          # tensor sequences
    ├── step_06_train/              # model checkpoint
    ├── step_07_evaluate/           # metrics & confusion matrices
    ├── step_08_inference/          # predictions
    ├── checkpoints/                # PyTorch Lightning checkpoints
    └── lightning_logs/             # Training logs
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

