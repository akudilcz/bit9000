# Simple Multi-Coin Token Predictor

A lightweight transformer model that predicts XRP price direction 8 hours ahead using multi-coin context. Directional binary classification (BUY if price rises, NO-BUY if flat/down) with post-training threshold calibration for ~5% signal frequency.

## ✅ System Status

**Latest Results (2025-10-26):**
- ✅ **Architecture**: CryptoTransformerV4 (8-layer encoder, 8-layer XRP decoder, 4 heads, d_model=208)
- ✅ **Binary Classification**: Directional prediction (BUY = price up, NO-BUY = flat/down)
- ✅ **Prediction Horizon**: 8 hours ahead (less noise, more directional)
- ✅ **Input Context**: 48 hours × 10 coins (BTC, ETH, LTC, XRP, BNB, ADA, XLM, TRX, DOGE, DOT) × 9 channels (price, volume, RSI, MACD, BB position, EMA-9, EMA-21, EMA-50, EMA-ratio)
- ✅ **Calibration**: Post-training threshold calibration achieving ~5% BUY signal rate
- ✅ **Target Performance**: BUY precision >65% (when we say BUY, we're right >65% of the time)
- ✅ **Regularization**: Aggressive (dropout=0.5, weight_decay=0.2, warmup=15 epochs, label_smoothing=0.12)
- ✅ **Training**: 200 epochs, ~64K samples, binary CrossEntropyLoss with pos_weight adaptation

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
python main.py pipeline augment     # Add technical indicators
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
cat artifacts/step_07_train/history.json

# Evaluation metrics
cat artifacts/step_08_evaluate/eval_results.json

# Latest predictions
cat artifacts/step_09_inference/predictions_*.json
```

## Pipeline Overview

The model uses a 10-step pipeline optimized for directional BUY/NO-BUY signal prediction:

1. **Reset**: Clear previous artifacts
2. **Download**: Fetch hourly OHLCV data (10 coins: BTC, ETH, LTC, XRP, BNB, ADA, XLM, TRX, DOGE, DOT; 2018-2025) + trim to common date range
3. **Clean**: Fill gaps, validate data quality
4. **Split**: Temporal train/val split (80/20, no shuffling to prevent leakage)
5. **Augment**: Add technical indicators (RSI, MACD, Bollinger Bands, EMAs)
6. **Tokenize**: Convert 9 channels to 256-bin tokens (quantile-based, uniform distribution)
7. **Sequences**: Create rolling windows (48h input × 10 coins × 9 channels → 8h ahead directional target)
8. **Train**: Train transformer with binary classification (BUY if future_price > current_price)
9. **Evaluate**: Compute BUY precision and signal rate via post-training calibration
10. **Inference**: Real-time directional predictions with calibrated threshold (~5% signal rate)

## Architecture

**CryptoTransformerV4** - Encoder-decoder transformer with dedicated pathways and binary classification:
- **Encoder**: Shared multi-coin context processor (8 layers, 4 heads)
  - Processes all 10 coins (BTC, ETH, LTC, XRP, BNB, ADA, XLM, TRX, DOGE, DOT) jointly
  - Input: 48h × 10 coins × 9 channels (price, volume, RSI, MACD, BB_position, EMA-9, EMA-21, EMA-50, EMA-ratio)
  - Output: Rich contextualized representations
  
- **Decoders**: 
  - BTC encoder (2 layers, dedicated pathway since BTC leads altcoins)
  - XRP decoder (8 layers, cross-attention to shared encoder + BTC encoder)
  - Time features: Cyclical hour/day encoding + sequence position
  
- **Prediction Head**: Binary classification (2 classes: NO-BUY=0, BUY=1)
  - Output: Logits for softmax → probabilities
  - Calibrated threshold: Finds optimal cutoff for ~5% signal rate
  
- **Classification Logic**: Directional
  - BUY (class 1): future_price_token > current_price_token (price rises)
  - NO-BUY (class 0): future_price_token <= current_price_token (flat/down)
  - Natural interpretation: "Buy when price will go up"

**Model Dimensions**:
- d_model: 208
- nhead: 4
- num_encoder_layers: 8
- num_decoder_layers: 8  
- dim_feedforward: 768
- dropout: 0.5
- Parameters: ~10.5M

**Loss Function**: CrossEntropyLoss with class weighting
- pos_weight: Adaptively scaled based on class imbalance
- Purpose: Handle imbalanced BUY/NO-BUY ratio

**Regularization** (aggressive, for small dataset):
- Dropout: 0.5
- Weight decay: 0.2
- Label smoothing: 0.12
- Gaussian noise: 0.025
- Gradient clipping: max_norm=0.5
- Warmup: 15 epochs (1e-5 → 1.5e-4 LR)
- Early stopping: patience=15 epochs, min_delta=0.0001

**Tokenization**: 256-bin quantile-based encoding
- Converts continuous price/volume/indicators to discrete tokens
- Quantile binning ensures uniform distribution across 0-255
- Fitted on training data only (no leakage)

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **BUY Signal Rate** | ~5% | Calibrated on validation set |
| **Target BUY Precision** | >65% | When we say BUY, we're right >65% |
| **Prediction Horizon** | 8 hours | Directional (UP/DOWN) |
| **Input Context** | 48 hours | 2 days of history |
| **Data Points** | ~64K train, ~4K val | Temporal split 80/20 |
| **Model Parameters** | ~10.5M | 8-layer encoder + decoder |
| **Training Time** | ~30-40 min | 200 epochs on single GPU |
| **Calibration Method** | Post-training | Finds threshold for target signal rate |

**Key Innovation**: Post-training threshold calibration automatically adjusts decision boundary to achieve target signal frequency while maximizing precision, enabling practical trading signal generation.

## Artifacts

Pipeline outputs are saved to `artifacts/`:
- `step_00_reset/`: Artifact metadata from reset
- `step_01_download/`: Raw OHLCV data (parquet files + visualizations) + trimmed to common date range
- `step_02_clean/`: Cleaned OHLCV data (gaps filled, quality metrics)
- `step_03_split/`: Train/validation split data (temporal split at 80/20)
- `step_04_augment/`: Augmented data with technical indicators (RSI, MACD, BB, EMAs)
- `step_05_tokenize/`: Tokenized sequences + fitted 256-bin thresholds
- `step_06_sequences/`: PyTorch tensor sequences (train_X.pt, train_y.pt, val_X.pt, val_y.pt)
- `step_07_train/`: Trained model checkpoint + training history and loss curves
- `step_08_evaluate/`: Evaluation metrics, accuracy plots, confusion matrices
- `step_09_inference/`: Latest predictions with probabilities and confidence scores
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
│   │   ├── step_01_download/        # Fetch OHLCV data + trim to common dates
│   │   ├── step_02_clean/           # Data cleaning
│   │   ├── step_03_split/           # Train/val split
│   │   ├── step_04_augment/         # Add technical indicators
│   │   ├── step_05_tokenize/        # Price → tokens
│   │   ├── step_06_sequences/       # Create rolling windows
│   │   ├── step_07_train/           # Model training
│   │   ├── step_08_evaluate/        # Validation & metrics
│   │   └── step_09_inference/       # Real-time prediction
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
    ├── step_00_reset/ through step_09_inference/
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

