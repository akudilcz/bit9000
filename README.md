# CryptoTransformerV4 - High Precision Trading Signal Predictor

A precision-focused transformer model that predicts XRP price direction 1 hour ahead using comprehensive technical analysis across 10 cryptocurrencies. Optimized for high-precision BUY/SELL signals with minimal false positives.

## ✅ System Status

**Latest Results (2025-10-27):**
- ✅ **Architecture**: CryptoTransformerV4 (3-layer encoder, 2-layer decoder, 8 heads, d_model=256)
- ✅ **Binary Classification**: Directional prediction (BUY = price up, NO-BUY = flat/down)
- ✅ **Prediction Horizon**: 1 hour ahead (high precision, low noise)
- ✅ **Input Context**: 48 hours × 10 coins × 18 channels (price, volume + 16 technical indicators)
- ✅ **Technical Indicators**: Comprehensive suite (RSI, MACD, EMAs, Stochastic, ADX, ATR, OBV, VWAP, etc.)
- ✅ **Precision Focus**: Asymmetric loss with 7455x penalty for wrong BUY calls
- ✅ **Target Performance**: BUY precision >75% with ~1% signal rate
- ✅ **High Confidence**: Only predictions with >80% confidence threshold

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Run all steps (download → train → evaluate)
python main.py

# Or run individual steps:
python -m src.pipeline.orchestrator --step 0  # Reset
python -m src.pipeline.orchestrator --step 1  # Download
python -m src.pipeline.orchestrator --step 2  # Clean
python -m src.pipeline.orchestrator --step 3  # Split
python -m src.pipeline.orchestrator --step 4  # Augment (technical indicators)
python -m src.pipeline.orchestrator --step 5  # Tokenize
python -m src.pipeline.orchestrator --step 6  # Sequences
python -m src.pipeline.orchestrator --step 7  # Train
python -m src.pipeline.orchestrator --step 8  # Evaluate
python -m src.pipeline.orchestrator --step 9  # Inference
```

### 3. Check Results
```bash
# Training metrics
cat artifacts/step_07_train/history.json

# Evaluation metrics
cat artifacts/step_08_evaluate/eval_results.json

# Latest predictions
cat artifacts/step_09_inference/predictions.json
```

## Pipeline Overview

The model uses a 10-step pipeline optimized for high-precision BUY/NO-BUY signal prediction:

1. **Reset**: Clear previous artifacts
2. **Download**: Fetch hourly OHLCV data (10 coins: BTC, ETH, LTC, XRP, BNB, ADA, XLM, TRX, DOGE, DOT; 2018-2025)
3. **Clean**: Fill gaps, validate data quality, remove outliers
4. **Split**: Temporal train/val split (80/20, no shuffling to prevent leakage)
5. **Augment**: Add 16 comprehensive technical indicators per coin
6. **Tokenize**: Convert 18 channels to 256-bin tokens (quantile-based, uniform distribution)
7. **Sequences**: Create rolling windows (48h input × 10 coins × 18 channels → 1h ahead directional target)
8. **Train**: Train transformer with asymmetric precision-focused loss
9. **Evaluate**: Compute BUY precision, recall, F1, and confidence analysis
10. **Inference**: Real-time directional predictions with high-confidence filtering

## Architecture

**CryptoTransformerV4** - Encoder-decoder transformer with specialized pathways and precision-focused training:

- **Encoder**: Shared multi-coin context processor (3 layers, 8 heads)
  - Processes all 10 coins jointly with 18 channels each
  - Input: 48h × 10 coins × 18 channels (price, volume + 16 technical indicators)
  - Output: Rich contextualized representations
  
- **Decoder**: XRP-specific decoder (2 layers, cross-attention to encoder)
  - BTC→XRP attention pathway leverages Bitcoin's market leadership
  - Time features: Cyclical hour/day encoding + sequence position
  
- **Prediction Head**: Binary classification (2 classes: NO-BUY=0, BUY=1)
  - Output: Logits for softmax → probabilities
  - High-confidence threshold: >80% confidence required for signals
  
- **Classification Logic**: Directional
  - BUY (class 1): future_price_token > current_price_token (price rises)
  - NO-BUY (class 0): future_price_token <= current_price_token (flat/down)
  - Natural interpretation: "Buy when price will go up"

**Model Dimensions**:
- d_model: 256
- nhead: 8
- num_encoder_layers: 3
- num_decoder_layers: 2
- dim_feedforward: 1024
- dropout: 0.2
- Parameters: ~2.1M

**Loss Function**: AsymmetricPrecisionLoss
- buy_penalty: 10.0 (penalty for wrong BUY predictions)
- no_buy_penalty: 1.0 (penalty for wrong NO-BUY predictions)
- **7455x higher penalty** for wrong BUY calls vs correct BUY calls
- Purpose: Maximize precision by heavily penalizing false positives

**Regularization** (balanced for precision):
- Dropout: 0.2
- Weight decay: 0.0001
- Label smoothing: 0.01
- Gaussian noise: 0.05
- Gradient clipping: max_norm=1.0
- Warmup: 3 epochs (1e-5 → 1.5e-3 LR)
- Early stopping: patience=30 epochs, precision-based stopping

**Tokenization**: 256-bin quantile-based encoding
- Converts continuous price/volume/indicators to discrete tokens
- Quantile binning ensures uniform distribution across 0-255
- Fitted on training data only (no leakage)

## Technical Indicators

### Comprehensive Indicator Suite (16 per coin)

**Momentum Indicators**:
- **RSI** (14-period): Overbought/oversold conditions (0-100)
- **Stochastic** (14-period): Price momentum relative to recent range (0-100)
- **Williams %R** (14-period): Momentum oscillator (-100 to 0)
- **Price Momentum** (10-period): Rate of price change (percentage)

**Trend Indicators**:
- **MACD**: Moving Average Convergence Divergence histogram
- **EMA-9, EMA-21, EMA-50**: Multi-timeframe exponential moving averages
- **EMA Ratio**: Fast/slow EMA relationship (0-1 normalized)
- **ADX** (14-period): Average Directional Index for trend strength (0-100)

**Volatility Indicators**:
- **Bollinger Band Position**: Price position within volatility bands (0-1)
- **ATR** (14-period): Average True Range for volatility measurement
- **Volatility Regime**: Current volatility percentile vs historical (0-1)

**Volume Indicators**:
- **OBV**: On-Balance Volume momentum indicator
- **Volume ROC** (10-period): Volume rate of change (percentage)
- **VWAP** (20-period): Volume Weighted Average Price

**Market Structure Indicators**:
- **Support/Resistance Strength**: Price clustering analysis (0-1)

### Normalization Strategy
All indicators normalized to [0,1] range for consistent tokenization:
- **Percentage-based**: RSI, Stochastic, ADX scaled by 100
- **Range-based**: Williams %R offset and scaled
- **Rolling normalization**: ATR, OBV normalized by rolling statistics
- **Price-relative**: EMAs, VWAP normalized relative to current price
- **Clipped ranges**: Volume ROC, Price Momentum clipped to reasonable ranges

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **BUY Signal Rate** | ~1% | Very selective for high precision |
| **Target BUY Precision** | >75% | When we say BUY, we're right >75% |
| **Prediction Horizon** | 1 hour | High precision, low noise |
| **Input Context** | 48 hours | 2 days of history |
| **Channels per Coin** | 18 | Price, volume + 16 technical indicators |
| **Total Features** | 180 | 10 coins × 18 channels |
| **Model Parameters** | ~2.1M | 3-layer encoder + 2-layer decoder |
| **Training Time** | ~15-20 min | 50 epochs on single GPU |
| **Confidence Threshold** | >80% | Only high-confidence predictions |

**Key Innovation**: Asymmetric loss function with 7455x penalty for wrong BUY calls ensures extremely conservative predictions, maximizing precision at the cost of recall.

## Loss Functions

### AsymmetricPrecisionLoss (Active)
**Purpose**: Heavily penalize wrong BUY calls to maximize precision  
**Penalty Ratio**: 7455x higher penalty for wrong BUY vs correct BUY  
**Parameters**:
- `buy_penalty`: 10.0 (penalty for wrong BUY predictions)
- `no_buy_penalty`: 1.0 (penalty for wrong NO-BUY predictions)

### Alternative Loss Functions
- **PrecisionFocusedLoss**: Configurable precision vs recall weighting
- **ConfidenceWeightedLoss**: Penalizes high-confidence wrong predictions
- **PrecisionRecallLoss**: Direct F-beta optimization (β=0.5 emphasizes precision)

## Artifacts

Pipeline outputs are saved to `artifacts/`:
- `step_00_reset/`: Artifact metadata from reset
- `step_01_download/`: Raw OHLCV data (parquet files + visualizations)
- `step_02_clean/`: Cleaned OHLCV data (gaps filled, quality metrics)
- `step_03_split/`: Train/validation split data (temporal split at 80/20)
- `step_04_augment/`: Augmented data with 16 technical indicators per coin
- `step_05_tokenize/`: Tokenized sequences + fitted 256-bin thresholds
- `step_06_sequences/`: PyTorch tensor sequences (train_X.pt, train_y.pt, val_X.pt, val_y.pt)
- `step_07_train/`: Trained model checkpoint + training history and precision metrics
- `step_08_evaluate/`: Evaluation metrics, precision/recall/F1 analysis, confusion matrices
- `step_09_inference/`: Latest predictions with confidence scores and precision analysis

## Development

### Project Structure
```
bit9000/
├── main.py                           # CLI entry point
├── config.yaml                       # Configuration file
├── requirements.txt                  # Dependencies
├── DESIGN.md                         # Technical design document
├── README.md                         # This file
│
├── src/
│   ├── model/
│   │   ├── __init__.py              # Model factory
│   │   ├── trainer.py               # Training logic
│   │   └── v4_transformer.py        # CryptoTransformerV4 architecture
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py           # Pipeline orchestration
│   │   ├── schemas.py               # Data schemas (Pydantic)
│   │   │
│   │   ├── step_00_reset/           # Clear artifacts
│   │   ├── step_01_download/        # Fetch OHLCV data
│   │   ├── step_02_clean/           # Data cleaning
│   │   ├── step_03_split/           # Train/val split
│   │   ├── step_04_augment/         # Add technical indicators
│   │   ├── step_05_tokenize/        # Price → tokens
│   │   ├── step_06_sequences/       # Create rolling windows
│   │   ├── step_07_train/           # Model training
│   │   ├── step_08_evaluate/        # Validation & metrics
│   │   └── step_09_inference/       # Real-time prediction
│   │
│   └── utils/
│       ├── technical_indicators.py  # 16 comprehensive indicators
│       ├── precision_loss.py        # Precision-focused loss functions
│       ├── ordinal_loss.py          # Ordinal-aware loss functions
│       └── logger.py                # Logging utilities
│
├── tests/
│   ├── model/                       # Model tests
│   ├── pipeline/                    # Pipeline tests
│   └── utils/                       # Utility tests
│
└── artifacts/                       # Pipeline outputs (gitignored)
    ├── step_00_reset/ through step_09_inference/
    └── checkpoints/                 # Model checkpoints
```

### Testing
```bash
# Test technical indicators
python -c "from src.utils.technical_indicators import add_technical_indicators; print('OK')"

# Test precision loss
python -c "from src.utils.precision_loss import AsymmetricPrecisionLoss; print('OK')"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## Configuration

### Key Settings (config.yaml)
```yaml
# Data
data:
  coins: [BTC, ETH, LTC, XRP, BNB, ADA, XLM, TRX, DOGE, DOT]
  target_coin: XRP
  interval: 1h

# Sequences
sequences:
  input_length: 48            # 48 hours of context
  num_channels: 18           # 18 channels per coin
  prediction_horizon: 1      # 1 hour ahead

# Model
model:
  d_model: 256
  nhead: 8
  num_encoder_layers: 3
  num_decoder_layers: 2
  dropout: 0.2

# Training (precision-focused)
training:
  loss_type: asymmetric
  buy_penalty: 10.0
  no_buy_penalty: 1.0
  epochs: 50
  learning_rate: 0.0015

# Inference (high confidence)
inference:
  target_signal_rate: 0.01
  confidence_threshold: 0.8
  precision_target: 0.75
```

## Documentation

- **DESIGN.md**: Complete technical design, architecture, and data flow
- **This README**: Quick start and project overview

## License

MIT