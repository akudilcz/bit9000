# CryptoTransformerV4 - High Precision Trading Signal Predictor

## Overview

**Objective**: Train a precision-focused transformer to predict XRP price direction 1 hour ahead using comprehensive technical analysis across 10 cryptocurrencies. Optimized for high-precision BUY/SELL signals with minimal false positives.

**Input**: `(48, 10, 18)` - 48 hours × 10 coins × 18 channels (price, volume + 16 technical indicators)  
**Output**: `(1,)` - Next hour XRP price direction (BUY/NO-BUY binary classification)  
**Architecture**: Encoder-decoder transformer with BTC→XRP attention pathway  
**Target**: >75% precision on BUY signals with ~1% signal rate

---

## Data Flow Summary

```
Step 0: Reset
  └─> Clean artifacts directory
      Input: None
      Output: Clean artifacts/ directory

Step 1: Download
  └─> Fetch OHLCV data from Binance API
      Input: None (uses config dates: 2018-05-05 to 2025-10-25)
      Output: raw_data.parquet [T × (10 coins × 5 OHLCV columns)]
      Contribution: Raw market data collection

Step 2: Clean
  └─> Fill gaps, remove outliers, validate data quality
      Input: raw_data.parquet [T × (10 coins × 5 OHLCV)]
      Output: clean_data.parquet [T × (10 coins × 5 OHLCV)]
      Contribution: Data quality assurance, outlier removal

Step 3: Split
  └─> Temporal train/validation split (80%/20%)
      Input: clean_data.parquet [T × (10 coins × 5 OHLCV)]
      Output: train_clean.parquet [T_train × (10 coins × 5)]
             val_clean.parquet [T_val × (10 coins × 5)]
      Contribution: Prevents data leakage, ensures temporal separation

Step 4: Augment
  └─> Add comprehensive technical indicators
      Input: train_clean.parquet, val_clean.parquet [T × (10 coins × 5 OHLCV)]
      Output: train_augmented.parquet, val_augmented.parquet [T × (10 coins × 23 features)]
      Contribution: 16 technical indicators per coin (RSI, MACD, EMAs, Stochastic, ADX, ATR, OBV, VWAP, etc.)

Step 5: Tokenize
  └─> Convert continuous values to 256-bin tokens
      Input: train_augmented.parquet, val_augmented.parquet [T × (10 coins × 23 features)]
      Output: train_tokens.parquet, val_tokens.parquet [T × (10 coins × 18 channels)]
             fitted_thresholds.json (quantile bin edges)
      Contribution: Quantile-based tokenization, uniform distribution, no data leakage

Step 6: Sequences
  └─> Create rolling window sequences for supervised learning
      Input: train_tokens.parquet, val_tokens.parquet [T × (10 coins × 18 channels)]
      Output: train_X.pt [(N_train, 48, 10, 18)] - input sequences
             train_y.pt [(N_train,)] - binary targets (BUY/NO-BUY)
             val_X.pt [(N_val, 48, 10, 18)]
             val_y.pt [(N_val,)]
      Contribution: Sliding window sequences, binary classification targets

Step 7: Train
  └─> Train CryptoTransformerV4 with precision-focused loss
      Input: train_X.pt, train_y.pt, val_X.pt, val_y.pt
      Output: model.pt (best checkpoint)
             history.json (training metrics)
             train_artifact.json (training results)
      Contribution: Asymmetric loss (7455x penalty for wrong BUY calls), precision-based early stopping

Step 8: Evaluate
  └─> Comprehensive validation with precision metrics
      Input: model.pt, val_X.pt, val_y.pt
      Output: eval_results.json (precision, recall, F1, confidence analysis)
             confusion_matrices/ (per-confidence-level analysis)
      Contribution: Precision-focused evaluation, confidence analysis, signal quality metrics

Step 9: Inference
  └─> Real-time prediction with confidence filtering
      Input: Latest OHLCV data
      Output: predictions.json {signal: BUY/NO-BUY, confidence: 0-1, timestamp}
      Contribution: High-confidence predictions only (>80% confidence), precision-focused calibration
```

**Key Transformations**:
- OHLCV → Technical Indicators: 16 comprehensive indicators per coin (momentum, trend, volatility, volume, market structure)
- Continuous → Tokens: Quantile-based binning to 256 tokens per channel for uniform distribution
- Tokens → Sequences: 48-hour sliding window → `(48h, 10 coins, 18 ch)` inputs, binary BUY/NO-BUY target
- Training: Asymmetric loss with 7455x penalty for wrong BUY predictions
- Inference: High-confidence predictions only (>80% confidence threshold)

---

## Design Principles

1. **Precision-First**: Optimize for high precision (>75%) over high recall, minimizing false BUY signals
2. **Comprehensive Analysis**: 18 channels per coin covering momentum, trend, volatility, volume, and market structure
3. **Binary Classification**: Simple BUY/NO-BUY decision for clear trading signals
4. **Asymmetric Loss**: Heavily penalize wrong BUY calls (7455x penalty) to ensure conservative predictions
5. **High Confidence**: Only issue signals with >80% confidence to maximize precision
6. **Temporal Integrity**: 48-hour context window with strict temporal train/validation split
7. **Multi-Coin Context**: BTC→XRP attention pathway leverages Bitcoin's market leadership
8. **No Data Leakage**: All preprocessing (binning, normalization) fit on training data only

---

## Pipeline Steps

### Step 0: Reset
**Purpose**: Clear all previous pipeline artifacts and start fresh  
**Input**: None  
**Output**: Clean artifacts/ directory  
**Contribution**: Ensures reproducible pipeline runs from scratch

### Step 1: Download
**Purpose**: Fetch historical OHLCV data for 10 cryptocurrencies from Binance API  
**Input**: None (uses config dates: 2018-05-05 to 2025-10-25)  
**Output**: raw_data.parquet [T × (10 coins × 5 OHLCV columns)]  
**Contribution**: Raw market data collection with quality validation

### Step 2: Clean
**Purpose**: Fill gaps, remove outliers, validate data quality  
**Input**: raw_data.parquet [T × (10 coins × 5 OHLCV)]  
**Output**: clean_data.parquet [T × (10 coins × 5 OHLCV)]  
**Contribution**: Data quality assurance, outlier removal, gap filling

### Step 3: Split
**Purpose**: Temporal train/validation split (80%/20%)  
**Input**: clean_data.parquet [T × (10 coins × 5 OHLCV)]  
**Output**: train_clean.parquet [T_train × (10 coins × 5)]  
         val_clean.parquet [T_val × (10 coins × 5)]  
**Contribution**: Prevents data leakage, ensures temporal separation

### Step 4: Augment
**Purpose**: Add comprehensive technical indicators for enhanced signal prediction  
**Input**: train_clean.parquet, val_clean.parquet [T × (10 coins × 5 OHLCV)]  
**Output**: train_augmented.parquet, val_augmented.parquet [T × (10 coins × 23 features)]  
**Contribution**: 16 technical indicators per coin:
- **Momentum**: RSI, Stochastic, Williams %R, Price Momentum
- **Trend**: MACD, EMA-9/21/50, EMA Ratio, ADX
- **Volatility**: Bollinger Band Position, ATR, Volatility Regime
- **Volume**: OBV, Volume ROC, VWAP
- **Market Structure**: Support/Resistance Strength

### Step 5: Tokenize
**Purpose**: Convert continuous values to 256-bin tokens for transformer input  
**Input**: train_augmented.parquet, val_augmented.parquet [T × (10 coins × 23 features)]  
**Output**: train_tokens.parquet, val_tokens.parquet [T × (10 coins × 18 channels)]  
         fitted_thresholds.json (quantile bin edges)  
**Contribution**: Quantile-based tokenization, uniform distribution, no data leakage

### Step 6: Sequences
**Purpose**: Create rolling window sequences for supervised learning  
**Input**: train_tokens.parquet, val_tokens.parquet [T × (10 coins × 18 channels)]  
**Output**: train_X.pt [(N_train, 48, 10, 18)] - input sequences  
         train_y.pt [(N_train,)] - binary targets (BUY/NO-BUY)  
         val_X.pt [(N_val, 48, 10, 18)]  
         val_y.pt [(N_val,)]  
**Contribution**: Sliding window sequences, binary classification targets

### Step 7: Train
**Purpose**: Train CryptoTransformerV4 with precision-focused loss  
**Input**: train_X.pt, train_y.pt, val_X.pt, val_y.pt  
**Output**: model.pt (best checkpoint)  
         history.json (training metrics)  
         train_artifact.json (training results)  
**Contribution**: Asymmetric loss (7455x penalty for wrong BUY calls), precision-based early stopping

### Step 8: Evaluate
**Purpose**: Comprehensive validation with precision metrics  
**Input**: model.pt, val_X.pt, val_y.pt  
**Output**: eval_results.json (precision, recall, F1, confidence analysis)  
         confusion_matrices/ (per-confidence-level analysis)  
**Contribution**: Precision-focused evaluation, confidence analysis, signal quality metrics

### Step 9: Inference
**Purpose**: Real-time prediction with confidence filtering  
**Input**: Latest OHLCV data  
**Output**: predictions.json {signal: BUY/NO-BUY, confidence: 0-1, timestamp}  
**Contribution**: High-confidence predictions only (>80% confidence), precision-focused calibration

---

## Model Architecture

### CryptoTransformerV4
**Type**: Encoder-decoder transformer with specialized pathways  
**Input**: `(batch_size, 48, 10, 18)` - 48 hours × 10 coins × 18 channels  
**Output**: `(batch_size, 2)` - Binary classification (BUY/NO-BUY)

**Key Components**:
1. **Channel Embeddings**: Variable-dimension embeddings for 18 channels per coin
2. **Channel Fusion**: Concatenate and project to d_model=256
3. **Coin Embeddings**: Learnable embeddings for 10 cryptocurrencies
4. **Positional Encoding**: Sinusoidal encoding for 48-hour sequences
5. **Encoder-Decoder**: 3 encoder layers, 2 decoder layers, 8 attention heads
6. **BTC→XRP Attention**: Specialized attention pathway from Bitcoin to XRP
7. **Output Head**: Binary classification with precision-focused loss

**Architecture Parameters**:
- `d_model`: 256 (embedding dimension)
- `nhead`: 8 (attention heads)
- `num_encoder_layers`: 3
- `num_decoder_layers`: 2
- `dim_feedforward`: 1024
- `dropout`: 0.2
- `coin_embedding_dim`: 16

---

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

---

## Configuration (config.yaml)

### Data Configuration
```yaml
data:
  coins: [BTC, ETH, LTC, XRP, BNB, ADA, XLM, TRX, DOGE, DOT]
  target_coin: XRP
  interval: 1h
  default_start_date: 2018-05-05
  default_end_date: 2025-10-25
```

### Sequence Configuration
```yaml
sequences:
  input_length: 48            # 48 hours of historical context
  output_length: 1           # Single prediction
  num_channels: 18          # 18 channels: price, volume + 16 technical indicators
  prediction_horizon: 1     # Predict 1 hour ahead
```

### Model Architecture
```yaml
model:
  vocab_size: 256
  num_classes: 256
  num_coins: 10
  binary_classification: false
  
  # Architecture
  d_model: 256
  nhead: 8
  num_encoder_layers: 3
  num_decoder_layers: 2
  dim_feedforward: 1024
  dropout: 0.2
  
  # Embeddings
  coin_embedding_dim: 16
  positional_encoding: sinusoidal
  max_seq_len: 1024
  
  # V4 Features
  multi_horizon_enabled: false
  btc_attention_enabled: true
  time_features_enabled: false
```

### Training Configuration
```yaml
training:
  device: cuda
  epochs: 50
  batch_size: 128
  num_workers: 0
  
  # Optimizer
  learning_rate: 0.0015
  weight_decay: 0.0001
  max_grad_norm: 1.0
  
  # Regularization
  dropout: 0.2
  label_smoothing: 0.01
  gaussian_noise: 0.05
  
  # Learning rate schedule
  scheduler: warmup_cosine
  warmup_epochs: 3
  warmup_start_lr: 0.00001
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 30
    min_delta: 0.0001
  
  # Loss function (precision-focused)
  loss_type: asymmetric
  buy_penalty: 10.0
  no_buy_penalty: 1.0
  use_class_weights: true
```

### Inference Configuration
```yaml
inference:
  target_signal_rate: 0.01
  calibration:
    mode: precision_at_most_rate
    min_signals: 10
    confidence_threshold: 0.8
    precision_target: 0.75
```

---

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

---

## Performance Targets

### Precision Goals
- **BUY Precision**: >75% (up from current 65%)
- **Signal Rate**: ~1% (very selective for high quality)
- **Confidence Threshold**: >80% (only high-confidence predictions)
- **False Positive Penalty**: 7455x higher than false negatives

### Model Performance
- **Training**: Precision-based early stopping
- **Validation**: Comprehensive precision metrics tracking
- **Inference**: High-confidence predictions only
- **Calibration**: Precision-focused threshold optimization

---

## File Structure

```
artifacts/
├── step_00_reset/
├── step_01_download/
├── step_02_clean/
├── step_03_split/
├── step_04_augment/
├── step_05_tokenize/
├── step_06_sequences/
├── step_07_train/
├── step_08_evaluate/
└── step_09_inference/

src/
├── model/
│   ├── trainer.py
│   └── v4_transformer.py
├── pipeline/
│   ├── orchestrator.py
│   ├── schemas.py
│   └── step_XX_*/ (one per pipeline step)
└── utils/
    ├── technical_indicators.py
    ├── precision_loss.py
    └── ordinal_loss.py
```

---

## Usage

### Run Complete Pipeline
```bash
python main.py
```

### Run Individual Steps
```bash
python -m src.pipeline.orchestrator --step 4  # Run augment step only
```

### Configuration
Edit `config.yaml` to modify:
- Data sources and date ranges
- Model architecture parameters
- Training hyperparameters
- Loss function settings
- Inference calibration parameters