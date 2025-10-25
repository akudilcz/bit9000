# Bit9000 Model Results - 8-Hour Directional BUY Prediction

## Latest Configuration
Successfully implemented a simplified single-horizon binary classification model predicting XRP price direction 8 hours ahead. The model uses 48 hours of historical context (price, volume, RSI, MACD, Bollinger Bands position) across 4 coins (BTC, ETH, LTC, XRP) with a transformer-based architecture (8-layer encoder, 8-layer XRP decoder, 4 attention heads, d_model=208). Training with aggressive regularization (dropout=0.5, weight_decay=0.2, warmup=15 epochs) on ~64K samples with post-training calibration achieving ~5% BUY signal rate. Directional classification: BUY if future_price > current_price, NO-BUY otherwise, enabling intuitive interpretation where precision = % of directional calls that were correct.

## Key Changes This Session
- Switched from 1-hour to 8-hour ahead prediction for less noise and more directional signals
- Implemented post-training threshold calibration to achieve ~5% signal rate while maximizing precision  
- Changed from threshold-based (token ≥ 171) to directional BUY classification (future > current)
- Added learnable threshold (reverted in favor of simpler calibration)
- Fixed window calculation for multi-horizon support
- Improved validation logging with clear precision metrics

## Performance Target
Aiming for BUY signal precision >65% (when we say BUY, we're right >65% of the time) with ~5% signal frequency on validation set.
