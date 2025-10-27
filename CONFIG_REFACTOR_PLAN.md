# Config Dictionary Refactoring Plan

## Overview

Refactor the entire application to use config dictionaries passed throughout the system, eliminating individual parameter passing and creating a cleaner, more maintainable API.

## Current Problems

- Functions have long parameter lists (e.g., `buy_penalty`, `no_buy_penalty`, `confidence_threshold`, etc.)
- Hard to add new configuration options
- Inconsistent parameter naming and defaults
- Difficult to maintain and test

## Target Architecture

```python
# Instead of this:
def create_precision_loss(loss_type, buy_penalty=10.0, no_buy_penalty=1.0, confidence_threshold=0.7):
    return AsymmetricPrecisionLoss(buy_penalty, no_buy_penalty)

# Use this:
def create_precision_loss(loss_type: str, config: dict = None):
    config = config or {}
    return AsymmetricPrecisionLoss(config)
```

## Refactoring Pattern

### 1. Function Signature
```python
def function_name(required_param, config: dict = None):
    config = config or {}
    # Extract parameters with defaults
    param1 = config.get('param1', default_value)
    param2 = config.get('param2', default_value)
```

### 2. Class Constructors
```python
class MyClass:
    def __init__(self, config: dict = None):
        config = config or {}
        self.param1 = config.get('param1', default)
        self.param2 = config.get('param2', default)
```

### 3. Config Passing Chain
```
main.py → orchestrator → blocks → utilities → functions
    ↓        ↓         ↓         ↓         ↓
  config   config   config   config   config
```

## Components to Refactor

### 1. Pipeline Blocks
**Current**: Individual parameters passed to block methods
**Target**: `block.run(artifact, config=self.config)`

- `DownloadBlock.run(config)` - API keys, date ranges, retry settings
- `CleanBlock.run(config)` - outlier thresholds, gap filling params
- `SplitBlock.run(config)` - train/val ratios, temporal split settings
- `AugmentBlock.run(config)` - indicator parameters, normalization settings
- `TokenizeBlock.run(config)` - vocab size, binning parameters
- `SequencesBlock.run(config)` - window sizes, stride settings
- `TrainBlock.run(config)` - learning rates, loss parameters, early stopping
- `EvaluateBlock.run(config)` - metric thresholds, confidence settings
- `InferenceBlock.run(config)` - prediction parameters, calibration settings

### 2. Model Creation
**Current**: Individual parameters passed to CryptoTransformerV4
**Target**: `create_model(config)`

- Model architecture parameters (d_model, nhead, layers)
- Embeddings and encoding settings
- BTC→XRP attention pathway config
- Multi-horizon and time feature flags

### 3. Loss Functions
**✅ DONE**: Precision loss functions already refactored
- `AsymmetricPrecisionLoss(config)`
- `PrecisionFocusedLoss(config)`
- `ConfidenceWeightedLoss(config)`
- `PrecisionRecallLoss(config)`

### 4. Technical Indicators
**✅ DONE**: Indicator functions already refactored
- `add_technical_indicators(df, config)`
- `normalize_indicators_for_tokenization(df, config)`

### 5. Data Processing Utilities
- `DataProcessor.__init__(config)` - outlier detection, cleaning parameters
- Download functions - API keys, rate limits, retry logic
- Tokenization functions - vocab size, binning methods

### 6. Training Logic
- `Trainer.__init__(config)` - optimizer settings, scheduler params
- Loss creation functions - loss type, penalty parameters
- Early stopping logic - patience, min delta, metric type

### 7. Evaluation Logic
- Metric calculation functions - threshold settings, confidence levels
- Calibration functions - signal rate targets, precision goals
- Validation logic - confidence filtering, metric aggregation

### 8. Inference Logic
- Prediction functions - confidence thresholds, signal filtering
- Calibration functions - threshold optimization parameters
- Output formatting - signal strength, confidence scores

## Benefits

1. **Cleaner APIs**: No more 10+ parameter functions
2. **Easier Configuration**: Add new options without changing function signatures
3. **Better Maintainability**: Config changes in one place
4. **Consistent Defaults**: Centralized default handling
5. **Easier Testing**: Mock config dictionaries for testing
6. **Better Documentation**: Config keys serve as API documentation

## Implementation Strategy

### Phase 1: Core Infrastructure
1. ✅ Loss functions (DONE)
2. ✅ Technical indicators (DONE)
3. ⏳ Model creation functions
4. ⏳ Pipeline block interfaces

### Phase 2: Pipeline Blocks
1. ⏳ Update block constructors to accept config
2. ⏳ Update block.run() methods to use config
3. ⏳ Update orchestrator to pass config to blocks

### Phase 3: Utilities and Helpers
1. ⏳ Data processing utilities
2. ⏳ Training utilities
3. ⏳ Evaluation utilities

### Phase 4: Integration Testing
1. ⏳ End-to-end pipeline testing
2. ⏳ Configuration validation
3. ⏳ Backward compatibility checks

## Config Structure

```yaml
# Top-level config structure
data:
  coins: [...]
  target_coin: XRP
  date_ranges: {...}

model:
  architecture: {...}
  training: {...}
  loss: {...}

pipeline:
  download: {...}
  clean: {...}
  augment: {...}
  tokenize: {...}
  sequences: {...}
  train: {...}
  evaluate: {...}
  inference: {...}
```

## Migration Guide

### Before:
```python
loss_fn = create_precision_loss(
    loss_type='asymmetric',
    buy_penalty=10.0,
    no_buy_penalty=1.0,
    confidence_threshold=0.7
)
```

### After:
```python
config = {'buy_penalty': 10.0, 'no_buy_penalty': 1.0, 'confidence_threshold': 0.7}
loss_fn = create_precision_loss('asymmetric', config)
```

## Risk Mitigation

1. **Backward Compatibility**: Keep old APIs working during transition
2. **Gradual Migration**: Migrate one component at a time
3. **Configuration Validation**: Add schema validation for config dictionaries
4. **Comprehensive Testing**: Test each component after refactoring
5. **Documentation Updates**: Update all docs to reflect new patterns

## Success Criteria

- ✅ All functions accept config dictionaries
- ✅ No functions have >3 parameters besides config
- ✅ Easy to add new configuration options
- ✅ All tests pass with new API
- ✅ Performance unchanged
- ✅ Code is more maintainable
