"""
Inspect and verify tokenized data structures, ranges, and statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def inspect_tokenized_data():
    """Load and inspect tokenized data"""
    
    # Load tokenized data
    train_path = Path("artifacts/step_05_tokenize/train_tokens.parquet")
    val_path = Path("artifacts/step_05_tokenize/val_tokens.parquet")
    artifact_path = Path("artifacts/step_05_tokenize/tokenize_artifact.json")
    
    print("="*80)
    print("TOKENIZED DATA INSPECTION")
    print("="*80)
    
    # Load artifact metadata
    with open(artifact_path, 'r') as f:
        artifact = json.load(f)
    
    print(f"\n📋 Artifact Metadata:")
    print(f"  Train shape: {artifact['train_shape']}")
    print(f"  Val shape: {artifact['val_shape']}")
    print(f"  Created: {artifact['metadata']['created_at']}")
    
    # Load actual dataframes
    train_tokens = pd.read_parquet(train_path)
    val_tokens = pd.read_parquet(val_path)
    
    print(f"\n📊 Actual DataFrame Shapes:")
    print(f"  Train: {train_tokens.shape} (dtype: {train_tokens.dtypes.iloc[0]})")
    print(f"  Val: {val_tokens.shape}")
    print(f"  Columns: {len(train_tokens.columns)}")
    
    # Show column structure
    print(f"\n📌 Column Structure (first 5 coins):")
    coins = ['BTC', 'ETH', 'LTC', 'XRP', 'BNB']
    for coin in coins:
        coin_cols = [col for col in train_tokens.columns if col.startswith(coin + '_')]
        print(f"  {coin}: {len(coin_cols)} channels")
        print(f"    {', '.join(coin_cols[:5])}...")
    
    # Token value statistics
    print(f"\n📈 Token Value Statistics:")
    print(f"  Min value: {train_tokens.values.min()}")
    print(f"  Max value: {train_tokens.values.max()}")
    print(f"  Expected range: 0-255")
    
    # Per-column statistics
    print(f"\n📊 Per-Column Statistics (sample columns):")
    sample_cols = ['BTC_price', 'BTC_volume', 'BTC_rsi', 'BTC_macd', 'BTC_bb_position',
                   'BTC_stochastic', 'BTC_adx', 'BTC_obv']
    for col in sample_cols:
        if col in train_tokens.columns:
            min_val = train_tokens[col].min()
            max_val = train_tokens[col].max()
            mean_val = train_tokens[col].mean()
            std_val = train_tokens[col].std()
            variance = train_tokens[col].var()
            print(f"  {col}:")
            print(f"    Range: [{min_val:.0f}, {max_val:.0f}]")
            print(f"    Mean: {mean_val:.2f}, Std: {std_val:.2f}, Var: {variance:.2f}")
            print(f"    Null count: {train_tokens[col].isna().sum()}")
    
    # Distribution across all tokens
    print(f"\n📊 Token Distribution Across All Values:")
    all_tokens = train_tokens.values.flatten()
    unique_vals, counts = np.unique(all_tokens[~np.isnan(all_tokens)].astype(int), return_counts=True)
    print(f"  Unique token values: {len(unique_vals)}")
    print(f"  Min used: {unique_vals.min()}, Max used: {unique_vals.max()}")
    print(f"  Total tokens: {len(all_tokens[~np.isnan(all_tokens)])}")
    
    # Coverage of 256 bins
    coverage = len(unique_vals) / 256 * 100
    print(f"  Bin coverage: {coverage:.1f}% ({len(unique_vals)}/256 bins used)")
    
    # Show distribution histogram
    print(f"\n📊 Distribution Histogram (bins 0-255):")
    for bin_start in range(0, 256, 16):
        bin_end = min(bin_start + 16, 256)
        count = sum(counts[(unique_vals >= bin_start) & (unique_vals < bin_end)])
        pct = count / len(all_tokens[~np.isnan(all_tokens)]) * 100
        bar = '█' * int(pct / 2)
        print(f"  Bins {bin_start:3d}-{bin_end-1:3d}: {bar} {pct:5.1f}%")
    
    # Compare train vs val distribution
    print(f"\n📊 Train vs Validation Statistics:")
    all_train = train_tokens.values.flatten()
    all_val = val_tokens.values.flatten()
    all_train_clean = all_train[~np.isnan(all_train)]
    all_val_clean = all_val[~np.isnan(all_val)]
    
    print(f"  Train - Mean: {all_train_clean.mean():.2f}, Std: {all_train_clean.std():.2f}")
    print(f"  Val   - Mean: {all_val_clean.mean():.2f}, Std: {all_val_clean.std():.2f}")
    
    # Data quality checks
    print(f"\n✅ Data Quality Checks:")
    null_train = train_tokens.isna().sum().sum()
    null_val = val_tokens.isna().sum().sum()
    print(f"  Null values in train: {null_train} ({null_train / (train_tokens.shape[0] * train_tokens.shape[1]) * 100:.4f}%)")
    print(f"  Null values in val: {null_val} ({null_val / (val_tokens.shape[0] * val_tokens.shape[1]) * 100:.4f}%)")
    
    # Check for inf values
    inf_train = np.isinf(train_tokens.values).sum()
    inf_val = np.isinf(val_tokens.values).sum()
    print(f"  Inf values in train: {inf_train}")
    print(f"  Inf values in val: {inf_val}")
    
    # Variance across different indicator types
    print(f"\n📊 Variance by Indicator Type:")
    for ind_type in ['price', 'volume', 'rsi', 'macd', 'bb_position', 'ema_9', 'stochastic', 'adx', 'obv']:
        cols = [col for col in train_tokens.columns if col.endswith('_' + ind_type)]
        if cols:
            variances = [train_tokens[col].var() for col in cols]
            avg_var = np.mean(variances)
            min_var = np.min(variances)
            max_var = np.max(variances)
            print(f"  {ind_type:15s}: Mean={avg_var:7.2f}, Min={min_var:7.2f}, Max={max_var:7.2f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    inspect_tokenized_data()
