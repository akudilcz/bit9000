"""Plotting utilities for data health visualization"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style - will be configured via parameters


def save_and_close(fig, save_path: Path, title: str = "", dpi: int = 100):
    """Save figure and close to prevent memory leaks"""
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return save_path


def plot_timeseries(
    df: pd.DataFrame,
    columns: List[str],
    save_path: Path,
    title: str = "Time Series Data",
    ylabel: str = "Value",
    max_cols: int = 6,
    last_n_days: int = 730,
    figsize: tuple = (12, 6),
    dpi: int = 100
) -> Path:
    """Plot multiple time series (shows only recent data for performance)"""
    # Filter to last N days for visualization
    if last_n_days and len(df) > 0:
        cutoff_date = df.index[-1] - pd.Timedelta(days=last_n_days)
        df_plot = df[df.index >= cutoff_date]
    else:
        df_plot = df
    
    n_cols = min(len(columns), max_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns[:max_cols]):
        if col in df_plot.columns:
            axes[i].plot(df_plot.index, df_plot[col], linewidth=1, alpha=0.8)
            axes[i].set_ylabel(col)
            axes[i].set_xlabel("Time")
            axes[i].grid(True, alpha=0.3)
    
    return save_and_close(fig, save_path, title, dpi)


def plot_price_overview(
    df: pd.DataFrame,
    coins: List[str],
    save_path: Path,
    last_n_days: int = 730,
    figsize: tuple = (14, 8),
    dpi: int = 100
) -> Path:
    """Plot price overview for multiple coins (shows only recent data for performance)"""
    # Filter to last N days for visualization
    if last_n_days and len(df) > 0:
        cutoff_date = df.index[-1] - pd.Timedelta(days=last_n_days)
        df_plot = df[df.index >= cutoff_date]
    else:
        df_plot = df
    
    n_coins = len(coins)
    fig, axes = plt.subplots(n_coins, 1, figsize=figsize)
    if n_coins == 1:
        axes = [axes]
    
    for i, coin in enumerate(coins):
        close_col = f"{coin}_close"
        volume_col = f"{coin}_volume"
        
        if close_col in df_plot.columns:
            # Price on left axis
            ax1 = axes[i]
            ax1.plot(df_plot.index, df_plot[close_col], color='blue', linewidth=1.5, label='Close Price')
            ax1.set_ylabel(f'{coin} Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)
            
            # Volume on right axis
            if volume_col in df_plot.columns:
                ax2 = ax1.twinx()
                ax2.bar(df_plot.index, df_plot[volume_col], alpha=0.3, color='gray', label='Volume')
                ax2.set_ylabel('Volume', color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
            
            ax1.set_title(f'{coin} Price & Volume')
    
    return save_and_close(fig, save_path, "Price & Volume Overview", dpi)


def plot_data_quality(
    df: pd.DataFrame,
    save_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 100
) -> Path:
    """Plot data quality metrics (NaN, duplicates, etc.)"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Missing values heatmap
    nan_counts = df.isnull().sum()
    non_zero_nans = nan_counts[nan_counts > 0]
    if len(non_zero_nans) > 0:
        axes[0, 0].barh(range(len(non_zero_nans)), non_zero_nans.values)
        axes[0, 0].set_yticks(range(len(non_zero_nans)))
        axes[0, 0].set_yticklabels(non_zero_nans.index, fontsize=8)
        axes[0, 0].set_xlabel('Missing Value Count')
        axes[0, 0].set_title('Missing Values by Column')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Missing Values ✓', 
                       ha='center', va='center', fontsize=14)
        axes[0, 0].axis('off')
    
    # 2. Data completeness over time
    completeness = (1 - df.isnull().mean(axis=1)) * 100
    axes[0, 1].plot(df.index, completeness, linewidth=1)
    axes[0, 1].set_ylabel('Completeness (%)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_title('Data Completeness Over Time')
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Value distribution (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
    if len(numeric_cols) > 0:
        axes[1, 0].hist([df[col].dropna() for col in numeric_cols], 
                       bins=30, alpha=0.5, label=numeric_cols)
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Value Distributions')
        axes[1, 0].legend(fontsize=6, ncol=2)
    
    # 4. Summary statistics table
    stats = df.describe().T[['mean', 'std', 'min', 'max']][:10]
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=stats.round(2).values,
                            rowLabels=stats.index,
                            colLabels=stats.columns,
                            cellLoc='right',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    axes[1, 1].set_title('Summary Statistics')
    
    return save_and_close(fig, save_path, "Data Quality Report", dpi)


def plot_feature_distributions(
    df: pd.DataFrame,
    save_path: Path,
    max_features: int = 12,
    figsize: tuple = (15, 12),
    dpi: int = 100
) -> Path:
    """Plot feature distributions"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_features]
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        axes[i].axvline(mean_val, color='red', linestyle='--', 
                       linewidth=1, label=f'μ={mean_val:.2f}')
        axes[i].legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].axis('off')
    
    return save_and_close(fig, save_path, "Feature Distributions", dpi)


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Path,
    max_features: int = 20,
    figsize: tuple = (12, 10),
    dpi: int = 100
) -> Path:
    """Plot correlation matrix heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_features]
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    return save_and_close(fig, save_path, "", dpi)


def plot_bin_distribution(
    bins_df: pd.DataFrame,
    save_path: Path,
    bin_labels: Dict[int, str],
    figsize: tuple = (12, 6),
    dpi: int = 100
) -> Path:
    """Plot bin distribution across coins"""
    # Get coin columns - they are named directly (BTC, ETH, etc.)
    bin_cols = [col for col in bins_df.columns]
    
    if len(bin_cols) == 0:
        # No bin columns found, return empty plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No bin columns found in dataframe', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return save_and_close(fig, save_path, "Bin Distribution Analysis", dpi)
    
    n_coins = min(len(bin_cols), 6)  # Limit to 6 coins for readability
    fig, axes = plt.subplots(n_coins, 1, figsize=figsize)
    if n_coins == 1:
        axes = [axes]
    
    for i, col in enumerate(bin_cols[:n_coins]):
        coin = col  # Column is already just the coin name (BTC, ETH, etc.)
        bin_counts = bins_df[col].value_counts().sort_index()
        
        colors = ['red', 'orange', 'gray', 'lightgreen', 'green']
        axes[i].bar(bin_counts.index, bin_counts.values, 
                   color=[colors[int(idx)] if int(idx) < len(colors) else 'blue' 
                         for idx in bin_counts.index])
        axes[i].set_xlabel('Bin')
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'{coin} - Bin Distribution')
        axes[i].set_xticks(range(len(bin_labels)))
        axes[i].set_xticklabels([bin_labels.get(i, str(i)) for i in range(len(bin_labels))], 
                               rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    return save_and_close(fig, save_path, "Bin Distribution Analysis", dpi)


def plot_training_history(
    history: Dict,
    save_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 100
) -> Path:
    """Plot training history (loss and accuracy curves)"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data from history (handle both formats: flat lists or nested splits)
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])
    train_accs = history.get('train_acc', [])
    val_accs = history.get('val_acc', [])
    train_dir_accs = history.get('train_dir_acc', [])
    val_dir_accs = history.get('val_dir_acc', [])
    
    if not train_losses:
        axes[0, 0].text(0.5, 0.5, 'No training history available', 
                       ha='center', va='center')
        axes[0, 0].axis('off')
        return save_and_close(fig, save_path, "Training History")
    
    # 1. Training loss across splits/epochs
    axes[0, 0].plot(range(1, len(train_losses)+1), train_losses, 
                   marker='o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(range(1, len(val_losses)+1), val_losses, 
                   marker='s', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Walk-Forward Split' if len(train_losses) < 50 else 'Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Across Walk-Forward Splits')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy across splits/epochs
    axes[0, 1].plot(range(1, len(train_accs)+1), train_accs, 
                   marker='o', label='Train Acc', linewidth=2)
    axes[0, 1].plot(range(1, len(val_accs)+1), val_accs, 
                   marker='s', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Walk-Forward Split' if len(train_accs) < 50 else 'Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Across Walk-Forward Splits')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Directional accuracy across splits
    if train_dir_accs:
        axes[1, 0].plot(range(1, len(train_dir_accs)+1), train_dir_accs, 
                       marker='o', label='Train Dir Acc', linewidth=2)
        axes[1, 0].plot(range(1, len(val_dir_accs)+1), val_dir_accs, 
                       marker='s', label='Val Dir Acc', linewidth=2)
        axes[1, 0].set_xlabel('Walk-Forward Split' if len(train_dir_accs) < 50 else 'Epoch')
        axes[1, 0].set_ylabel('Directional Accuracy')
        axes[1, 0].set_title('Directional Accuracy Across Walk-Forward Splits')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Directional accuracy not available', 
                       ha='center', va='center')
        axes[1, 0].axis('off')
    
    # 4. Summary table
    summary_data = {
        'Split': list(range(1, len(train_losses)+1)),
        'Train Loss': [f"{l:.4f}" for l in train_losses],
        'Val Loss': [f"{l:.4f}" for l in val_losses],
        'Val Acc': [f"{a:.4f}" for a in val_accs]
    }
    summary_df = pd.DataFrame(summary_data)
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary_df.values,
                            colLabels=summary_df.columns,
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[1, 1].set_title('Training Summary')
    
    return save_and_close(fig, save_path, "Training History", dpi)


def plot_sequence_samples(
    X: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    n_samples: int = 4,
    feature_indices: List[int] = None,
    figsize: tuple = (16, 12),
    dpi: int = 100
) -> Path:
    """Plot sample sequences"""
    if feature_indices is None:
        feature_indices = list(range(min(5, X.shape[2])))
    
    n_features = len(feature_indices)
    fig, axes = plt.subplots(n_samples, n_features, 
                            figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        for j, feat_idx in enumerate(feature_indices):
            axes[i, j].plot(X[i, :, feat_idx])
            axes[i, j].set_title(f'Sample {i+1}, Feature {feat_idx}', fontsize=9)
            axes[i, j].set_xlabel('Time Step')
            axes[i, j].set_ylabel('Value')
            axes[i, j].grid(True, alpha=0.3)
    
    return save_and_close(fig, save_path, "Sequence Samples", dpi)

