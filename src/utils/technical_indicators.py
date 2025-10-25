"""Technical indicators for cryptocurrency data

Provides RSI, MACD, Bollinger Bands and other indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        series: Price series
        period: RSI period (default 14)
        
    Returns:
        RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Fill NaN with 50 (neutral)
    rsi = rsi.fillna(50)
    
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        series: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        MACD histogram (MACD - Signal)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    macd_hist = macd_line - signal_line
    
    # Fill NaN with 0
    macd_hist = macd_hist.fillna(0)
    
    return macd_hist


def calculate_bollinger_position(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Calculate position within Bollinger Bands
    
    Args:
        series: Price series
        period: Rolling window period (default 20)
        num_std: Number of standard deviations (default 2.0)
        
    Returns:
        Position: 0 = lower band, 0.5 = middle, 1.0 = upper band
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # Calculate position (0 to 1)
    band_width = upper_band - lower_band
    position = (series - lower_band) / band_width
    
    # Clip to [0, 1] for extreme values
    position = position.clip(0, 1)
    
    # Fill NaN with 0.5 (middle of bands)
    position = position.fillna(0.5)
    
    return position


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to dataframe with multi-coin support
    
    Expected input format:
        - Columns: {COIN}_close, {COIN}_volume for each coin
        - Index: DatetimeIndex
    
    Output format:
        - Original columns + {COIN}_rsi, {COIN}_macd, {COIN}_bb_position
    
    Args:
        df: DataFrame with price/volume data
        
    Returns:
        DataFrame with added technical indicators
    """
    logger.info("Adding technical indicators...")
    
    df = df.copy()
    
    # Detect coins from column names
    close_cols = [c for c in df.columns if c.endswith('_close')]
    coins = [c.replace('_close', '') for c in close_cols]
    
    logger.info(f"Detected {len(coins)} coins: {coins}")
    
    for coin in coins:
        close_col = f"{coin}_close"
        
        if close_col not in df.columns:
            logger.warning(f"Missing close price for {coin}, skipping")
            continue
        
        # RSI (14-period)
        df[f"{coin}_rsi"] = calculate_rsi(df[close_col], period=14)
        
        # MACD histogram
        df[f"{coin}_macd"] = calculate_macd(df[close_col])
        
        # Bollinger Band position
        df[f"{coin}_bb_position"] = calculate_bollinger_position(df[close_col], period=20)
        
        logger.info(f"  Added indicators for {coin}")
    
    # Log statistics
    indicator_cols = [c for c in df.columns if any(c.endswith(x) for x in ['_rsi', '_macd', '_bb_position'])]
    logger.info(f"Added {len(indicator_cols)} indicator columns")
    
    # Check for any remaining NaNs
    nan_counts = df[indicator_cols].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN counts in indicators:\n{nan_counts[nan_counts > 0]}")
        logger.info("Forward filling remaining NaNs...")
        df[indicator_cols] = df[indicator_cols].ffill().bfill()
    
    return df


def normalize_indicators_for_tokenization(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize technical indicators to similar ranges as price/volume for tokenization
    
    Args:
        df: DataFrame with indicators
        
    Returns:
        Tuple of (normalized_df, normalization_params)
    """
    df = df.copy()
    norm_params = {}
    
    # Detect coins
    close_cols = [c for c in df.columns if c.endswith('_close')]
    coins = [c.replace('_close', '') for c in close_cols]
    
    for coin in coins:
        # RSI: already 0-100, scale to 0-1
        rsi_col = f"{coin}_rsi"
        if rsi_col in df.columns:
            df[rsi_col] = df[rsi_col] / 100.0
            norm_params[rsi_col] = {'type': 'scale', 'factor': 100.0}
        
        # MACD: normalize by rolling std
        macd_col = f"{coin}_macd"
        if macd_col in df.columns:
            rolling_std = df[macd_col].rolling(window=100, min_periods=1).std()
            df[macd_col] = df[macd_col] / (rolling_std + 1e-8)
            # Clip to reasonable range [-5, 5] sigma
            df[macd_col] = df[macd_col].clip(-5, 5)
            # Scale to [0, 1]
            df[macd_col] = (df[macd_col] + 5) / 10.0
            norm_params[macd_col] = {'type': 'zscore_clip', 'window': 100}
        
        # BB position: already 0-1, no change needed
        bb_col = f"{coin}_bb_position"
        if bb_col in df.columns:
            norm_params[bb_col] = {'type': 'none'}
    
    logger.info("Normalized indicators for tokenization")
    
    return df, norm_params

