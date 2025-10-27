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


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        series: Price series
        period: EMA period
        
    Returns:
        EMA values
    """
    ema = series.ewm(span=period, adjust=False).mean()
    ema = ema.fillna(series.mean())
    return ema


def calculate_ema_ratio(series: pd.Series, fast_period: int = 9, slow_period: int = 21) -> pd.Series:
    """
    Calculate ratio of fast EMA to slow EMA (momentum indicator)
    Higher ratio = stronger uptrend, Lower ratio = stronger downtrend
    
    Args:
        series: Price series
        fast_period: Fast EMA period (default 9)
        slow_period: Slow EMA period (default 21)
        
    Returns:
        EMA ratio (fast / slow), normalized to 0-1 range
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate ratio (fast / slow)
    ratio = ema_fast / (ema_slow + 1e-8)
    
    # Normalize to approximately [0, 2] range where 1 = equal
    # Then clip and scale to [0, 1]
    ratio = ratio.clip(0.5, 1.5)  # 0.5x to 1.5x range
    ratio_normalized = (ratio - 0.5) / 1.0  # Scale to [0, 1]
    ratio_normalized = ratio_normalized.fillna(0.5)
    
    return ratio_normalized


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
        - Plus: {COIN}_ema_9, {COIN}_ema_21, {COIN}_ema_50, {COIN}_ema_ratio
    
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
        
        # EMAs - multiple periods for multi-timeframe analysis
        df[f"{coin}_ema_9"] = calculate_ema(df[close_col], period=9)
        df[f"{coin}_ema_21"] = calculate_ema(df[close_col], period=21)
        df[f"{coin}_ema_50"] = calculate_ema(df[close_col], period=50)
        
        # EMA Ratio - fast/slow momentum indicator
        df[f"{coin}_ema_ratio"] = calculate_ema_ratio(df[close_col], fast_period=9, slow_period=21)
        
        logger.info(f"  Added indicators for {coin}")
    
    # Log statistics
    indicator_cols = [c for c in df.columns if any(c.endswith(x) for x in ['_rsi', '_macd', '_bb_position', '_ema_9', '_ema_21', '_ema_50', '_ema_ratio'])]
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
        
        # EMAs: normalize relative to current price
        # This helps the model learn price momentum relative to different timescales
        close_col = f"{coin}_close"
        for ema_period in [9, 21, 50]:
            ema_col = f"{coin}_ema_{ema_period}"
            if ema_col in df.columns and close_col in df.columns:
                # Calculate ratio of EMA to current price
                # Normalize to [0, 1] where 0.5 = equal, <0.5 = price above EMA, >0.5 = price below EMA
                ratio = df[ema_col] / (df[close_col] + 1e-8)
                ratio = ratio.clip(0.5, 1.5)  # 0.5x to 1.5x
                ratio_normalized = (ratio - 0.5) / 1.0  # Scale to [0, 1]
                df[ema_col] = ratio_normalized.fillna(0.5)
                norm_params[ema_col] = {'type': 'ema_ratio', 'period': ema_period}
        
        # EMA Ratio: already normalized to [0, 1], no change needed
        ema_ratio_col = f"{coin}_ema_ratio"
        if ema_ratio_col in df.columns:
            norm_params[ema_ratio_col] = {'type': 'none'}
    
    logger.info("Normalized indicators for tokenization")
    
    return df, norm_params

