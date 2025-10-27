"""Technical indicators for cryptocurrency data

Provides comprehensive technical analysis indicators optimized for crypto trading signals:
- Momentum: RSI, Stochastic, Williams %R
- Trend: MACD, EMAs, ADX, Parabolic SAR
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Volume: OBV, VWAP, Volume Rate of Change
- Market Structure: Support/Resistance, Price Action Patterns
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


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> pd.Series:
    """
    Calculate Stochastic Oscillator (%K)
    
    Args:
        high: High prices
        low: Low prices  
        close: Close prices
        k_period: %K period (default 14)
        d_period: %D period (default 3)
        
    Returns:
        Stochastic %K values (0-100)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k_percent = k_percent.fillna(50)  # Neutral when no data
    
    return k_percent


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 14) -> pd.Series:
    """
    Calculate Williams %R
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)
        
    Returns:
        Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.fillna(-50)  # Neutral when no data
    
    return williams_r


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) - volatility indicator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
        
    Returns:
        ATR values
    """
    # True Range calculation
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr.fillna(true_range.mean())


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) - trend strength
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default 14)
        
    Returns:
        ADX values (0-100)
    """
    # Calculate directional movements
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = -low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate smoothed values
    atr = true_range.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx.fillna(25)  # Neutral trend strength


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV) - volume momentum
    
    Args:
        close: Close prices
        volume: Volume data
        
    Returns:
        OBV values
    """
    price_change = close.diff()
    obv = volume.copy()
    obv[price_change < 0] = -volume[price_change < 0]
    obv[price_change == 0] = 0
    
    obv = obv.cumsum()
    return obv.fillna(0)


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: Rolling window period
        
    Returns:
        VWAP values
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    return vwap.fillna(typical_price)


def calculate_volume_rate_of_change(volume: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Volume Rate of Change
    
    Args:
        volume: Volume data
        period: Lookback period
        
    Returns:
        Volume ROC percentage
    """
    volume_roc = ((volume - volume.shift(period)) / volume.shift(period)) * 100
    return volume_roc.fillna(0)


def calculate_price_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Price Momentum (rate of change)
    
    Args:
        close: Close prices
        period: Lookback period
        
    Returns:
        Price momentum percentage
    """
    momentum = ((close - close.shift(period)) / close.shift(period)) * 100
    return momentum.fillna(0)


def calculate_support_resistance_strength(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Support/Resistance Strength based on price clustering
    
    Args:
        close: Close prices
        period: Lookback period
        
    Returns:
        Support/resistance strength (0-1)
    """
    # Calculate price levels that price frequently returns to
    rolling_min = close.rolling(window=period).min()
    rolling_max = close.rolling(window=period).max()
    
    # Calculate how close current price is to recent highs/lows
    price_range = rolling_max - rolling_min
    support_strength = (close - rolling_min) / (price_range + 1e-8)
    
    return support_strength.fillna(0.5)


def calculate_volatility_regime(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Volatility Regime (low/medium/high volatility periods)
    
    Args:
        close: Close prices
        period: Lookback period
        
    Returns:
        Volatility regime (0-1, where 0.5 is normal)
    """
    returns = close.pct_change()
    volatility = returns.rolling(window=period).std()
    
    # Normalize volatility relative to historical levels
    vol_percentile = volatility.rolling(window=period*5).rank(pct=True)
    
    return vol_percentile.fillna(0.5)


def add_technical_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to dataframe with multi-coin support
    
    Expected input format:
        - Columns: {COIN}_close, {COIN}_volume, {COIN}_high, {COIN}_low for each coin
        - Index: DatetimeIndex
    
    Output format:
        - Original columns + comprehensive technical indicators
        - Total channels: 18 per coin (price, volume, + 16 indicators)
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary with indicator parameters
        
    Returns:
        DataFrame with added technical indicators
    """
    config = config or {}
    logger.info("Adding comprehensive technical indicators...")
    
    df = df.copy()
    
    # Detect coins from column names
    close_cols = [c for c in df.columns if c.endswith('_close')]
    coins = [c.replace('_close', '') for c in close_cols]
    
    logger.info(f"Detected {len(coins)} coins: {coins}")
    
    for coin in coins:
        close_col = f"{coin}_close"
        high_col = f"{coin}_high"
        low_col = f"{coin}_low"
        volume_col = f"{coin}_volume"
        
        if close_col not in df.columns:
            logger.warning(f"Missing close price for {coin}, skipping")
            continue
        
        # Basic indicators (existing)
        df[f"{coin}_rsi"] = calculate_rsi(df[close_col], period=14)
        df[f"{coin}_macd"] = calculate_macd(df[close_col])
        df[f"{coin}_bb_position"] = calculate_bollinger_position(df[close_col], period=20)
        
        # EMAs
        df[f"{coin}_ema_9"] = calculate_ema(df[close_col], period=9)
        df[f"{coin}_ema_21"] = calculate_ema(df[close_col], period=21)
        df[f"{coin}_ema_50"] = calculate_ema(df[close_col], period=50)
        df[f"{coin}_ema_ratio"] = calculate_ema_ratio(df[close_col], fast_period=9, slow_period=21)
        
        # New momentum indicators
        if high_col in df.columns and low_col in df.columns:
            df[f"{coin}_stochastic"] = calculate_stochastic(df[high_col], df[low_col], df[close_col])
            df[f"{coin}_williams_r"] = calculate_williams_r(df[high_col], df[low_col], df[close_col])
            df[f"{coin}_atr"] = calculate_atr(df[high_col], df[low_col], df[close_col])
            df[f"{coin}_adx"] = calculate_adx(df[high_col], df[low_col], df[close_col])
        else:
            logger.warning(f"Missing high/low prices for {coin}, skipping OHLC-based indicators")
            # Fill with neutral values
            df[f"{coin}_stochastic"] = 50
            df[f"{coin}_williams_r"] = -50
            df[f"{coin}_atr"] = df[close_col].rolling(14).std().fillna(df[close_col].std())
            df[f"{coin}_adx"] = 25
        
        # Volume indicators
        if volume_col in df.columns:
            df[f"{coin}_obv"] = calculate_obv(df[close_col], df[volume_col])
            df[f"{coin}_volume_roc"] = calculate_volume_rate_of_change(df[volume_col])
            
            if high_col in df.columns and low_col in df.columns:
                df[f"{coin}_vwap"] = calculate_vwap(df[high_col], df[low_col], df[close_col], df[volume_col])
            else:
                df[f"{coin}_vwap"] = df[close_col]  # Fallback to close price
        else:
            logger.warning(f"Missing volume for {coin}, skipping volume indicators")
            df[f"{coin}_obv"] = 0
            df[f"{coin}_volume_roc"] = 0
            df[f"{coin}_vwap"] = df[close_col]
        
        # Price action indicators
        df[f"{coin}_price_momentum"] = calculate_price_momentum(df[close_col])
        df[f"{coin}_support_resistance"] = calculate_support_resistance_strength(df[close_col])
        df[f"{coin}_volatility_regime"] = calculate_volatility_regime(df[close_col])
        
        logger.info(f"  Added 16 indicators for {coin}")
    
    # Log statistics
    indicator_cols = [c for c in df.columns if any(c.endswith(x) for x in [
        '_rsi', '_macd', '_bb_position', '_ema_9', '_ema_21', '_ema_50', '_ema_ratio',
        '_stochastic', '_williams_r', '_atr', '_adx', '_obv', '_volume_roc', '_vwap',
        '_price_momentum', '_support_resistance', '_volatility_regime'
    ])]
    logger.info(f"Added {len(indicator_cols)} indicator columns")
    
    # Check for any remaining NaNs
    nan_counts = df[indicator_cols].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN counts in indicators:\n{nan_counts[nan_counts > 0]}")
        logger.info("Forward filling remaining NaNs...")
        df[indicator_cols] = df[indicator_cols].ffill().bfill()
    
    return df


def normalize_indicators_for_tokenization(df: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize technical indicators to similar ranges as price/volume for tokenization
    
    Args:
        df: DataFrame with indicators
        config: Configuration dictionary with normalization parameters
        
    Returns:
        Tuple of (normalized_df, normalization_params)
    """
    config = config or {}
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
        
        # New momentum indicators normalization
        # Stochastic: 0-100, scale to 0-1
        stochastic_col = f"{coin}_stochastic"
        if stochastic_col in df.columns:
            df[stochastic_col] = df[stochastic_col] / 100.0
            norm_params[stochastic_col] = {'type': 'scale', 'factor': 100.0}
        
        # Williams %R: -100 to 0, scale to 0-1
        williams_col = f"{coin}_williams_r"
        if williams_col in df.columns:
            df[williams_col] = (df[williams_col] + 100) / 100.0
            norm_params[williams_col] = {'type': 'scale_offset', 'offset': 100, 'factor': 100.0}
        
        # ATR: normalize by rolling mean
        atr_col = f"{coin}_atr"
        if atr_col in df.columns:
            rolling_mean = df[atr_col].rolling(window=100, min_periods=1).mean()
            df[atr_col] = df[atr_col] / (rolling_mean + 1e-8)
            df[atr_col] = df[atr_col].clip(0, 3) / 3.0  # Scale to [0, 1]
            norm_params[atr_col] = {'type': 'rolling_normalize', 'window': 100}
        
        # ADX: 0-100, scale to 0-1
        adx_col = f"{coin}_adx"
        if adx_col in df.columns:
            df[adx_col] = df[adx_col] / 100.0
            norm_params[adx_col] = {'type': 'scale', 'factor': 100.0}
        
        # Volume indicators normalization
        # OBV: normalize by rolling std
        obv_col = f"{coin}_obv"
        if obv_col in df.columns:
            rolling_std = df[obv_col].rolling(window=100, min_periods=1).std()
            df[obv_col] = df[obv_col] / (rolling_std + 1e-8)
            df[obv_col] = df[obv_col].clip(-5, 5)
            df[obv_col] = (df[obv_col] + 5) / 10.0  # Scale to [0, 1]
            norm_params[obv_col] = {'type': 'zscore_clip', 'window': 100}
        
        # Volume ROC: already percentage, normalize to 0-1
        volume_roc_col = f"{coin}_volume_roc"
        if volume_roc_col in df.columns:
            df[volume_roc_col] = df[volume_roc_col].clip(-100, 100)
            df[volume_roc_col] = (df[volume_roc_col] + 100) / 200.0  # Scale to [0, 1]
            norm_params[volume_roc_col] = {'type': 'clip_scale', 'min': -100, 'max': 100}
        
        # VWAP: normalize relative to current price
        vwap_col = f"{coin}_vwap"
        if vwap_col in df.columns and close_col in df.columns:
            ratio = df[vwap_col] / (df[close_col] + 1e-8)
            ratio = ratio.clip(0.5, 1.5)
            df[vwap_col] = (ratio - 0.5) / 1.0  # Scale to [0, 1]
            norm_params[vwap_col] = {'type': 'price_ratio'}
        
        # Price action indicators normalization
        # Price momentum: already percentage, normalize to 0-1
        momentum_col = f"{coin}_price_momentum"
        if momentum_col in df.columns:
            df[momentum_col] = df[momentum_col].clip(-50, 50)
            df[momentum_col] = (df[momentum_col] + 50) / 100.0  # Scale to [0, 1]
            norm_params[momentum_col] = {'type': 'clip_scale', 'min': -50, 'max': 50}
        
        # Support/Resistance: already 0-1, no change needed
        support_col = f"{coin}_support_resistance"
        if support_col in df.columns:
            norm_params[support_col] = {'type': 'none'}
        
        # Volatility regime: already 0-1, no change needed
        volatility_col = f"{coin}_volatility_regime"
        if volatility_col in df.columns:
            norm_params[volatility_col] = {'type': 'none'}
    
    logger.info("Normalized comprehensive indicators for tokenization")
    
    return df, norm_params

