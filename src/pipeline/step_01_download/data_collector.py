"""Binance data collection for cryptocurrencies"""

import os
import time
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np
import ccxt
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCollector:
    """Collect cryptocurrency data from Binance"""
    
    def __init__(self, config: Dict):
        """
        Initialize data collector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.coins = self.data_config.get('coins', [])
        self.interval = self.data_config['interval']
        self.data_dir = self.data_config['data_dir']
        
        # Get reliability settings from config
        collection_config = self.data_config['collection']
        self.max_retries = collection_config['max_retries']
        self.retry_delay = collection_config['retry_delay']
        self.request_timeout = collection_config['request_timeout']
        self.rate_limit_delay = collection_config['rate_limit_delay']
        
        # Initialize CCXT Binance exchange (no API key needed for public data)
        self.binance = ccxt.binance({
            'enableRateLimit': True,  # Respect rate limits
            'timeout': self.request_timeout * 1000,  # Convert to milliseconds
            'options': {
                'defaultType': 'spot',  # Use spot market
            }
        })
        
        
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_all_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Collect data for all coins from Binance
        
        Args:
            start_date: Start date (YYYY-MM-DD or datetime.date)
            end_date: End date (YYYY-MM-DD or datetime.date)
            
        Returns:
            Combined DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = self.data_config['default_start_date']
        
        if end_date is None:
            end_date = self.data_config['default_end_date']
        
        # Ensure dates are strings in YYYY-MM-DD format
        if hasattr(start_date, 'strftime'):
            start_date = start_date.strftime('%Y-%m-%d')
        if hasattr(end_date, 'strftime'):
            end_date = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Collecting data for {len(self.coins)} coins from {start_date} to {end_date}")
        
        all_data = {}
        
        for coin in tqdm(self.coins, desc="Collecting data"):
            coin_data = self._collect_coin_data(coin, start_date, end_date)
            if coin_data is not None and not coin_data.empty:
                # Rename columns to be coin-specific
                coin_upper = coin.upper()
                coin_data = coin_data.rename(columns={
                    'open': f'{coin_upper}_open',
                    'high': f'{coin_upper}_high',
                    'low': f'{coin_upper}_low',
                    'close': f'{coin_upper}_close',
                    'volume': f'{coin_upper}_volume'
                })
                all_data[coin] = coin_data
            else:
                logger.warning(f"No data collected for {coin}")
        
        if not all_data:
            raise ValueError("No data collected for any coins")
        
        # Combine all coin data into a single DataFrame
        combined_df = pd.concat(all_data.values(), axis=1)
        combined_df = combined_df.ffill().bfill()
        
        # Ensure common timestamps
        common_timestamps = combined_df.index
        if len(common_timestamps) == 0:
            raise ValueError("No common timestamps found across coins")
        
        logger.info(f"Found {len(common_timestamps)} common timestamps")
        
        return combined_df
    
    
    def _collect_coin_data(self, coin: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Collect data for a single coin from Binance with retry logic
        
        Args:
            coin: Coin symbol or code
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data or None
        """
        for attempt in range(self.max_retries + 1):
            try:
                data = self._fetch_binance(coin, start_date, end_date)
                
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {coin} from Binance")
                    return data
                else:
                    if attempt < self.max_retries:
                        logger.warning(f"No data collected for {coin}, retrying... (attempt {attempt + 1}/{self.max_retries + 1})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.warning(f"No data collected for {coin} after {self.max_retries + 1} attempts")
                        return None
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Failed to fetch {coin}: {e}, retrying... (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(f"Failed to fetch {coin} after {self.max_retries + 1} attempts: {e}")
                    return None
    
    def _fetch_binance(self, coin: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Binance via CCXT with pagination for complete history
        
        Args:
            coin: Coin code (e.g., BTC, ETH)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert coin format to Binance symbol (BTC -> BTC/USDT)
            coin_code = coin.replace('-USD', '').replace('-', '')
            symbol = f'{coin_code}/USDT'
            timeframe = self.data_config['interval']  # Use config interval
            
            # Check if symbol exists on Binance
            try:
                self.binance.load_markets()
                if symbol not in self.binance.markets:
                    logger.warning(f"Symbol {symbol} not found on Binance")
                    return None
            except Exception as e:
                logger.error(f"Failed to load Binance markets: {e}")
                return None
            
            # Convert dates to timestamps
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            since = int(start_dt.timestamp() * 1000)  # CCXT uses milliseconds
            end_ts = int(end_dt.timestamp() * 1000)
            
            # Fetch all data with pagination
            all_ohlcv = []
            logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
            
            while since < end_ts:
                try:
                    # Fetch batch of OHLCV data (max batch_limit candles per request)
                    batch_limit = self.data_config['collection']['batch_limit']
                    ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, since, limit=batch_limit)
                    
                    if not ohlcv:
                        break
                    
                    # Filter to only include data before end_date
                    ohlcv_filtered = [candle for candle in ohlcv if candle[0] < end_ts]
                    all_ohlcv.extend(ohlcv_filtered)
                    
                    # If we got less than batch_limit candles or reached end date, we're done
                    if len(ohlcv) < batch_limit or ohlcv[-1][0] >= end_ts:
                        break
                    
                    # Next batch starts after the last timestamp
                    since = ohlcv[-1][0] + 1
                    
                    # Small delay to respect rate limits
                    time.sleep(self.rate_limit_delay)
                    
                except ccxt.NetworkError as e:
                    logger.warning(f"Network error fetching {symbol}: {e}, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error fetching {symbol}: {e}")
                    break
            
            if not all_ohlcv:
                logger.warning(f"No data fetched for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates if any
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()
            
            logger.info(f"Successfully fetched {len(df)} hourly candles for {symbol}")
            
            # Quality check: detect sequences of identical prices
            price_unchanged = (df['close'] == df['close'].shift(1))
            max_consecutive_unchanged = price_unchanged.groupby((~price_unchanged).cumsum()).sum().max()
            
            max_unchanged_threshold = self.data_config['collection']['max_unchanged_threshold']
            if max_consecutive_unchanged > max_unchanged_threshold:
                logger.warning(f"{symbol}: Found {max_consecutive_unchanged} consecutive hours with unchanged price")
            
            # Check for missing hours
            expected_hours = int((end_dt - start_dt).total_seconds() / 3600)
            min_data_ratio = self.data_config['collection']['min_data_ratio']
            if len(df) < expected_hours * min_data_ratio:
                logger.warning(f"{symbol}: Expected ~{expected_hours} hours, got {len(df)} ({len(df)/expected_hours*100:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {coin} from Binance: {e}")
            return None
    
    
    
    def _combine_coin_data(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine data from multiple coins into a single DataFrame
        
        Args:
            all_data: Dictionary mapping coin symbols to DataFrames
            
        Returns:
            Combined DataFrame with multi-level columns
        """
        # Get common timestamps across all coins
        common_index = None
        for coin, df in all_data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        logger.info(f"Found {len(common_index)} common timestamps")
        
        # Combine data
        combined_data = {}
        for coin, df in all_data.items():
            df_aligned = df.loc[common_index]
            for col in df.columns:
                combined_data[f"{coin}_{col}"] = df_aligned[col]
        
        combined_df = pd.DataFrame(combined_data, index=common_index)
        
        # Handle missing values
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        # Sort by index
        combined_df = combined_df.sort_index()
        
        return combined_df
    
    def load_data(self, filename: str = 'raw_data.csv') -> pd.DataFrame:
        """
        Load previously collected data
        
        Args:
            filename: Data filename
            
        Returns:
            DataFrame with data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded data from {filepath}, shape: {df.shape}")
        
        return df

