"""Download block: fetch raw OHLCV data"""

from typing import Optional
from datetime import datetime
import pandas as pd

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import RawDataArtifact, ArtifactMetadata
from src.pipeline.step_01_download.data_collector import DataCollector
from src.utils.logger import get_logger
from src.utils.plot_utils import plot_price_overview, plot_data_quality

logger = get_logger(__name__)


class DownloadBlock(PipelineBlock):
    """Download raw cryptocurrency data"""
    
    def run(self) -> RawDataArtifact:
        """
        Download raw data using config parameters

        Returns:
            RawDataArtifact
        """
        # Extract parameters from config
        data_config = self.config.get('data', {})
        start_date = data_config.get('default_start_date')
        end_date = data_config.get('default_end_date')

        logger.info(f"Running download block: {start_date} to {end_date}")

        # Use existing DataCollector
        collector = DataCollector(self.config)
        
        # Fetch top coins if auto_fetch_coins is enabled
        if self.config['data'].get('auto_fetch_coins', False):
            auto_fetch_limit = self.config['data'].get('collection', {}).get('auto_fetch_limit', 50)
            logger.info(f"Fetching top {auto_fetch_limit} coins from LiveCoinWatch...")
            top_coins = collector.fetch_top_coins(limit=auto_fetch_limit)
            if top_coins:
                self.config['data']['coins'] = [coin.get('code', coin.get('symbol')) for coin in top_coins]
                logger.info(f"Updated config with {len(self.config['data']['coins'])} coins")
                # Update collector with new coins
                collector.coins = self.config['data']['coins']
        
        df = collector.collect_all_data(start_date, end_date)

        # Post-process: Trim to dates where all coins have data
        logger.info("Post-processing: Finding common date range across all coins...")
        df_trimmed = self._trim_to_common_dates(df)
        
        logger.info(f"Trimmed from {len(df)} to {len(df_trimmed)} timesteps")
        logger.info(f"Date range: {df_trimmed.index[0]} to {df_trimmed.index[-1]}")
        
        # Write to outputs
        path = self.artifact_io.write_dataframe(
            df_trimmed,
            block_name="step_01_download",
            artifact_name="raw_data",
            metadata=self.create_metadata()
        )
        
        # Create artifact
        artifact = RawDataArtifact(
            path=path,
            start_date=df_trimmed.index[0],
            end_date=df_trimmed.index[-1],
            num_timesteps=len(df_trimmed),
            num_coins=len([c for c in df_trimmed.columns if c.endswith('_close')]),
            columns=list(df_trimmed.columns),
            freq="H",
            metadata=self.create_metadata()
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_01_download",
            artifact_name="raw_data_artifact"
        )
        
        # Generate visualizations
        plots_dir = self.artifact_io.get_block_dir("step_01_download")
        coins = [c for c in df_trimmed.columns if c.endswith('_close')]
        coins = [c.replace('_close', '') for c in coins]  # Show all coins from config
        
        try:
            plot_price_overview(df_trimmed, coins, plots_dir / "price_overview.png")
            logger.info("Generated price overview plot")
            
            plot_data_quality(df_trimmed, plots_dir / "data_quality.png")
            logger.info("Generated data quality plot")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
        
        logger.info(f"Download complete: {artifact.num_timesteps} timesteps, {artifact.num_coins} coins")
        return artifact

    def _trim_to_common_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trim DataFrame to dates where all coins have non-null data
        
        Args:
            df: DataFrame with OHLCV data for all coins
            
        Returns:
            Trimmed DataFrame with common date range
        """
        # Get close price columns (one per coin)
        close_cols = [c for c in df.columns if c.endswith('_close')]
        
        if not close_cols:
            logger.warning("No close price columns found")
            return df
        
        # Find first and last non-null dates for each coin
        coin_ranges = {}
        for col in close_cols:
            coin = col.replace('_close', '')
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                coin_ranges[coin] = {
                    'start': non_null_data.index[0],
                    'end': non_null_data.index[-1]
                }
                logger.info(f"{coin}: {coin_ranges[coin]['start']} to {coin_ranges[coin]['end']} ({len(non_null_data)} timesteps)")
            else:
                logger.warning(f"{coin}: No valid data found")
        
        if not coin_ranges:
            logger.error("No coins have valid data")
            return df
        
        # Find common date range (intersection of all coin ranges)
        common_start = max(r['start'] for r in coin_ranges.values())
        common_end = min(r['end'] for r in coin_ranges.values())
        
        logger.info(f"Common date range: {common_start} to {common_end}")
        
        # Trim to common range
        df_trimmed = df.loc[common_start:common_end].copy()
        
        # Verify all coins have data in trimmed range
        for col in close_cols:
            coin = col.replace('_close', '')
            null_count = df_trimmed[col].isna().sum()
            if null_count > 0:
                logger.warning(f"{coin}: {null_count} null values in trimmed data")
        
        return df_trimmed


