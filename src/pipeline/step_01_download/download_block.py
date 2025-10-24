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
    
    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> RawDataArtifact:
        """
        Download raw data
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            RawDataArtifact
        """
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
        
        # Write to outputs
        path = self.artifact_io.write_dataframe(
            df,
            block_name="step_01_download",
            artifact_name="raw_data",
            metadata=self.create_metadata()
        )
        
        # Create artifact
        artifact = RawDataArtifact(
            path=path,
            start_date=df.index[0],
            end_date=df.index[-1],
            num_timesteps=len(df),
            num_coins=len([c for c in df.columns if c.endswith('_close')]),
            columns=list(df.columns),
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
        coins = [c for c in df.columns if c.endswith('_close')]
        coins = [c.replace('_close', '') for c in coins]  # Show all coins from config
        
        try:
            plot_price_overview(df, coins, plots_dir / "price_overview.png")
            logger.info("Generated price overview plot")
            
            plot_data_quality(df, plots_dir / "data_quality.png")
            logger.info("Generated data quality plot")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
        
        logger.info(f"Download complete: {artifact.num_timesteps} timesteps, {artifact.num_coins} coins")
        return artifact


