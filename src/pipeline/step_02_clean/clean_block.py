"""Clean block: data quality and cleaning"""

import pandas as pd
from pathlib import Path

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import CleanDataArtifact, RawDataArtifact
from src.utils.logger import get_logger
from src.utils.plot_utils import plot_data_quality, plot_timeseries

logger = get_logger(__name__)


class CleanBlock(PipelineBlock):
    """Clean and validate data quality"""
    
    def run(self, raw_artifact: RawDataArtifact = None) -> CleanDataArtifact:
        """
        Clean raw data
        
        Args:
            raw_artifact: RawDataArtifact from download block (optional, will load from disk if not provided)
            
        Returns:
            CleanDataArtifact
        """
        logger.info("Running clean block")
        
        # Load raw artifact if not provided
        if raw_artifact is None:
            raw_artifact_data = self.artifact_io.read_json('artifacts/step_01_download/raw_data_artifact.json')
            raw_artifact = RawDataArtifact(**raw_artifact_data)
        
        # Load raw data
        df = self.artifact_io.read_dataframe(raw_artifact.path)
        
        # Quality metrics before cleaning
        initial_nans = df.isna().sum().sum()
        initial_shape = df.shape
        
        # Cleaning operations
        # 1. Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # 2. Sort by index
        df = df.sort_index()
        
        # 3. Forward fill then backward fill
        df = df.ffill().bfill()
        
        # 4. Check for remaining NaNs
        remaining_nans = df.isna().sum().sum()
        
        # 5. Verify index is monotonic
        if not df.index.is_monotonic_increasing:
            raise ValueError("Index is not monotonic after cleaning")
        
        
        # Quality metrics
        quality_metrics = {
            "initial_nans": int(initial_nans),
            "remaining_nans": int(remaining_nans),
            "duplicate_rows_removed": initial_shape[0] - len(df),
            "nan_rate": float(remaining_nans / (df.shape[0] * df.shape[1])),
            "index_monotonic": bool(df.index.is_monotonic_increasing)
        }
        
        # Write cleaned data
        path = self.artifact_io.write_dataframe(
            df,
            block_name="step_02_clean",
            artifact_name="clean_data",
            metadata=self.create_metadata(
                upstream_inputs={"raw_data": str(raw_artifact.path)}
            )
        )
        
        # Create artifact
        artifact = CleanDataArtifact(
            path=path,
            start_date=df.index[0],
            end_date=df.index[-1],
            num_timesteps=len(df),
            num_coins=raw_artifact.num_coins,
            quality_metrics=quality_metrics,
            metadata=self.create_metadata(
                upstream_inputs={"raw_data": str(raw_artifact.path)}
            )
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_02_clean",
            artifact_name="clean_data_artifact"
        )
        
        # Generate visualizations
        plots_dir = self.artifact_io.get_block_dir("step_02_clean")
        
        try:
            plot_data_quality(df, plots_dir / "data_quality.png")
            logger.info("Generated data quality plot")
            
            # Plot all coin time series
            close_cols = [c for c in df.columns if c.endswith('_close')]
            if close_cols:
                plot_timeseries(df, close_cols, plots_dir / "clean_timeseries.png",
                              title="Cleaned Price Data", ylabel="Price", max_cols=len(close_cols))
                logger.info("Generated timeseries plot")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
        
        logger.info(f"Clean complete: {quality_metrics}")
        return artifact



