"""IO utilities for reading/writing pipeline artifacts"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import torch

from src.utils.logger import get_logger
from src.pipeline.schemas import ArtifactMetadata

logger = get_logger(__name__)


class ArtifactIO:
    """Read/write pipeline artifacts"""
    
    def __init__(self, base_dir: str):
        """
        Initialize artifact IO
        
        Args:
            base_dir: Base directory for all artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cleaned_blocks = set()
    
    def get_block_dir(self, block_name: str, clean: bool = False) -> Path:
        """
        Get directory for a specific block
        
        Args:
            block_name: Name of the block
            clean: If True, delete all existing files in the directory (only once per block)
        """
        block_dir = self.base_dir / block_name
        
        if clean and block_name not in self._cleaned_blocks:
            if block_dir.exists():
                import shutil
                import time
                
                # Only clean non-log files to avoid conflicts
                # Remove everything except log files, then recreate the directory
                try:
                    for item in block_dir.iterdir():
                        if item.is_file() and not (item.suffix == '.log' or '.log.' in item.name):
                            try:
                                item.unlink()
                            except (OSError, PermissionError) as e:
                                logger.warning(f"Could not delete {item}: {e}")
                        elif item.is_dir():
                            try:
                                shutil.rmtree(item)
                            except (OSError, PermissionError) as e:
                                logger.warning(f"Could not delete directory {item}: {e}")
                    logger.info(f"Cleaned block directory (excluding log files): {block_dir}")
                except Exception as e:
                    logger.warning(f"Could not fully clean {block_dir}: {e}. Continuing anyway.")
            
            self._cleaned_blocks.add(block_name)
        
        block_dir.mkdir(parents=True, exist_ok=True)
        return block_dir
    
    def compute_hash(self, data: Any) -> str:
        """Compute hash of data for versioning"""
        if isinstance(data, pd.DataFrame):
            content = data.to_json()
        elif isinstance(data, np.ndarray):
            content = data.tobytes()
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        if isinstance(content, str):
            content = content.encode()
        
        return hashlib.md5(content).hexdigest()
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        block_name: str,
        artifact_name: str,
        metadata: Optional[ArtifactMetadata] = None
    ) -> Path:
        """
        Write DataFrame to parquet with full metadata preservation
        
        Parquet format preserves:
        - Index (including MultiIndex)
        - Column names and types
        - Categoricals
        - Timezone-aware datetimes
        
        Args:
            df: DataFrame to write
            block_name: Name of the block
            artifact_name: Name of the artifact
            metadata: Optional metadata
            
        Returns:
            Path to written file
        """
        block_dir = self.get_block_dir(block_name, clean=True)
        
        data_path = block_dir / f"{artifact_name}.parquet"
        
        # Write with full metadata preservation
        df.to_parquet(
            data_path,
            engine='pyarrow',  # Use pyarrow for best metadata support
            compression='snappy',  # Good balance of speed/compression
            index=True,  # Always preserve index
        )
        
        logger.info(f"Wrote {artifact_name} to {data_path}")
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Index: {df.index.name if df.index.name else type(df.index).__name__}")
        logger.info(f"  - Columns: {len(df.columns)}")
        
        return data_path
    
    def write_array(
        self,
        arr: np.ndarray,
        block_name: str,
        artifact_name: str,
        metadata: Optional[ArtifactMetadata] = None
    ) -> Path:
        """Write numpy array"""
        block_dir = self.get_block_dir(block_name, clean=True)
        
        data_path = block_dir / f"{artifact_name}.npz"
        np.savez_compressed(data_path, data=arr)
        
        logger.info(f"Wrote {artifact_name} to {data_path}")
        return data_path
    
    def write_model(
        self,
        model_state: Dict[str, Any],
        block_name: str,
        artifact_name: str,
        metadata: Optional[ArtifactMetadata] = None
    ) -> Path:
        """Write PyTorch model state"""
        block_dir = self.get_block_dir(block_name, clean=True)
        
        data_path = block_dir / f"{artifact_name}.pt"
        torch.save(model_state, data_path)
        
        logger.info(f"Wrote model to {data_path}")
        return data_path
    
    def write_json(
        self,
        data: Dict[str, Any],
        block_name: str,
        artifact_name: str,
        metadata: Optional[ArtifactMetadata] = None
    ) -> Path:
        """Write JSON data"""
        block_dir = self.get_block_dir(block_name, clean=True)
        
        data_path = block_dir / f"{artifact_name}.json"
        
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Wrote {artifact_name} to {data_path}")
        return data_path
    
    def read_dataframe(self, path: Path) -> pd.DataFrame:
        """
        Read DataFrame from parquet with full metadata restoration
        
        Args:
            path: Path to parquet file
            
        Returns:
            DataFrame with all metadata preserved
        """
        logger.info(f"Reading DataFrame from {path}")
        df = pd.read_parquet(path, engine='pyarrow')
        
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Index: {df.index.name if df.index.name else type(df.index).__name__}")
        logger.info(f"  - Columns: {len(df.columns)}")
        
        return df
    
    def read_array(self, path: Path) -> np.ndarray:
        """Read array from path"""
        logger.info(f"Reading array from {path}")
        loaded = np.load(path)
        return loaded['data']
    
    def read_model(self, path: Path, device: str = 'cpu') -> Dict[str, Any]:
        """Read model state from path"""
        logger.info(f"Reading model from {path}")
        return torch.load(path, map_location=device)
    
    def read_json(self, path: Path) -> Dict[str, Any]:
        """Read JSON from path"""
        logger.info(f"Reading JSON from {path}")
        with open(path, 'r') as f:
            return json.load(f)
