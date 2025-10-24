"""Base class for pipeline blocks"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import ArtifactMetadata
from src.utils.logger import get_logger, Logger

logger = get_logger(__name__)


class PipelineBlock(ABC):
    """Base class for all pipeline blocks"""
    
    def __init__(self, config: Dict[str, Any], artifact_io: ArtifactIO):
        """
        Initialize pipeline block
        
        Args:
            config: Configuration dictionary
            artifact_io: Artifact IO handler
        """
        self.config = config
        self.artifact_io = artifact_io
        self.block_name = self.__class__.__name__.replace("Block", "").lower()
        
        # Setup logging to artifact directory
        self._setup_block_logging()
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Run the pipeline block
        
        Returns:
            Block output (artifact)
        """
        pass
    
    def create_metadata(self, upstream_inputs: Dict[str, str] = None) -> ArtifactMetadata:
        """Create metadata for an artifact"""
        return ArtifactMetadata(
            schema_name=self.block_name,
            upstream_inputs=upstream_inputs or {}
        )
    
    def _setup_block_logging(self) -> None:
        """Setup logging to write to the block's artifact directory"""
        # Skip per-block logging to avoid file conflicts
        # All logging will go to console only
        pass
    
    def _cleanup_logging(self) -> None:
        """Cleanup logging handlers when block completes"""
        root_logger = get_logger()
        Logger.close_all_handlers(root_logger)



