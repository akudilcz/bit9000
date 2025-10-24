"""Reset block: clean up all artifacts for a fresh start"""

import shutil
import logging
from pathlib import Path
from datetime import datetime

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import ResetArtifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResetBlock(PipelineBlock):
    """Reset pipeline by deleting all artifacts"""
    
    def run(self) -> ResetArtifact:
        """
        Delete all artifacts in the artifacts directory
        
        Returns:
            ResetArtifact with reset operation details
        """
        logger.info("Running reset block")
        
        artifacts_dir = Path(self.artifact_io.base_dir)
        cleaned_directories = []
        artifacts_cleaned = False
        
        if artifacts_dir.exists():
            logger.info(f"Deleting artifacts directory: {artifacts_dir}")
            
            # Record what's being cleaned
            for subdir in artifacts_dir.iterdir():
                if subdir.is_dir():
                    cleaned_directories.append(str(subdir.name))
            
            # Close all file handlers in all loggers to release file locks
            import time
            from src.utils.logger import Logger
            
            # Close handlers for all loggers
            for name in logging.Logger.manager.loggerDict:
                log = logging.getLogger(name)
                Logger.close_all_handlers(log)
            
            # Also close handlers for the main logger
            main_logger = logging.getLogger("crypto_bot")
            Logger.close_all_handlers(main_logger)
            
            # Try to remove directory, retry on Windows file locks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(artifacts_dir)
                    artifacts_cleaned = True
                    logger.info("✓ All artifacts deleted")
                    break
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(0.2)  # Brief pause before retry
                    else:
                        logger.error(f"Could not fully delete {artifacts_dir}: {e}")
                        raise
        else:
            logger.info("Artifacts directory does not exist, nothing to reset")
        
        # Create artifact
        reset_timestamp = datetime.now()
        artifact = ResetArtifact(
            reset_timestamp=reset_timestamp,
            artifacts_cleaned=artifacts_cleaned,
            cleaned_directories=cleaned_directories,
            metadata=self.create_metadata(upstream_inputs={})
        )
        
        # Write artifact manifest (create step_00_reset directory if needed)
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_00_reset",
            artifact_name="reset_artifact"
        )
        
        logger.info(f"Reset complete: cleaned {len(cleaned_directories)} directories")
        return artifact

