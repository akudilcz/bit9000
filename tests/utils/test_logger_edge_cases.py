import logging
import tempfile
from pathlib import Path

from src.utils.logger import Logger, get_logger


def test_logger_with_valid_config_file(tmp_path):
    # Create a valid config file
    config_path = tmp_path / "test_config.yaml"
    log_file_path = tmp_path / "test.log"
    config_content = f"""
logging:
  level: DEBUG
  format: '%(name)s - %(levelname)s - %(message)s'
  file: '{log_file_path}'
"""
    config_path.write_text(config_content)
    
    # Reset singleton
    Logger._instance = None
    
    logger = Logger.get_logger(name="test", config_path=str(config_path))
    assert logger.level == logging.DEBUG
    assert logger.name == "test"


def test_get_logger_convenience_function():
    # Reset singleton
    Logger._instance = None
    
    logger = get_logger("convenience_test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "convenience_test"


def test_logger_file_handler_creation(tmp_path):
    # Reset singleton
    Logger._instance = None
    
    config_path = tmp_path / "config.yaml"
    config_content = f"""
logging:
  level: INFO
  file: '{tmp_path / "test.log"}'
"""
    config_path.write_text(config_content)
    
    logger = Logger.get_logger(name="file_test", config_path=str(config_path))
    
    # Should have both console and file handlers
    assert len(logger.handlers) >= 1
    # Check that log file was created
    log_file = tmp_path / "test.log"
    assert log_file.exists()
