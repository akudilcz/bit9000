import logging

from src.utils.logger import Logger


def test_logger_fallback_when_config_missing(tmp_path):
    # Reset singleton to ensure fresh creation
    Logger._instance = None

    missing_config = tmp_path / "missing_config.yaml"
    logger = Logger.get_logger(name="test_logger", config_path=str(missing_config))

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    # Default level is INFO when config is missing
    assert logger.level == logging.INFO
    # Console handler is always added; no file handler by default
    assert len(logger.handlers) == 1



