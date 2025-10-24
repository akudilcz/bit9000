"""Logging utilities for the crypto prediction bot"""

import logging
import os
from typing import Optional
import yaml


class Logger:
    """Centralized logging utility"""
    
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = "crypto_bot", config_path: str = "config.yaml") -> logging.Logger:
        """
        Get or create a logger instance
        
        Args:
            name: Logger name
            config_path: Path to configuration file
            
        Returns:
            Logger instance
        """
        if cls._instance is None:
            cls._instance = cls._create_logger(name, config_path)
        return cls._instance
    
    @classmethod
    def _create_logger(cls, name: str, config_path: str) -> logging.Logger:
        """
        Create a new logger instance
        
        Args:
            name: Logger name
            config_path: Path to configuration file
            
        Returns:
            Configured logger instance
        """
        # Load config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            log_config = config.get('logging', {})
        except FileNotFoundError:
            log_config = {}
        
        # Get logging parameters
        level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', None)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (only if log_file is specified and not null)
        if log_file and log_file.lower() != 'null' and log_file.strip():
            # Ensure the directory exists before creating the file handler
            log_dir = os.path.dirname(log_file)
            if log_dir:  # Only create directory if there's a directory path
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level))
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @classmethod
    def add_file_handler(cls, logger: logging.Logger, log_file: str, level: str = "INFO") -> None:
        """
        Add a file handler to an existing logger
        
        Args:
            logger: Logger instance
            log_file: Path to log file
            level: Logging level
        """
        # Skip per-block log files to avoid Windows file locking conflicts
        # All logging will go to console only
        return
    
    @classmethod
    def close_all_handlers(cls, logger: logging.Logger) -> None:
        """
        Close all file handlers for a logger
        
        Args:
            logger: Logger instance
        """
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)


def get_logger(name: str = "crypto_bot") -> logging.Logger:
    """
    Convenience function to get logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return Logger.get_logger(name)

