"""Config validator to ensure all required config values are present"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration to prevent missing config issues"""
    
    # Define required config paths
    REQUIRED_PATHS = [
        # Data configuration
        ('data', 'coins'),
        ('data', 'target_coin'),
        ('data', 'interval'),
        
        # Binning configuration
        ('binning', 'lookback_hours'),
        ('binning', 'target_shift_hours'),
        ('binning', 'num_bins'),
        ('binning', 'method'),
        ('binning', 'quartiles'),
        ('binning', 'labels'),
        ('binning', 'feature_engineering_params'),
        ('binning', 'feature_engineering_params', 'epsilon'),
        ('binning', 'feature_engineering_params', 'max_ratio'),
        ('binning', 'feature_engineering_params', 'max_volume_ratio'),
        ('binning', 'feature_engineering_params', 'rsi_period'),
        ('binning', 'feature_engineering_params', 'rsi_min_periods'),
        ('binning', 'feature_engineering_params', 'rsi_default'),
        ('binning', 'feature_engineering_params', 'vol_window'),
        ('binning', 'feature_engineering_params', 'vol_min_periods'),
        ('binning', 'feature_engineering_params', 'momentum_4h'),
        ('binning', 'feature_engineering_params', 'momentum_24h'),
        
        # Model configuration
        ('model', 'num_coins'),
        ('model', 'num_target_coins'),
        ('model', 'features_per_coin'),
        ('model', 'sequence_length'),
        ('model', 'num_classes'),
        ('model', 'vocab_size'),
        
        # Training configuration
        ('training', 'epochs'),
        ('training', 'batch_size'),
        ('training', 'learning_rate'),
    ]
    
    @staticmethod
    def validate(config: Dict[str, Any], strict: bool = True) -> List[str]:
        """
        Validate configuration has all required fields
        
        Args:
            config: Configuration dictionary
            strict: If True, raise exception on missing fields. If False, return list of missing fields.
            
        Returns:
            List of missing config paths (empty if all present)
            
        Raises:
            ValueError: If strict=True and any required fields are missing
        """
        missing = []
        
        for path in ConfigValidator.REQUIRED_PATHS:
            current = config
            path_str = '.'.join(path)
            
            try:
                for key in path:
                    if key not in current:
                        missing.append(path_str)
                        break
                    current = current[key]
            except (KeyError, TypeError):
                missing.append(path_str)
        
        if missing:
            error_msg = (
                f"Missing {len(missing)} required config value(s):\n" +
                '\n'.join(f"  - {path}" for path in missing) +
                "\n\nPlease ensure all required config values are defined in config.yaml"
            )
            
            if strict:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
        
        return missing
    
    @staticmethod
    def log_critical_values(config: Dict[str, Any]) -> None:
        """
        Log critical configuration values for debugging
        
        Args:
            config: Configuration dictionary
        """
        logger.info("=" * 80)
        logger.info("CRITICAL CONFIGURATION VALUES")
        logger.info("=" * 80)
        
        critical_values = [
            ('binning.lookback_hours', config.get('binning', {}).get('lookback_hours')),
            ('binning.target_shift_hours', config.get('binning', {}).get('target_shift_hours')),
            ('binning.num_bins', config.get('binning', {}).get('num_bins')),
            ('model.sequence_length', config.get('model', {}).get('sequence_length')),
            ('model.num_classes', config.get('model', {}).get('num_classes')),
            ('model.vocab_size', config.get('model', {}).get('vocab_size')),
            ('data.target_coin', config.get('data', {}).get('target_coin')),
        ]
        
        for path, value in critical_values:
            logger.info(f"  {path}: {value}")
        
        logger.info("=" * 80)

