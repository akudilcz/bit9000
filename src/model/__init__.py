"""Model architecture and training modules"""

from src.model.token_predictor import SimpleTokenPredictor
from src.model.token_predictor_v2 import HighPerformanceTokenPredictor


def create_model(config):
    """
    Create model instance based on config['model']['type']
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance (SimpleTokenPredictor or HighPerformanceTokenPredictor)
    """
    model_type = config['model'].get('type', 'SimpleTokenPredictor')
    
    if model_type == 'SimpleTokenPredictor':
        return SimpleTokenPredictor(config)
    elif model_type == 'HighPerformanceTokenPredictor':
        return HighPerformanceTokenPredictor(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: SimpleTokenPredictor, HighPerformanceTokenPredictor")


__all__ = ['SimpleTokenPredictor', 'HighPerformanceTokenPredictor', 'create_model']
