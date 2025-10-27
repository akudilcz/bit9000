"""Model architecture and training modules"""

from src.model.v5_transformer import CryptoTransformerV5
from src.model.v6_decoder import GPTDecoder


def create_model(config):
    """
    Create model instance based on config

    Args:
        config: Configuration dictionary

    Returns:
        Model instance (CryptoTransformerV5 or GPTDecoder)
    """
    model_cfg = config['model']

    # Validate coins exist (throws early if misconfigured)
    coins = config['data']['coins']
    target_coin = config['data']['target_coin']
    try:
        _ = coins.index(target_coin)
        if 'BTC' in coins:
            _ = coins.index('BTC')
    except ValueError as e:
        raise ValueError(f"Required coin not found in coins list {coins}: {e}")

    # Select model version (default to v6)
    version = model_cfg.get('version', 'v6').lower()
    if version == 'v6':
        return GPTDecoder(config)
    elif version == 'v5':
        return CryptoTransformerV5(config)
    else:
        raise ValueError(f"Unknown model version '{version}'. Supported: 'v5', 'v6'")


__all__ = ['CryptoTransformerV5', 'GPTDecoder', 'create_model']
