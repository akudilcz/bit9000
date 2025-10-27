"""Model architecture and training modules"""

from src.model.v5_transformer import CryptoTransformerV5


def create_model(config):
    """
    Create model instance based on config

    Args:
        config: Configuration dictionary

    Returns:
        CryptoTransformerV5 instance
    """
    model_cfg = config['model']

    # Validate coins exist (throws early if misconfigured)
    coins = config['data']['coins']
    target_coin = config['data']['target_coin']
    try:
        _ = coins.index(target_coin)
        _ = coins.index('BTC')
    except ValueError as e:
        raise ValueError(f"Required coin not found in coins list {coins}: {e}")

    # Select model version (default to v5)
    version = model_cfg.get('version', 'v5').lower()
    if version == 'v5':
        return CryptoTransformerV5(config)
    else:
        raise ValueError(f"Unknown model version '{version}'. Only 'v5' is supported.")


__all__ = ['CryptoTransformerV5', 'create_model']
