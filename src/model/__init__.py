"""Model architecture and training modules"""

from src.model.v4_transformer import CryptoTransformerV4


def create_model(config):
    """
    Create model instance based on config

    Args:
        config: Configuration dictionary

    Returns:
        CryptoTransformerV4 instance
    """
    model_cfg = config['model']
    
    # Get coin indices
    coins = config['data']['coins']
    target_coin = config['data']['target_coin']
    try:
        target_coin_idx = coins.index(target_coin)
        btc_idx = coins.index('BTC')
    except ValueError as e:
        raise ValueError(f"Required coin not found in coins list {coins}: {e}")
    
    return CryptoTransformerV4(config)


__all__ = ['CryptoTransformerV4', 'create_model']
