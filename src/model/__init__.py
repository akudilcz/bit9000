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
    
    return CryptoTransformerV4(
        vocab_size=model_cfg['vocab_size'],
        num_classes=model_cfg['num_classes'],
        num_coins=model_cfg['num_coins'],
        d_model=model_cfg.get('d_model', 64),
        nhead=model_cfg.get('nhead', 4),
        num_encoder_layers=model_cfg.get('num_encoder_layers', 3),
        num_decoder_layers=model_cfg.get('num_decoder_layers', 2),
        dim_feedforward=model_cfg.get('dim_feedforward', 512),
        dropout=model_cfg.get('dropout', 0.25),
        coin_embedding_dim=model_cfg.get('coin_embedding_dim', 32),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        target_coin_idx=target_coin_idx,
        btc_coin_idx=btc_idx,
        binary_classification=model_cfg.get('binary_classification', False),
        num_channels=config['sequences'].get('num_channels', 9),  # NEW: Pass num_channels from config
    )


__all__ = ['CryptoTransformerV4', 'create_model']
