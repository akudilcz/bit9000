"""Model architecture and training modules"""

from src.model.v1_transformer import CryptoTransformerV1
from src.model.v2_transformer import CryptoTransformerV2
from src.model.v3_transformer import CryptoTransformerV3


def create_model(config):
    """
    Create model instance based on config['model']['type']

    Args:
        config: Configuration dictionary

    Returns:
        Model instance (CryptoTransformerV1, CryptoTransformerV2, or CryptoTransformerV3)
    """
    model_type = config['model'].get('type', 'CryptoTransformerV1')
    model_cfg = config['model']

    if model_type == 'CryptoTransformerV1':
        return CryptoTransformerV1(config)
    elif model_type == 'CryptoTransformerV2':
        return CryptoTransformerV2(config)
    elif model_type == 'CryptoTransformerV3':
        # V3: Encoder-Decoder with multi-task learning
        # Determine target coin index from config
        coins = config['data']['coins']
        target_coin = config['data']['target_coin']
        try:
            target_coin_idx = coins.index(target_coin)
        except ValueError:
            raise ValueError(f"Target coin '{target_coin}' not found in coins list: {coins}")
        
        return CryptoTransformerV3(
            vocab_size=model_cfg['vocab_size'],
            num_classes=model_cfg['num_classes'],
            num_coins=model_cfg['num_coins'],
            d_model=model_cfg.get('d_model', 512),
            nhead=model_cfg.get('nhead', 8),
            num_encoder_layers=model_cfg.get('num_encoder_layers', 4),
            num_decoder_layers=model_cfg.get('num_decoder_layers', 4),
            dim_feedforward=model_cfg.get('dim_feedforward', 1024),
            dropout=model_cfg.get('dropout', 0.1),
            coin_embedding_dim=model_cfg.get('coin_embedding_dim', 32),
            positional_encoding=model_cfg.get('positional_encoding', 'learned'),
            max_seq_len=model_cfg.get('max_seq_len', 256),
            multitask_enabled=model_cfg.get('multitask_enabled', True),
            enable_regression=model_cfg.get('enable_regression', True),
            enable_quantiles=model_cfg.get('enable_quantiles', False),
            target_coin_idx=target_coin_idx,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: CryptoTransformerV1, CryptoTransformerV2, CryptoTransformerV3")


__all__ = ['CryptoTransformerV1', 'CryptoTransformerV2', 'CryptoTransformerV3', 'create_model']
