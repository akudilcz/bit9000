"""CryptoTransformerV5 - Production-focused, shape-safe, time-only PE, channel attention fusion

Overview:
- Input tokens: x ∈ Z^{B×L×C×Ch} with values in [0, vocab_size-1]
- Channel Fusion: Per (time, coin) we perform channel-aware attention pooling over Ch channels
- Time-only Positional Encoding: Position encodes time step only, repeated across coins
- Shared Encoder: Transformer encoder over sequence length (L*C) with time-only PE
- Optional BTC→XRP cross-attention: Use XRP(last) query to attend BTC timeline (stable, optional)
- Output: Single-horizon token logits for target coin (default: 256-class)

Shapes through the model:
- Input x: (B, L, C, Ch)
- After per-channel embedding and channel-attention fusion:
  fused_per_slot: (B, L*C, d_model)
- After time-only PE: (B, L*C, d_model)
- After shared encoder: shared_memory: (B, L*C, d_model)
- Gather per-coin sequences (time dimension): BTC: (B, L, d_model); XRP: (B, L, d_model)
- Query = XRP at last step: (B, 1, d_model)
- Cross-attend Query→BTC (optional): (B, 1, d_model)
- Final representation: (B, d_model)
- Head logits: (B, num_classes)
"""

from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn

from src.utils.logger import get_logger


logger = get_logger(__name__)


class TimeOnlyPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding that encodes time positions only.

    Expects sequence dimension to be L*C flattened as [t0 coin0..coinC-1, t1 coin0.., ...].
    We build sinusoidal PE for length L and then repeat each time position across the C coins.

    Input:  x: (B, L*C, D)
    Output: x_with_pe: (B, L*C, D)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, L: int, C: int) -> torch.Tensor:
        # x: (B, L*C, D) where D=self.d_model
        B, LC, D = x.shape
        assert LC == L * C, f"Expected sequence length {L*C}, got {LC}"

        # Build sinusoidal PE of length L
        position = torch.arange(L, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=x.device)
                             * -(math.log(10000.0) / D))

        pe = torch.zeros(L, D, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if D % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Tile across coins to align to flattened order [t0 coins..., t1 coins..., ...]
        pe_b = pe.unsqueeze(0).repeat(B, 1, 1)           # (B, L, D)
        pe_flat = pe_b.repeat_interleave(C, dim=1)       # (B, L*C, D)

        x = x + pe_flat
        return self.dropout(x)


class ChannelAttentionFuser(nn.Module):
    """Fuse per-channel token embeddings into a single vector via attention pooling.

    Inputs per slot (one (time, coin) pair):
    - tokens: (Ch,) int tokens

    Computation:
    - Embed tokens: token_embed: (Ch, H)
    - Add channel identity embeddings: channel_id_embed: (Ch, H)
    - Self-attend via a single learned query that pools across the Ch channels
    - Project pooled vector to d_model and add coin embedding (already projected to d_model)

    Batched shape for all slots:
    - tokens_flat: (B*L*C, Ch) -> embeddings: (B*L*C, Ch, H)
    - attention pooled: (B*L*C, 1, H) -> squeeze -> (B*L*C, H)
    - linear to d_model: (B*L*C, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        num_channels: int,
        d_model: int,
        channel_hidden_dim: int,
        channel_nhead: int,
        dropout: float,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_channels = num_channels
        self.d_model = d_model
        self.channel_hidden_dim = channel_hidden_dim

        # Shared token embedding for all channels
        self.token_embedding = nn.Embedding(self.vocab_size, self.channel_hidden_dim)

        # Channel identity embedding to differentiate features (price, volume, indicatorX, ...)
        self.channel_id_embedding = nn.Embedding(self.num_channels, self.channel_hidden_dim)

        # Layer norm + dropout on channel embeddings
        self.channel_norm = nn.LayerNorm(self.channel_hidden_dim)
        self.channel_dropout = nn.Dropout(dropout)

        # One-query multi-head attention to pool across channels
        self.query_vector = nn.Parameter(torch.randn(1, 1, self.channel_hidden_dim))  # (1, 1, H)
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.channel_hidden_dim,
            num_heads=channel_nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Projection to model dimension
        self.to_d_model = nn.Sequential(
            nn.Linear(self.channel_hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens_flat: torch.Tensor) -> torch.Tensor:
        # tokens_flat: (B*L*C, Ch) int
        BLC, Ch = tokens_flat.shape
        assert Ch == self.num_channels, (
            f"ChannelAttentionFuser expected {self.num_channels} channels, got {Ch}. "
            f"Align config.sequences.num_channels with actual sequences."
        )

        # Build per-channel embeddings
        # Gather embeddings for all channels in a vectorized way
        # tokens_emb: (B*L*C, Ch, H)
        tokens_emb = self.token_embedding(tokens_flat.long())

        # Add channel identity embeddings
        # channel_ids: (Ch,) -> (1, Ch) -> (B*L*C, Ch)
        device = tokens_flat.device
        channel_ids = torch.arange(Ch, device=device).unsqueeze(0).expand(BLC, Ch)
        channel_id_emb = self.channel_id_embedding(channel_ids)  # (B*L*C, Ch, H)

        # Combine token and channel id embeddings, then normalize
        channels_repr = tokens_emb + channel_id_emb            # (B*L*C, Ch, H)
        channels_repr = self.channel_norm(channels_repr)
        channels_repr = self.channel_dropout(channels_repr)

        # Attention pooling across channels using a learned query
        # Query: (B*L*C, 1, H) by repeating the learned query per slot
        query = self.query_vector.expand(BLC, -1, -1)          # (B*L*C, 1, H)
        pooled, _ = self.channel_attn(query, channels_repr, channels_repr)  # (B*L*C, 1, H)
        pooled = pooled.squeeze(1)                             # (B*L*C, H)

        # Map into model dimension
        fused = self.to_d_model(pooled)                        # (B*L*C, d_model)
        return fused


class CryptoTransformerV5(nn.Module):
    """Single-horizon Transformer with time-only PE and channel-attention fusion.

    Focuses on correctness:
    - Strict runtime validation of shapes and channel counts
    - Time-only positional encoding to avoid coin-time entanglement
    - Robust and well-documented forward path
    """

    def __init__(self, config: dict = None):
        super().__init__()

        # Configuration plumbing
        config = config or {}
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        seq_config = config.get('sequences', {})

        # Core sizes
        self.vocab_size = int(model_config.get('vocab_size', 256))            # Number of discrete tokens
        self.num_classes = int(model_config.get('num_classes', 256))          # Output classes
        self.d_model = int(model_config.get('d_model', 256))                  # Model width
        self.nhead = int(model_config.get('nhead', 8))                        # Encoder heads
        self.num_encoder_layers = int(model_config.get('num_encoder_layers', 3))
        self.dim_feedforward = int(model_config.get('dim_feedforward', 1024))
        self.dropout_rate = float(model_config.get('dropout', 0.2))

        # Channel fusion hyperparameters
        self.num_channels = int(seq_config.get('num_channels', 9))
        self.channel_hidden_dim = int(model_config.get('channel_hidden_dim', max(32, self.d_model // 4)))
        self.channel_nhead = int(model_config.get('channel_nhead', max(1, self.channel_hidden_dim // 32)))

        # Coins and indices
        coins = data_config.get('coins', ['BTC', 'ETH', 'XRP'])
        target_coin = data_config.get('target_coin', 'XRP')
        try:
            self.num_coins = len(coins)
            self.target_coin_idx = coins.index(target_coin)
            self.btc_coin_idx = coins.index('BTC')
        except ValueError as e:
            raise ValueError(f"Required coin not found in coins list {coins}: {e}")

        # Embeddings and fusion
        # Coin embedding projected to d_model for clean addition
        self.coin_embedding = nn.Embedding(self.num_coins, self.d_model)
        self.channel_fuser = ChannelAttentionFuser(
            vocab_size=self.vocab_size,
            num_channels=self.num_channels,
            d_model=self.d_model,
            channel_hidden_dim=self.channel_hidden_dim,
            channel_nhead=self.channel_nhead,
            dropout=self.dropout_rate,
        )

        # Time-only positional encoding
        self.positional_encoding = TimeOnlyPositionalEncoding(self.d_model, self.dropout_rate)

        # Shared encoder over flattened (time×coin) sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.shared_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        # Optional explicit BTC→XRP cross-attention (kept small and stable)
        self.btc_to_xrp_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout_rate,
            batch_first=True,
        )

        # Prediction head (single-horizon by default)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, self.num_classes),
        )

        # Weight init
        self._init_weights()

        # Logs for visibility
        logger.info(
            f"Initialized CryptoTransformerV5 | d_model={self.d_model}, nhead={self.nhead}, "
            f"enc_layers={self.num_encoder_layers}, ff={self.dim_feedforward}, coins={self.num_coins}, "
            f"channels={self.num_channels}, channel_hidden_dim={self.channel_hidden_dim}"
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _flatten_and_fuse(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, L, C, Ch)
        assert x.dim() == 4, f"Expected 4D input (B,L,C,Ch), got {x.shape}"
        B, L, C, Ch = x.shape
        assert Ch == self.num_channels, (
            f"Input channels ({Ch}) != config num_channels ({self.num_channels}). "
            f"Update config.sequences.num_channels or sequence builder."
        )
        assert C == self.num_coins, f"Input coins ({C}) != configured coins ({self.num_coins})"

        # Flatten slots (time×coin) for channel fusion
        # tokens_flat: (B*L*C, Ch)
        tokens_flat = x.reshape(B * L * C, Ch)

        # Fuse channels to d_model per slot
        fused = self.channel_fuser(tokens_flat)  # (B*L*C, d_model)

        # Add coin embedding in d_model space
        # Build coin indices in flattened order [0..C-1] repeating per time and per batch
        device = x.device
        coin_ids_one_step = torch.arange(C, device=device)                         # (C,)
        coin_ids_time = coin_ids_one_step.repeat(L)                                # (L*C,)
        coin_ids_batch = coin_ids_time.unsqueeze(0).repeat(B, 1)                   # (B, L*C)
        coin_ids_flat = coin_ids_batch.reshape(B * L * C)                          # (B*L*C,)
        fused = fused + self.coin_embedding(coin_ids_flat)                          # (B*L*C, d_model)

        # Reshape back to (B, L*C, d_model) for encoding
        fused_seq = fused.view(B, L * C, self.d_model)
        return fused_seq, L, C

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Validate dtype and range if possible (tokens)
        if not x.dtype == torch.long:
            x = x.long()

        # 1) Flatten across (time, coin) and fuse channels
        # fused_seq: (B, L*C, d_model)
        fused_seq, L, C = self._flatten_and_fuse(x)

        # 2) Add time-only positional encoding (repeats PE across coins)
        encoded_in = self.positional_encoding(fused_seq, L=L, C=C)  # (B, L*C, d_model)

        # 3) Shared encoder over the flattened sequence
        shared_memory = self.shared_encoder(encoded_in)             # (B, L*C, d_model)

        # 4) Gather BTC and target (XRP) timelines (one vector per time step)
        device = x.device
        idx_btc = torch.arange(self.btc_coin_idx, L * C, C, device=device)   # (L,)
        idx_xrp = torch.arange(self.target_coin_idx, L * C, C, device=device)  # (L,)
        btc_seq = shared_memory[:, idx_btc, :]                                # (B, L, d_model)
        xrp_seq = shared_memory[:, idx_xrp, :]                                # (B, L, d_model)

        # 5) Use last XRP time step as query
        xrp_query = xrp_seq[:, -1:, :]                                        # (B, 1, d_model)

        # 6) Optional explicit BTC→XRP cross-attention (stabilized)
        btc_context, _ = self.btc_to_xrp_attention(
            query=xrp_query,
            key=btc_seq,
            value=btc_seq,
        )                                                                      # (B, 1, d_model)

        # 7) Fuse query and BTC context, then predict
        final_repr = (xrp_query + btc_context).squeeze(1)                      # (B, d_model)
        logits = self.head(final_repr)                                         # (B, num_classes)

        return {
            'horizon_1h': {'logits': logits}
        }

    def generate(self, x: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        # Forward then sample class ids
        out = self.forward(x)
        logits = out['horizon_1h']['logits'] / max(1e-6, float(temperature))
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        return {'horizon_1h': pred}


