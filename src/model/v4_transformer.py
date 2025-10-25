"""CryptoTransformerV4 - Multi-Horizon with BTC→XRP Attention and Time Features

Key Improvements over V3:
1. BTC→XRP Dedicated Attention: Explicit pathway since BTC leads altcoins
2. Multi-Horizon Prediction: Simultaneously predict 1h, 2h, 4h, 8h ahead
3. Time-based Features: Hour of day, day of week, sequence position embeddings
4. Improved representation learning through auxiliary prediction tasks

Architecture:
- Shared Encoder: Process all coins with temporal attention
- BTC Encoder: Dedicated pathway for BTC features
- XRP Decoder: Attends to both shared encoder + BTC encoder
- Multi-Horizon Heads: 4 prediction heads for different time horizons
- Time Embeddings: Cyclical encoding of hour/day patterns

Input:  (batch, seq_len, num_coins, num_channels) + time_features
Output: dict with 'horizon_1h', 'horizon_2h', 'horizon_4h', 'horizon_8h'
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (does not learn, more generalizable)"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        B, L, D = x.shape
        
        # Sinusoidal positional encoding
        position = torch.arange(L, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=x.device) * 
                             -(math.log(10000.0) / D))
        
        pe = torch.zeros(L, D, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if D % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        x = x + pe.unsqueeze(0)
        return self.dropout(x)


class CyclicalTimeEncoding(nn.Module):
    """Encode cyclical time features (hour, day of week) using sin/cos"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Project cyclical features to d_model
        self.time_projection = nn.Linear(4, d_model)  # 2 for hour + 2 for day
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len) - unix timestamps or datetime indices
            
        Returns:
            time_embeddings: (batch, seq_len, d_model)
        """
        # For now, we'll compute hour and day from the timestamp
        # Assuming timestamps are in hours since epoch (modify as needed)
        
        # Extract hour of day (0-23)
        hour = (timestamps % 24).float()
        hour_sin = torch.sin(2 * math.pi * hour / 24)
        hour_cos = torch.cos(2 * math.pi * hour / 24)
        
        # Extract day of week (0-6)
        day = ((timestamps // 24) % 7).float()
        day_sin = torch.sin(2 * math.pi * day / 7)
        day_cos = torch.cos(2 * math.pi * day / 7)
        
        # Stack: (batch, seq_len, 4)
        cyclical_features = torch.stack([hour_sin, hour_cos, day_sin, day_cos], dim=-1)
        
        # Project to d_model
        time_emb = self.time_projection(cyclical_features)  # (batch, seq_len, d_model)
        
        return time_emb


class CryptoTransformerV4(nn.Module):
    """Multi-Horizon Transformer with BTC→XRP attention and time features
    
    Key Features:
    - Dedicated BTC encoder (since BTC leads the market)
    - Multi-horizon prediction heads (1h, 2h, 4h, 8h)
    - Time-based features (hour, day, sequence position)
    - Cross-attention from XRP decoder to BTC features
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        num_classes: int = 256,
        num_coins: int = 4,  # BTC, ETH, LTC, XRP
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        coin_embedding_dim: int = 32,
        max_seq_len: int = 1024,
        target_coin_idx: int = 3,  # XRP
        btc_coin_idx: int = 0,  # BTC
        binary_classification: bool = False,  # NEW: Binary BUY/NO-BUY classification
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.num_coins = num_coins
        self.d_model = d_model
        self.target_coin_idx = target_coin_idx
        self.btc_coin_idx = btc_coin_idx
        self.max_seq_len = max_seq_len
        self.binary_classification = binary_classification
        
        # Coin-specific embeddings
        self.coin_embedding = nn.Embedding(num_coins, coin_embedding_dim)
        
        # Token embeddings for price and volume (reused for all 5 channels)
        self.price_embedding = nn.Embedding(vocab_size, d_model // 4)
        self.volume_embedding = nn.Embedding(vocab_size, d_model // 4)
        
        # Channel fusion: price + volume + indicator_avg + coin
        fusion_dim = (d_model // 4) * 3 + coin_embedding_dim
        self.channel_fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Time-based feature encoding
        self.time_encoding = CyclicalTimeEncoding(d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model, dropout)
        
        # Shared encoder: process all coins
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.shared_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # BTC-specific encoder (dedicated pathway)
        self.btc_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,  # Smaller dedicated encoder
            norm=nn.LayerNorm(d_model)
        )
        
        # XRP decoder: attends to shared encoder + BTC encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.xrp_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # BTC→XRP cross-attention (explicit pathway)
        self.btc_to_xrp_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-horizon prediction heads (1h, 2h, 4h, 8h)
        self.horizon_heads = nn.ModuleDict({
            'horizon_1h': self._make_prediction_head(d_model, num_classes if not binary_classification else 2, dropout),
            'horizon_2h': self._make_prediction_head(d_model, num_classes if not binary_classification else 2, dropout),
            'horizon_4h': self._make_prediction_head(d_model, num_classes if not binary_classification else 2, dropout),
            'horizon_8h': self._make_prediction_head(d_model, num_classes if not binary_classification else 2, dropout),
        })
        
        self._init_weights()
        
        param_count = self.count_parameters()
        logger.info(f"Initialized CryptoTransformerV4 with {param_count:,} parameters")
        logger.info(f"  Shared Encoder: {num_encoder_layers} layers, BTC Encoder: 2 layers, XRP Decoder: {num_decoder_layers} layers")
        logger.info(f"  d_model: {d_model}, nhead: {nhead}, dim_ff: {dim_feedforward}")
        logger.info(f"  Multi-horizon: 1h, 2h, 4h, 8h prediction heads ({'binary BUY/NO-BUY' if binary_classification else f'{num_classes} classes'})")
        logger.info(f"  BTC→XRP dedicated attention pathway enabled")
        logger.info(f"  Time features: hour, day of week, sequence position")
    
    def _make_prediction_head(self, d_model: int, num_classes: int, dropout: float) -> nn.Module:
        """Create a prediction head for one time horizon"""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _embed_sequence(
        self,
        x: torch.Tensor,
        coin_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed token sequences with coin context
        
        Args:
            x: (batch, seq_len, num_coins, num_channels) - tokenized data
            coin_indices: Optional coin IDs
            
        Returns:
            embeddings: (batch, seq_len * num_coins, d_model)
        """
        if x.dim() == 4:
            # Multi-coin input
            B, L, C, Ch = x.shape
            assert C == self.num_coins, f"Expected {self.num_coins} coins, got {C}"
            
            # Flatten to (batch, seq_len * num_coins, num_channels)
            x_flat = x.reshape(B, L * C, Ch)
            
            # Coin indices
            if coin_indices is None:
                coin_ids = torch.arange(C, device=x.device).repeat(L)
                coin_indices = coin_ids.unsqueeze(0).expand(B, -1)
        else:
            # Single coin input
            B, L, Ch = x.shape
            x_flat = x
            
            if coin_indices is None:
                coin_indices = torch.zeros(B, L, dtype=torch.long, device=x.device)
        
        # Embed channels
        price_emb = self.price_embedding(x_flat[:, :, 0])
        volume_emb = self.volume_embedding(x_flat[:, :, 1])
        
        # Embed indicators if present (5 channels)
        if Ch == 5:
            rsi_emb = self.price_embedding(x_flat[:, :, 2])
            macd_emb = self.volume_embedding(x_flat[:, :, 3])
            bb_emb = self.price_embedding(x_flat[:, :, 4])
            indicator_avg = (rsi_emb + macd_emb + bb_emb) / 3.0
            combined = torch.cat([price_emb, volume_emb, indicator_avg, self.coin_embedding(coin_indices)], dim=-1)
        else:
            # 2 channels - pad with zeros
            zero_pad = torch.zeros_like(price_emb)
            combined = torch.cat([price_emb, volume_emb, zero_pad, self.coin_embedding(coin_indices)], dim=-1)
        
        embedded = self.channel_fusion(combined)
        
        return embedded
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-horizon prediction
        
        Args:
            x: (batch, seq_len, num_coins, num_channels) - input tokens
            timestamps: (batch, seq_len) - optional timestamps for time features
            
        Returns:
            dict with keys: 'horizon_1h', 'horizon_2h', 'horizon_4h', 'horizon_8h'
            Each contains: {'logits': (batch, num_classes)}
        """
        B, L, C, Ch = x.shape
        
        # 1. Embed all coins
        encoder_input = self._embed_sequence(x)  # (B, L*C, d_model)
        
        # Reshape for temporal processing: (B, L*C, d_model)
        # Add time features if provided
        if timestamps is not None:
            # Time encoding per timestep
            time_emb = self.time_encoding(timestamps)  # (B, L, d_model)
            # Expand to all coins: repeat for each coin
            time_emb = time_emb.unsqueeze(2).repeat(1, 1, C, 1)  # (B, L, C, d_model)
            time_emb = time_emb.reshape(B, L * C, self.d_model)  # (B, L*C, d_model)
            encoder_input = encoder_input + time_emb
        
        # Add positional encoding
        encoder_input = self.position_encoding(encoder_input)
        
        # 2. Shared encoder: process all coins
        shared_memory = self.shared_encoder(encoder_input)  # (B, L*C, d_model)
        
        # 3. BTC-specific encoder
        # Extract BTC features: every C-th element starting from btc_coin_idx
        btc_indices = torch.arange(self.btc_coin_idx, L * C, C, device=x.device)
        btc_features = encoder_input[:, btc_indices, :]  # (B, L, d_model)
        btc_memory = self.btc_encoder(btc_features)  # (B, L, d_model)
        
        # 4. XRP decoder: use last timestep of XRP as query
        # Extract XRP features
        xrp_indices = torch.arange(self.target_coin_idx, L * C, C, device=x.device)
        xrp_features = encoder_input[:, xrp_indices, :]  # (B, L, d_model)
        
        # Use last timestep as query for prediction
        xrp_query = xrp_features[:, -1:, :]  # (B, 1, d_model)
        
        # Decode with attention to shared memory
        xrp_decoded = self.xrp_decoder(
            tgt=xrp_query,
            memory=shared_memory
        )  # (B, 1, d_model)
        
        # 5. BTC→XRP cross-attention (explicit pathway)
        btc_context, _ = self.btc_to_xrp_attention(
            query=xrp_decoded,
            key=btc_memory,
            value=btc_memory
        )  # (B, 1, d_model)
        
        # 6. Fuse XRP decoded + BTC context
        final_repr = xrp_decoded + btc_context  # (B, 1, d_model)
        final_repr = final_repr.squeeze(1)  # (B, d_model)
        
        # 7. Multi-horizon predictions
        outputs = {}
        for horizon_name, head in self.horizon_heads.items():
            outputs[horizon_name] = {
                'logits': head(final_repr)  # (B, num_classes)
            }
        
        return outputs
    
    def generate(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for all horizons
        
        Args:
            x: (batch, seq_len, num_coins, num_channels)
            timestamps: Optional timestamps
            temperature: Sampling temperature
            
        Returns:
            dict with 'horizon_1h', 'horizon_2h', 'horizon_4h', 'horizon_8h'
            Each contains predicted token ID
        """
        outputs = self.forward(x, timestamps)
        
        predictions = {}
        for horizon, output in outputs.items():
            logits = output['logits'] / temperature
            probs = torch.softmax(logits, dim=-1)
            predictions[horizon] = torch.argmax(probs, dim=-1)
        
        return predictions

