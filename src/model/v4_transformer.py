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
        self.time_projection = nn.Linear(4, self.d_model)  # 2 for hour + 2 for day
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len) - unix timestamps or datetime indices
            
        Returns:
            time_embeddings: (batch, seq_len, self.d_model)
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
        time_emb = self.time_projection(cyclical_features)  # (batch, seq_len, self.d_model)
        
        return time_emb


class CryptoTransformerV4(nn.Module):
    """Multi-Horizon Transformer with BTC→XRP attention and time features
    
    Key Features:
    - Dedicated BTC encoder (since BTC leads the market)
    - Multi-horizon prediction heads (1h, 2h, 4h, 8h)
    - Time-based features (hour, day, sequence position)
    - Cross-attention from XRP decoder to BTC features
    """
    
    def __init__(self, config: dict = None):
        super().__init__()

        config = config or {}
        model_config = config.get('model', {})

        # Extract all parameters from config with defaults
        self.vocab_size = model_config.get('vocab_size', 256)
        self.num_classes = model_config.get('num_classes', 256)
        self.num_coins = len(config.get('data', {}).get('coins', []))
        self.d_model = model_config.get('d_model', 256)
        self.nhead = model_config.get('nhead', 8)
        self.num_encoder_layers = model_config.get('num_encoder_layers', 3)
        self.num_decoder_layers = model_config.get('num_decoder_layers', 3)
        self.dim_feedforward = model_config.get('dim_feedforward', 1024)
        self.dropout_rate = model_config.get('dropout', 0.3)
        self.coin_embedding_dim = model_config.get('coin_embedding_dim', 32)
        self.max_seq_len = model_config.get('max_seq_len', 1024)

        # Get coin indices from data config
        data_config = config.get('data', {})
        coins = data_config.get('coins', ['BTC', 'ETH', 'LTC', 'XRP'])
        target_coin = data_config.get('target_coin', 'XRP')
        btc_coin = 'BTC'

        try:
            self.target_coin_idx = coins.index(target_coin)
            self.btc_coin_idx = coins.index(btc_coin)
        except ValueError as e:
            raise ValueError(f"Required coin not found in coins list {coins}: {e}")

        # Special parameters
        self.num_channels = config.get('sequences', {}).get('num_channels', 9)

        # Coin-specific embeddings
        self.coin_embedding = nn.Embedding(self.num_coins, self.coin_embedding_dim)
        
        # Token embeddings for configurable channels (price, volume, RSI, MACD, BB position, EMA-9, EMA-21, EMA-50, EMA-ratio)
        # Use smaller embedding dimensions that sum to d_model when combined with coin embedding
        # Ensure we don't lose information by using proper division
        total_channel_dim = self.d_model - self.coin_embedding_dim
        channel_embedding_dim = total_channel_dim // self.num_channels
        remainder = total_channel_dim % self.num_channels

        # Distribute remainder across first few channels to avoid information loss
        channel_dims = [channel_embedding_dim] * self.num_channels
        for i in range(remainder):
            channel_dims[i] += 1
            
        logger.info(f"Channel embedding dimensions: {channel_dims} (total: {sum(channel_dims)})")
        
        self.channel_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, dim) for dim in channel_dims
        ])
        
        # Channel fusion: concatenate all channel embeddings + coin embedding
        fusion_dim = sum(channel_dims) + self.coin_embedding_dim
        self.channel_fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Time-based feature encoding
        self.time_encoding = CyclicalTimeEncoding(self.d_model)
        self.position_encoding = SinusoidalPositionalEncoding(self.d_model, self.dropout_rate)
        
        # Shared encoder: process all coins
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.shared_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # BTC-specific encoder (dedicated pathway)
        self.btc_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,  # Smaller dedicated encoder
            norm=nn.LayerNorm(self.d_model)
        )
        
        # XRP decoder: attends to shared encoder + BTC encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.xrp_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_decoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # BTC→XRP cross-attention (explicit pathway)
        self.btc_to_xrp_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Prediction heads
        if getattr(__import__('builtins'), 'getattr')(self, 'multi_horizon_enabled', None) is None:
            # Read multi_horizon flag from config via constructor? Default to True for backward compat
            # We infer single/multi by creating single head when num_decoder_layers>0 and multi_horizon is disabled in config
            pass

        # For simplicity, derive single/multi from environment: if config sets multi_horizon_enabled False,
        # model factory will still pass num_classes; we keep single head when disabled.
        self.single_horizon = False
        if hasattr(self, 'single_horizon_flag'):
            self.single_horizon = bool(self.single_horizon_flag)
        # Fallback: create single head when expecting binary or single-horizon mode via config flag on instance (set later)
        # To avoid config plumbing here, expose both and higher-level code can use only what's needed.

        # By default create both; single-horizon will use only 'horizon_1h'
        self.horizon_heads = nn.ModuleDict({
            'horizon_1h': self._make_prediction_head(self.d_model, self.num_classes, self.dropout_rate)
        })
        
        self._init_weights()
        
        param_count = self.count_parameters()
        logger.info(f"Initialized CryptoTransformerV4 with {param_count:,} parameters")
        logger.info(f"  Shared Encoder: {self.num_encoder_layers} layers, BTC Encoder: 2 layers, XRP Decoder: {self.num_decoder_layers} layers")
        logger.info(f"  d_model: {self.d_model}, nhead: {self.nhead}, dim_ff: {self.dim_feedforward}")
        logger.info(f"  Prediction head: single-horizon (256-class token prediction)")
        logger.info(f"  BTC→XRP dedicated attention pathway enabled")
        logger.info(f"  Time features: hour, day of week, sequence position")
    
    def _make_prediction_head(self, d_model: int, num_classes: int, dropout: float) -> nn.Module:
        """Create a prediction head for one time horizon"""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_classes)
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
        # Handle configurable channels: price, volume, RSI, MACD, BB position, EMA-9, EMA-21, EMA-50, EMA-ratio
        embedded_channels = []
        for i in range(Ch):  # Only iterate over actual channels present
            if i < len(self.channel_embeddings):  # Safety check
                channel_tokens = x_flat[:, :, i].long()  # Ensure long type for embedding
                embedded_channels.append(self.channel_embeddings[i](channel_tokens))
            else:
                logger.warning(f"Input has {Ch} channels but model expects {len(self.channel_embeddings)} channels")
                break
        
        # Concatenate all channel embeddings
        combined_channel_embeddings = torch.cat(embedded_channels, dim=-1)
        
        # Add coin embedding
        combined = torch.cat([combined_channel_embeddings, self.coin_embedding(coin_indices)], dim=-1)
        
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
            Each contains: {'logits': (batch, self.num_classes)}
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
        # Extract BTC features from encoded shared memory: every C-th element starting from btc_coin_idx
        btc_indices = torch.arange(self.btc_coin_idx, L * C, C, device=x.device)
        btc_features = shared_memory[:, btc_indices, :]  # (B, L, d_model)
        btc_memory = self.btc_encoder(btc_features)  # (B, L, d_model)
        
        # 4. XRP decoder: use last timestep of XRP as query
        # Extract XRP features from encoded shared memory
        xrp_indices = torch.arange(self.target_coin_idx, L * C, C, device=x.device)
        xrp_features = shared_memory[:, xrp_indices, :]  # (B, L, d_model)
        
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

