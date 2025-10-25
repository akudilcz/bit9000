"""CryptoTransformerV3 - Encoder-Decoder Architecture with Multi-Task Learning

Philosophy:
- Encoder: Process all 10 coins independently with temporal attention (168h history)
- Decoder: Focus on target coin (XRP), attend to encoder memory for cross-asset information
- Multi-head outputs: classification (256 bins) + optional regression + quantile heads
- Loss: Distance-weighted cross-entropy + auxiliary tasks for ordinal awareness

Architecture Improvements over V1/V2:
- Separate encoder/decoder roles (better than decoder-only)
- Coin embeddings capture asset-specific behavior
- Multi-task learning stabilizes ordinal prediction
- Learned positional encodings scale to 168h sequences

Input:  (batch, 168, 10, 2) - 168h × 10 coins × 2 channels (price+volume)
Output: dict with 'logits' (batch, 256), optional 'regression', 'quantiles'
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
        x = x + self.pe(positions)  # (B, L, D)
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CryptoTransformerV3(nn.Module):
    """Encoder-Decoder Transformer for multi-asset crypto price prediction
    
    Encoder: Processes all coins (multi-asset context)
    Decoder: Predicts target coin, attends to encoder memory
    Multi-task heads: classification + regression + quantiles
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        num_classes: int = 256,
        num_coins: int = 10,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        coin_embedding_dim: int = 32,
        positional_encoding: str = 'learned',
        max_seq_len: int = 256,
        multitask_enabled: bool = True,
        enable_regression: bool = True,
        enable_quantiles: bool = False,
        target_coin_idx: int = 3,  # XRP is at index 3 by default
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.num_coins = num_coins
        self.d_model = d_model
        self.multitask_enabled = multitask_enabled
        self.enable_regression = enable_regression
        self.enable_quantiles = enable_quantiles
        self.target_coin_idx = target_coin_idx
        self.max_seq_len = max_seq_len
        
        # Coin-specific embeddings (learn each coin's behavior)
        self.coin_embedding = nn.Embedding(num_coins, coin_embedding_dim)
        
        # Token embeddings for price and volume (2 channels)
        self.price_embedding = nn.Embedding(vocab_size, d_model // 4)
        self.volume_embedding = nn.Embedding(vocab_size, d_model // 4)
        
        # Channel fusion: combine price + volume + coin embedding
        fusion_dim = (d_model // 4) * 2 + coin_embedding_dim
        self.channel_fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encodings
        if positional_encoding == 'learned':
            self.encoder_pos = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
            self.decoder_pos = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.encoder_pos = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.decoder_pos = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder: process all coins with temporal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Decoder: predict target coin, attend to encoder memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Multi-task heads
        # Primary: 256-class classification
        self.classification_head = nn.Linear(d_model, num_classes)
        
        # Auxiliary: regression head for expected token (continuous)
        if enable_regression:
            self.regression_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # Predict expected token index [0, 255]
            )
        
        # Auxiliary: quantile heads (τ = 0.1, 0.5, 0.9)
        if enable_quantiles:
            self.quantile_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )
                for _ in range(3)  # 3 quantiles
            ])
        
        self._init_weights()
        
        logger.info(f"Initialized CryptoTransformerV3 with {self.count_parameters():,} parameters")
        logger.info(f"  Encoder: {num_encoder_layers} layers, Decoder: {num_decoder_layers} layers")
        logger.info(f"  d_model: {d_model}, nhead: {nhead}, dim_ff: {dim_feedforward}")
        logger.info(f"  Multitask: regression={enable_regression}, quantiles={enable_quantiles}")
        logger.info(f"  Target coin index: {target_coin_idx}")
        logger.info(f"  Max sequence length: {max_seq_len} (encoder can handle up to {max_seq_len // num_coins} timesteps × {num_coins} coins)")
    
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
        Embed price/volume/indicator tokens and add coin embeddings
        
        Args:
            x: (batch, seq_len, num_coins, num_channels) or (batch, seq_len, num_channels) for single coin
               num_channels can be 2 (price, volume) or 5 (price, volume, rsi, macd, bb_position)
            coin_indices: (batch, seq_len, num_coins) or (batch, seq_len) coin IDs
        
        Returns:
            embeddings: (batch, seq_len * num_coins, d_model) or (batch, seq_len, d_model)
        """
        if x.dim() == 4:
            # Multi-coin input: (batch, seq_len, num_coins, num_channels)
            B, L, C, Ch = x.shape
            assert C == self.num_coins, f"Expected {self.num_coins} coins, got {C}"
            
            # Flatten to (batch, seq_len * num_coins, num_channels)
            x_flat = x.reshape(B, L * C, Ch)
            
            # Coin indices: (batch, seq_len * num_coins)
            if coin_indices is None:
                coin_ids = torch.arange(C, device=x.device).repeat(L)  # [0,1,...,C-1, 0,1,...,C-1, ...]
                coin_indices = coin_ids.unsqueeze(0).expand(B, -1)  # (B, L*C)
        else:
            # Single coin input: (batch, seq_len, num_channels)
            B, L, Ch = x.shape
            x_flat = x  # (B, L, num_channels)
            
            if coin_indices is None:
                # Assume target coin (index 0 by convention)
                coin_indices = torch.zeros(B, L, dtype=torch.long, device=x.device)
        
        # Embed each channel (handle both 2-channel and 5-channel inputs)
        # Always use first 2 channels as price and volume
        price_emb = self.price_embedding(x_flat[:, :, 0])  # (B, L*C or L, d_model//4)
        volume_emb = self.volume_embedding(x_flat[:, :, 1])  # (B, L*C or L, d_model//4)
        
        # If we have 5 channels, embed the technical indicators too
        if Ch == 5:
            rsi_emb = self.price_embedding(x_flat[:, :, 2])  # Reuse price embedding for RSI (0-100 range)
            macd_emb = self.volume_embedding(x_flat[:, :, 3])  # Reuse volume embedding for MACD
            bb_emb = self.price_embedding(x_flat[:, :, 4])  # Reuse price embedding for BB position (0-1 range)
        
        # Embed coin IDs
        coin_emb = self.coin_embedding(coin_indices)  # (B, L*C or L, coin_emb_dim)
        
        # Fuse: price + volume + [indicators] + coin
        if Ch == 5:
            # Average the indicator embeddings to keep same dimensionality
            indicator_avg = (rsi_emb + macd_emb + bb_emb) / 3.0  # (B, L*C or L, d_model//4)
            combined = torch.cat([price_emb, volume_emb, indicator_avg, coin_emb], dim=-1)
        else:
            combined = torch.cat([price_emb, volume_emb, coin_emb], dim=-1)
        
        embedded = self.channel_fusion(combined)  # (B, L*C or L, d_model)
        
        return embedded
    
    def forward(
        self,
        x: torch.Tensor,
        target_history: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode all coins, decode target coin
        
        Args:
            x: Encoder input (batch, seq_len, num_coins, 2) - all coins
            target_history: Decoder input (batch, tgt_len, 2) - target coin history
                           If None, uses last timestep of x as BOS
            tgt_mask: Causal mask for decoder (tgt_len, tgt_len)
        
        Returns:
            dict with:
                - 'logits': (batch, num_classes) classification logits
                - 'regression': (batch, 1) expected token (if enabled)
                - 'quantiles': (batch, 3) quantile predictions (if enabled)
        """
        B = x.size(0)
        device = x.device
        
        # Validate sequence length doesn't exceed max_seq_len
        if x.dim() == 4:
            # Multi-coin input: (batch, seq_len, num_coins, 2)
            seq_len = x.size(1)
            effective_seq_len = seq_len * self.num_coins
            if effective_seq_len > self.max_seq_len:
                raise ValueError(
                    f"Encoder input sequence length {effective_seq_len} "
                    f"(seq_len={seq_len} × num_coins={self.num_coins}) "
                    f"exceeds max_seq_len={self.max_seq_len}. Increase max_seq_len parameter."
                )
        
        # 1. Encode: process all coins (multi-asset context)
        encoder_input = self._embed_sequence(x)  # (B, seq_len * num_coins, d_model)
        encoder_input = self.encoder_pos(encoder_input)
        memory = self.encoder(encoder_input)  # (B, seq_len * num_coins, d_model)
        
        # 2. Decode: predict target coin
        if target_history is None:
            # Use BOS token (middle bin for neutral starting point)
            bos_value = self.num_classes // 2  # For 3 classes: 1, for 256 classes: 128
            bos_token = torch.full((B, 1, 2), bos_value, dtype=torch.long, device=device)
            target_history = bos_token
        
        # Embed decoder input (target coin only)
        decoder_input = self._embed_sequence(
            target_history,
            coin_indices=torch.full((B, target_history.size(1)), self.target_coin_idx, dtype=torch.long, device=device)
        )  # (B, tgt_len, d_model)
        decoder_input = self.decoder_pos(decoder_input)
        
        # Create causal mask if needed
        if tgt_mask is None and target_history.size(1) > 1:
            tgt_len = target_history.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device
            )
        
        # Decode with cross-attention to encoder memory
        decoder_output = self.decoder(
            decoder_input,
            memory,
            tgt_mask=tgt_mask
        )  # (B, tgt_len, d_model)
        
        # Use last token for prediction
        final_repr = decoder_output[:, -1, :]  # (B, d_model)
        
        # 3. Multi-task heads
        outputs = {}
        
        # Primary: classification logits
        outputs['logits'] = self.classification_head(final_repr)  # (B, num_classes)
        
        # Auxiliary: regression (expected token index)
        if self.enable_regression:
            outputs['regression'] = self.regression_head(final_repr)  # (B, 1)
        
        # Auxiliary: quantiles (τ = 0.1, 0.5, 0.9)
        if self.enable_quantiles:
            quantiles = torch.cat([
                head(final_repr) for head in self.quantile_heads
            ], dim=-1)  # (B, 3)
            outputs['quantiles'] = quantiles
        
        return outputs
    
    def generate(
        self,
        x: torch.Tensor,
        max_length: int = 8,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive generation for inference
        
        Args:
            x: Encoder input (batch, seq_len, num_coins, 2)
            max_length: Number of future steps to predict
            temperature: Sampling temperature (not used for argmax)
        
        Returns:
            predictions: (batch, max_length) predicted tokens
        """
        self.eval()
        B = x.size(0)
        device = x.device
        
        # Encode once (memory stays fixed)
        encoder_input = self._embed_sequence(x)
        encoder_input = self.encoder_pos(encoder_input)
        memory = self.encoder(encoder_input)
        
        # Start with BOS token
        bos_value = self.num_classes // 2  # For 3 classes: 1, for 256 classes: 128
        generated = torch.full((B, 1, 2), bos_value, dtype=torch.long, device=device)
        predictions = []
        
        for _ in range(max_length):
            # Embed decoder input (target coin only)
            decoder_input = self._embed_sequence(
                generated,
                coin_indices=torch.full((B, generated.size(1)), self.target_coin_idx, dtype=torch.long, device=device)
            )  # (B, tgt_len, d_model)
            decoder_input = self.decoder_pos(decoder_input)
            
            # Create causal mask if needed
            if generated.size(1) > 1:
                tgt_len = generated.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt_len, device=device
                )
            else:
                tgt_mask = None
            
            # Decode with cross-attention to encoder memory
            decoder_output = self.decoder(
                decoder_input,
                memory,
                tgt_mask=tgt_mask
            )  # (B, tgt_len, d_model)
            
            # Use last token for prediction
            final_repr = decoder_output[:, -1, :]  # (B, d_model)
            
            # Get logits
            logits = self.classification_head(final_repr)  # (B, num_classes)
            
            # Sample next token (argmax)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            predictions.append(next_token)
            
            # Append to history (use same token for price and volume)
            next_input = torch.stack([next_token, next_token], dim=-1)  # (B, 1, 2)
            generated = torch.cat([generated, next_input], dim=1)  # (B, seq_len+1, 2)
        
        return torch.cat(predictions, dim=1)  # (B, max_length)

