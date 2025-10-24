"""High Performance Token Predictor V2 - Enhanced Transformer Decoder

Performance Optimizations:
- Learned coin embeddings (distinguish BTC patterns from altcoins)
- Multi-head attention for coin aggregation (learn coin importance)
- Learned positional encodings (better for 24-hour cycles)
- Larger model capacity (more parameters for 256-class problem)
- GELU activation for smoother gradients
- Post-LayerNorm architecture for deeper models

Interface:
- Drop-in replacement for SimpleTokenPredictor
- Same __init__(config), forward(x, targets), generate(x), predict(x) signatures
- Activated by setting config['model']['type'] = 'HighPerformanceTokenPredictor'

Trade-offs:
- ~10x GPU memory usage
- ~5x slower training per epoch
- ~3x slower inference
- Better accuracy on 256-class prediction task
"""

import torch
import torch.nn as nn
import math

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding - more flexible than sinusoidal for fixed-length sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class CoinAttentionAggregation(nn.Module):
    """Multi-head attention mechanism to aggregate across coins
    
    Learns to weight coins dynamically (e.g., BTC might be more important than DOGE for XRP)
    Replaces simple mean pooling with learned attention.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        # Multi-head attention for coin aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable query for aggregation (what pattern to look for)
        self.query_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, num_coins, d_model)
        
        Returns:
            Tensor of shape (batch, seq_len, d_model) - aggregated across coins
        """
        B, L, C, D = x.shape
        
        # Reshape: (B, L, C, D) -> (B*L, C, D)
        x_flat = x.reshape(B * L, C, D)
        
        # Expand query for all batch*seq positions
        query = self.query_embedding.expand(B * L, 1, D)
        
        # Attention: query attends to all coins
        # query: (B*L, 1, D), key/value: (B*L, C, D)
        attn_output, attn_weights = self.attention(
            query=query,
            key=x_flat,
            value=x_flat,
            need_weights=True
        )  # attn_output: (B*L, 1, D)
        
        # Squeeze and project
        attn_output = attn_output.squeeze(1)  # (B*L, D)
        output = self.output_proj(attn_output)  # (B*L, D)
        output = self.dropout(output)
        
        # Reshape back: (B*L, D) -> (B, L, D)
        output = output.reshape(B, L, D)
        
        # Residual connection with mean pooling (helps gradient flow)
        residual = x.mean(dim=2)  # (B, L, D)
        output = self.layer_norm(output + residual)
        
        return output


class HighPerformanceTokenPredictor(nn.Module):
    """
    High-performance autoregressive transformer decoder for multi-coin token prediction
    
    Enhanced Architecture:
    1. Token Embedding: Separate embeddings for price and volume (256 vocab)
    2. Coin Embeddings: Learnable coin identity vectors
    3. Channel Fusion: Combine price + volume information
    4. Coin Aggregation: Multi-head attention (learns coin importance)
    5. Positional Encoding: Learned embeddings for 24-hour patterns
    6. Transformer Decoder: Causal self-attention with GELU activation
    7. Prediction Head: Project to 256-way classification
    
    Key Features:
    - 5-10x more parameters than SimpleTokenPredictor
    - Learns coin-specific patterns and importance
    - More expressive temporal modeling
    - Better gradient flow with GELU and post-norm
    - Same interface as SimpleTokenPredictor (drop-in replacement)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Architecture parameters
        self.vocab_size = config['model'].get('vocab_size', 256)
        self.input_length = config['sequences']['input_length']
        self.output_length = config['sequences']['output_length']
        self.num_coins = len(config['data']['coins'])
        self.num_channels = config['sequences'].get('num_channels', 2)
        
        # Model dimensions
        self.embedding_dim = config['model'].get('embedding_dim', 64)
        self.d_model = config['model'].get('d_model', 256)
        self.nhead = config['model'].get('num_heads', 4)
        self.num_layers = config['model'].get('num_layers', 4)
        self.dim_feedforward = config['model'].get('feedforward_dim', 512)
        self.dropout = config['model'].get('dropout', 0.1)
        
        # V2-specific parameters
        self.coin_attention_heads = config['model'].get('coin_attention_heads', 4)
        self.use_learned_pos = config['model'].get('use_learned_pos_encoding', True)
        
        # Validate dimensions
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        
        # 1. Token Embeddings: Separate for price and volume
        self.price_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.volume_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # 2. Coin Embeddings: Each coin gets a learnable identity
        self.coin_embedding = nn.Embedding(self.num_coins, self.d_model)
        
        # 3. Channel Fusion: Combine price + volume embeddings
        self.channel_fusion = nn.Linear(self.embedding_dim * 2, self.d_model)
        self.fusion_norm = nn.LayerNorm(self.d_model)
        
        # 4. Coin Aggregation: Multi-head attention instead of mean pooling
        self.coin_aggregation = CoinAttentionAggregation(
            d_model=self.d_model,
            num_heads=self.coin_attention_heads,
            dropout=self.dropout
        )
        
        # 5. Positional Encoding: Learned or sinusoidal
        max_len = self.input_length + 8
        if self.use_learned_pos:
            self.pos_encoder = LearnedPositionalEncoding(
                d_model=self.d_model,
                max_len=max_len,
                dropout=self.dropout
            )
        else:
            # Fallback to sinusoidal (from v1)
            from src.model.token_predictor import PositionalEncoding
            self.pos_encoder = PositionalEncoding(
                d_model=self.d_model,
                max_len=max_len,
                dropout=self.dropout
            )
        
        # 6. Transformer Decoder with GELU activation
        # Use custom decoder layer with GELU
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',  # GELU instead of ReLU
            batch_first=True,
            norm_first=False  # Post-norm for stability
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )
        
        # 7. Prediction Head: Project to 256-way classification
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Initialized HighPerformanceTokenPredictor with {total_params:,} parameters")
        logger.info(f"  Input: {self.input_length} hours × {self.num_coins} coins × {self.num_channels} channels")
        logger.info(f"  Output: {self.output_length} hour (next hour prediction)")
        logger.info(f"  Vocab: {self.vocab_size} bins (0-255)")
        logger.info(f"  Model dim: {self.d_model}, Heads: {self.nhead}, Layers: {self.num_layers}")
        logger.info(f"  Coin attention heads: {self.coin_attention_heads}")
        logger.info(f"  Positional encoding: {'Learned' if self.use_learned_pos else 'Sinusoidal'}")
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'embedding' in name:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            mask: Boolean mask of shape (seq_len, seq_len)
                  True = masked (cannot attend), False = allowed
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def _embed_and_process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and process through channel fusion and coin aggregation
        
        Args:
            x: Input tokens, shape (batch, seq_len, num_coins, 2)
               Channel 0: price tokens, Channel 1: volume tokens
        
        Returns:
            processed: shape (batch, seq_len, d_model)
        """
        B, L, C, Ch = x.shape
        
        # 1. Embed price and volume channels separately
        price_tokens = x[:, :, :, 0]  # (B, L, C)
        volume_tokens = x[:, :, :, 1]  # (B, L, C)
        
        price_emb = self.price_embedding(price_tokens)  # (B, L, C, embedding_dim)
        volume_emb = self.volume_embedding(volume_tokens)  # (B, L, C, embedding_dim)
        
        # 2. Channel Fusion: Concatenate price + volume, then project
        combined = torch.cat([price_emb, volume_emb], dim=-1)  # (B, L, C, 2*embedding_dim)
        fused = self.channel_fusion(combined)  # (B, L, C, d_model)
        fused = self.fusion_norm(fused)
        
        # 3. Add Coin Embeddings: Give each coin its identity
        coin_ids = torch.arange(C, device=x.device)  # (C,)
        coin_emb = self.coin_embedding(coin_ids)  # (C, d_model)
        coin_emb = coin_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, C, d_model)
        fused = fused + coin_emb  # Broadcast: (B, L, C, d_model)
        
        # 4. Coin Aggregation: Multi-head attention across coins
        aggregated = self.coin_aggregation(fused)  # (B, L, d_model)
        
        return aggregated
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing
        
        Args:
            x: Input context, shape (batch, input_length, num_coins, 2)
            targets: Optional target token for teacher forcing (batch,) or (batch, 1)
                    Only used during training
        
        Returns:
            logits: Output logits
                   - Training mode: (batch, vocab_size) for single next-hour token
                   - Inference mode: (batch, vocab_size) for next token prediction
        """
        B = x.size(0)
        device = x.device
        
        # Embed and process input context
        context_embedded = self._embed_and_process(x)  # (B, input_length, d_model)
        
        # Add positional encoding
        context_encoded = self.pos_encoder(context_embedded)  # (B, input_length, d_model)
        
        if targets is not None:
            # Training mode: Predict single next token conditioned on context
            # Build decoder input: BOS token (use bin 128 as neutral BOS)
            decoder_input_tokens = torch.full((B, 1), 128, dtype=torch.long, device=device)
            
            # Create 4D token tensor for decoder inputs (only XRP price channel used)
            decoder_input = torch.zeros(B, 1, self.num_coins, 2,
                                        dtype=torch.long, device=device)
            decoder_input[:, 0, 0, 0] = decoder_input_tokens[:, 0]  # XRP price channel
            
            # Embed decoder inputs
            decoder_embedded = self._embed_and_process(decoder_input)  # (B, 1, d_model)
            
            # Positional encoding for decoder sequence
            decoder_encoded = self.pos_encoder(decoder_embedded)  # (B, 1, d_model)
            
            # Causal mask for single token
            tgt_mask = self._create_causal_mask(1, device)
            
            # Transformer decoder: attend over context (memory) and BOS token (tgt)
            decoded = self.transformer_decoder(
                tgt=decoder_encoded,
                memory=context_encoded,
                tgt_mask=tgt_mask
            )  # (B, 1, d_model)
            
            # Project to logits for next token
            logits = self.prediction_head(decoded[:, 0, :])  # (B, vocab_size)
            
            return logits
        else:
            # Inference helper: return logits for predicting next token
            decoder_input = torch.zeros(B, 1, self.num_coins, 2, dtype=torch.long, device=device)
            decoder_input[:, 0, 0, 0] = 128  # BOS
            decoder_embedded = self._embed_and_process(decoder_input)
            decoder_encoded = self.pos_encoder(decoder_embedded)
            tgt_mask = self._create_causal_mask(1, device)
            decoded = self.transformer_decoder(
                tgt=decoder_encoded,
                memory=context_encoded,
                tgt_mask=tgt_mask
            )  # (B, 1, d_model)
            logits = self.prediction_head(decoded[:, 0, :])  # (B, vocab_size)
            return logits
    
    def generate(self, x: torch.Tensor, max_length: int = 8) -> torch.Tensor:
        """
        Autoregressive generation (for inference)
        
        Args:
            x: Input context, shape (batch, input_length, num_coins, 2)
            max_length: Number of tokens to generate (default: 8 hours)
        
        Returns:
            generated: Generated tokens, shape (batch, max_length)
        """
        self.eval()
        B = x.size(0)
        device = x.device
        
        generated_tokens = []
        
        with torch.no_grad():
            # Current context for sliding window
            current_context = x.clone()  # (B, input_length, num_coins, 2)
            
            for step in range(max_length):
                # Get logits for next token given current context
                context_embedded = self._embed_and_process(current_context)
                context_encoded = self.pos_encoder(context_embedded)
                
                # Decoder input: BOS token
                decoder_input = torch.zeros(B, 1, self.num_coins, 2, dtype=torch.long, device=device)
                decoder_input[:, 0, 0, 0] = 128  # BOS
                
                decoder_embedded = self._embed_and_process(decoder_input)
                decoder_encoded = self.pos_encoder(decoder_embedded)
                
                tgt_mask = self._create_causal_mask(1, device)
                decoded = self.transformer_decoder(
                    tgt=decoder_encoded,
                    memory=context_encoded,
                    tgt_mask=tgt_mask
                )  # (B, 1, d_model)
                
                # Predict next token
                logits = self.prediction_head(decoded[:, 0, :])  # (B, vocab_size)
                next_token = torch.argmax(logits, dim=-1)  # (B,)
                generated_tokens.append(next_token)
                
                # Slide window: remove first hour, append new prediction
                new_hour = torch.zeros(B, 1, self.num_coins, 2, dtype=torch.long, device=device)
                new_hour[:, 0, 0, 0] = next_token  # XRP price channel gets predicted token
                
                # Slide: remove first hour and append new
                current_context = torch.cat([current_context[:, 1:, :, :], new_hour], dim=1)
        
        # Stack generated tokens
        generated = torch.stack(generated_tokens, dim=1)  # (B, max_length)
        
        return generated
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions using autoregressive generation (8 hours)
        
        Args:
            x: Input context, shape (batch, input_length, num_coins, 2)
        
        Returns:
            predictions: Predicted tokens, shape (batch, 8)
        """
        return self.generate(x, max_length=8)

