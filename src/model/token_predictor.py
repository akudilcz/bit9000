"""Simple Token Predictor - Autoregressive Transformer Decoder

Philosophy:
- Input: 24 hours × N coins × 2 channels (price + volume tokens)
- Output: 1 token (next hour of XRP price), generated autoregressively for 8 hours at inference
- Vocabulary: 256 bins {0-255} for continuous price quantization
- Architecture: Decoder-only with causal masking (like GPT)
- No engineered features, just raw token patterns

Design:
- Separate embeddings for price and volume channels (256 vocab each)
- Channel fusion to combine price and volume information
- Coin aggregation via mean pooling
- Causal self-attention (position i cannot see positions > i)
- Single-step prediction during training (next 1 hour)
- Autoregressive generation at inference time (generates 8 hours by predicting 1 at a time)
- Teacher forcing during training

============ DESIGN VERIFICATION ============
✅ Vocabulary: 256 bins (0-255) - config['model']['vocab_size'] = 256
✅ Input: (batch, 24, 10, 2) - 24h × 10 coins × 2 channels (price+volume)
✅ Output: (batch, 256) - Single next-hour token prediction (256 classes)
✅ Token Embeddings: Separate price_embedding(256→64) and volume_embedding(256→64)
✅ Channel Fusion: Concatenate + Linear(128→256) to get d_model=256
✅ Coin Aggregation: Mean pooling across 10 coins (dim=2)
✅ Positional Encoding: Sinusoidal with max_len=32 (24+8 buffer)
✅ Transformer Decoder: 4 layers, 4 heads, causal masking (tgt_mask upper triangular)
✅ Prediction Head: Linear(256→256) for 256-class output
✅ Training: Single BOS token (128) → predict next 1 token with teacher forcing
✅ Inference: Autoregressive generation loop - slide window for 8 steps
✅ Memory computation: Once per forward, reused across decoder steps
✅ BOS token: Use bin 128 (middle of 256 range) as neutral starting point
✅ Data shapes verified:
   - train_X: (N, 24, 10, 2), dtype=long
   - train_y: (N,), dtype=long (single token per sample)
   - logits: (batch, vocab_size=256) for CrossEntropyLoss
   - generated: (batch, max_length=8) for autoregressive inference
"""

import torch
import torch.nn as nn
import math

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0)
        return self.dropout(x)


class SimpleTokenPredictor(nn.Module):
    """
    Autoregressive transformer decoder for multi-coin, multi-channel token prediction
    
    Architecture:
    1. Token Embedding: Separate embeddings for price and volume channels (256 vocab)
    2. Channel Fusion: Combine price + volume information
    3. Coin Aggregation: Pool across coins at each timestep
    4. Positional Encoding: Add temporal position information
    5. Transformer Decoder: Causal self-attention (decoder-only, like GPT)
    6. Prediction Head: Project to 256-way classification (next 1 hour)
    
    Key Features:
    - Accepts 4D input: (batch, seq_len, num_coins, 2)
    - Causal masking ensures no future information leakage
    - Predicts 1 token (next hour) during training
    - Autoregressive generation at inference (generates 8 hours)
    - Teacher forcing during training
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Architecture parameters
        self.vocab_size = config['model'].get('vocab_size', 256)  # 256 bins
        self.input_length = config['sequences']['input_length']  # 24
        self.output_length = config['sequences']['output_length']  # 1 (single next-hour prediction)
        self.num_coins = len(config['data']['coins'])
        self.num_channels = config['sequences'].get('num_channels', 2)  # price + volume
        
        # Model dimensions
        self.embedding_dim = config['model'].get('embedding_dim', 64)
        self.d_model = config['model'].get('d_model', 256)
        self.nhead = config['model'].get('num_heads', 4)
        self.num_layers = config['model'].get('num_layers', 4)
        self.dim_feedforward = config['model'].get('feedforward_dim', 512)
        self.dropout = config['model'].get('dropout', 0.1)
        
        # Validate dimensions
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        
        # 1. Token Embeddings: Separate for price and volume
        self.price_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.volume_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # 2. Channel Fusion: Combine price + volume embeddings
        # Concatenate [price_emb; volume_emb] then project to d_model
        self.channel_fusion = nn.Linear(self.embedding_dim * 2, self.d_model)
        
        # 3. Coin Aggregation: Project after mean pooling
        self.coin_projection = nn.Linear(self.d_model, self.d_model)
        
        # 4. Positional Encoding
        # Max positions = input_length (24 hours for inference generation)
        max_len = self.input_length + 8  # 24 + 8 for generation buffer
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            max_len=max_len,
            dropout=self.dropout
        )
        
        # 5. Transformer Decoder with Causal Masking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )
        
        # 6. Prediction Head: Project to 256-way classification
        self.prediction_head = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Initialized SimpleTokenPredictor with {total_params:,} parameters")
        logger.info(f"  Input: {self.input_length} hours × {self.num_coins} coins × {self.num_channels} channels")
        logger.info(f"  Output: {self.output_length} hour (next hour prediction, autoregressively generated to {self.output_length + 7} hours)")
        logger.info(f"  Vocab: {self.vocab_size} bins (0-255 for continuous quantization)")
        logger.info(f"  Model dim: {self.d_model}, Heads: {self.nhead}, Layers: {self.num_layers}")
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
        # Upper triangular matrix (excluding diagonal)
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
        
        # 3. Coin Aggregation: Mean pool across coins
        aggregated = fused.mean(dim=2)  # (B, L, d_model)
        processed = self.coin_projection(aggregated)  # (B, L, d_model)
        
        return processed
    
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
            # Shape: (B, 1, num_coins, 2)
            decoder_input = torch.zeros(B, 1, self.num_coins, 2,
                                        dtype=torch.long, device=device)
            decoder_input[:, 0, 0, 0] = decoder_input_tokens[:, 0]  # XRP price channel
            
            # Embed decoder inputs
            decoder_embedded = self._embed_and_process(decoder_input)  # (B, 1, d_model)
            
            # Positional encoding for decoder sequence
            decoder_encoded = self.pos_encoder(decoder_embedded)  # (B, 1, d_model)
            
            # Causal mask for single token (trivial, but included for consistency)
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
            # Encode context once
            memory = self.pos_encoder(self._embed_and_process(x))  # (B, input_length, d_model)
            
            # Current context for sliding window (starts as the initial 24-hour window)
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
                # Create new token tensor for the predicted hour (use as price token)
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


