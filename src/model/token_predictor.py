"""Simple Token Predictor - Autoregressive Transformer Decoder

Philosophy:
- Input: 24 hours × N coins × 2 channels (price + volume tokens)
- Output: 8 tokens (next 8 hours of XRP price), generated autoregressively
- Vocabulary: 3 tokens {down=0, steady=1, up=2} per channel
- Architecture: Decoder-only with causal masking (like GPT)
- No engineered features, just raw token patterns

Design:
- Separate embeddings for price and volume channels
- Channel fusion to combine price and volume information
- Coin aggregation via mean pooling
- Causal self-attention (position i cannot see positions > i)
- Autoregressive generation at inference time
- Teacher forcing during training
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
    1. Token Embedding: Separate embeddings for price and volume channels
    2. Channel Fusion: Combine price + volume information
    3. Coin Aggregation: Pool across coins at each timestep
    4. Positional Encoding: Add temporal position information
    5. Transformer Decoder: Causal self-attention (decoder-only, like GPT)
    6. Prediction Head: Project to 3-way classification per step
    
    Key Features:
    - Accepts 4D input: (batch, seq_len, num_coins, 2)
    - Causal masking ensures no future information leakage
    - Autoregressive generation at inference
    - Teacher forcing during training
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Architecture parameters
        self.vocab_size = 3  # Always 3 tokens: down, steady, up
        self.input_length = config['sequences']['input_length']  # 24
        self.output_length = config['sequences']['output_length']  # 8
        self.num_coins = len(config['data']['coins'])
        self.num_channels = config['sequences'].get('num_channels', 2)  # price + volume
        
        # Model dimensions
        self.embedding_dim = config['model'].get('embedding_dim', 64)
        self.d_model = config['model'].get('d_model', 256)
        self.nhead = config['model'].get('num_heads', 4)
        self.num_layers = config['model'].get('num_layers', 4)
        self.dim_feedforward = config['model'].get('feedforward_dim', 256)
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
        # Max positions = input_length + output_length (for autoregressive generation)
        max_len = self.input_length + self.output_length
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
        
        # 6. Prediction Head: Project to 3-way classification
        self.prediction_head = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Initialized SimpleTokenPredictor with {total_params:,} parameters")
        logger.info(f"  Input: {self.input_length} hours × {self.num_coins} coins × {self.num_channels} channels")
        logger.info(f"  Output: {self.output_length} hours (autoregressive)")
        logger.info(f"  Vocab: {self.vocab_size} tokens per channel")
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
            targets: Optional target tokens for teacher forcing (batch, output_length)
                    Only used during training
        
        Returns:
            logits: Output logits
                   - Training mode: (batch, output_length, vocab_size)
                   - Inference mode: (batch, 1, vocab_size) for next token
        """
        B = x.size(0)
        device = x.device
        
        # Embed and process input context
        context_embedded = self._embed_and_process(x)  # (B, input_length, d_model)
        
        # Add positional encoding
        context_encoded = self.pos_encoder(context_embedded)  # (B, input_length, d_model)
        
        if targets is not None:
            # Training mode: Proper teacher forcing
            # Predict targets conditioned on context and previous target tokens (shifted)
            target_length = targets.size(1)

            # Build decoder input tokens by shifting targets right with a BOS token
            # Use class 'steady' (1) as BOS to avoid changing vocab size
            decoder_input_tokens = torch.full((B, target_length), 1, dtype=torch.long, device=device)
            decoder_input_tokens[:, 1:] = targets[:, :-1]

            # Create 4D token tensor for decoder inputs (only XRP price channel used)
            # Shape: (B, target_length, num_coins, 2)
            decoder_input = torch.zeros(B, target_length, self.num_coins, 2,
                                        dtype=torch.long, device=device)
            decoder_input[:, :, 0, 0] = decoder_input_tokens  # XRP price channel

            # Embed decoder inputs
            decoder_embedded = self._embed_and_process(decoder_input)  # (B, target_length, d_model)

            # Positional encoding for decoder sequence
            decoder_encoded = self.pos_encoder(decoder_embedded)  # (B, target_length, d_model)

            # Causal mask over decoder sequence
            tgt_mask = self._create_causal_mask(target_length, device)

            # Transformer decoder: attend over context (memory) and masked decoder inputs (tgt)
            decoded = self.transformer_decoder(
                tgt=decoder_encoded,
                memory=context_encoded,
                tgt_mask=tgt_mask
            )  # (B, target_length, d_model)

            # Project to logits for each target step
            logits = self.prediction_head(decoded)  # (B, target_length, vocab_size)

            return logits
        else:
            # Inference helper: return logits for a single BOS token as decoder input
            # This path is not used for full generation; see generate()
            decoder_input = torch.zeros(B, 1, self.num_coins, 2, dtype=torch.long, device=device)
            decoder_input[:, 0, 0, 0] = 1  # BOS = 'steady'
            decoder_embedded = self._embed_and_process(decoder_input)
            decoder_encoded = self.pos_encoder(decoder_embedded)
            tgt_mask = self._create_causal_mask(1, device)
            decoded = self.transformer_decoder(
                tgt=decoder_encoded,
                memory=context_encoded,
                tgt_mask=tgt_mask
            )  # (B, 1, d_model)
            logits = self.prediction_head(decoded)  # (B, 1, vocab)
            return logits
    
    def generate(self, x: torch.Tensor, max_length: int = None) -> torch.Tensor:
        """
        Autoregressive generation (for inference)
        
        Args:
            x: Input context, shape (batch, input_length, num_coins, 2)
            max_length: Number of tokens to generate (default: output_length)
        
        Returns:
            generated: Generated tokens, shape (batch, max_length)
        """
        if max_length is None:
            max_length = self.output_length
        
        self.eval()
        B = x.size(0)
        device = x.device
        
        generated_tokens = []
        
        with torch.no_grad():
            # Encode context once
            memory = self.pos_encoder(self._embed_and_process(x))  # (B, input_length, d_model)

            # Decoder input tokens (start with BOS)
            decoder_tokens = torch.full((B, 1), 1, dtype=torch.long, device=device)  # BOS='steady'

            for step in range(max_length):
                # Build 4D decoder input from tokens
                dec_len = decoder_tokens.size(1)
                decoder_input = torch.zeros(B, dec_len, self.num_coins, 2, dtype=torch.long, device=device)
                decoder_input[:, :, 0, 0] = decoder_tokens

                # Embed + position encode
                decoder_embedded = self._embed_and_process(decoder_input)
                decoder_encoded = self.pos_encoder(decoder_embedded)

                # Mask and decode
                tgt_mask = self._create_causal_mask(dec_len, device)
                decoded = self.transformer_decoder(
                    tgt=decoder_encoded,
                    memory=memory,
                    tgt_mask=tgt_mask
                )  # (B, dec_len, d_model)

                # Predict next token from last position
                logits = self.prediction_head(decoded[:, -1:, :])  # (B, 1, vocab)
                next_token = torch.argmax(logits[:, 0, :], dim=-1)  # (B,)
                generated_tokens.append(next_token)

                # Append for next iteration
                decoder_tokens = torch.cat([decoder_tokens, next_token.unsqueeze(1)], dim=1)
        
        # Stack generated tokens (exclude initial BOS)
        generated = torch.stack(generated_tokens, dim=1)  # (B, max_length)
        
        return generated
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions using autoregressive generation
        
        Args:
            x: Input context, shape (batch, input_length, num_coins, 2)
        
        Returns:
            predictions: Predicted tokens, shape (batch, output_length)
        """
        return self.generate(x, max_length=self.output_length)
    
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction probabilities via autoregressive generation
        
        Args:
            x: Input context, shape (batch, input_length, num_coins, 2)
        
        Returns:
            probs: Prediction probabilities, shape (batch, output_length, vocab_size)
        """
        self.eval()
        B = x.size(0)
        device = x.device
        
        all_probs = []
        
        with torch.no_grad():
            current_input = x
            
            for step in range(self.output_length):
                logits = self.forward(current_input)  # (B, 1, vocab_size)
                probs = torch.softmax(logits[:, 0, :], dim=-1)  # (B, vocab_size)
                all_probs.append(probs)
                
                # Sample next token for next iteration
                next_token = torch.argmax(probs, dim=-1)  # (B,)
                
                # TODO: Update current_input with next_token
        
        # Stack probabilities
        all_probs = torch.stack(all_probs, dim=1)  # (B, output_length, vocab_size)
        
        return all_probs


