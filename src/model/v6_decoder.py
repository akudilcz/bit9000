"""GPTDecoder - Decoder-only transformer for autoregressive next-token prediction

Overview:
- Pure GPT-style architecture with causal masking
- Input: Flattened 1D token sequences with special tokens
- Special tokens: data (0-20), timestep (21-68), coin (69-78)
- Total vocab size: 79 tokens
- Predicts next token at each position (teacher forcing during training)

Architecture:
1. Token embedding (vocab_size -> d_model)
2. Positional embedding (learned, position -> d_model)
3. Stack of decoder layers with causal self-attention
4. Output projection (d_model -> vocab_size)

Generation:
- Autoregressive sampling with temperature and top-k
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GPTDecoder(nn.Module):
    """GPT-style decoder-only transformer for next-token prediction"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        # Configuration
        config = config or {}
        model_config = config.get('model', {})
        
        # Core sizes
        self.vocab_size = int(model_config.get('vocab_size', 79))
        self.d_model = int(model_config.get('d_model', 256))
        self.nhead = int(model_config.get('nhead', 8))
        self.num_decoder_layers = int(model_config.get('num_decoder_layers', 10))
        self.dim_feedforward = int(model_config.get('dim_feedforward', 1024))
        self.dropout_rate = float(model_config.get('dropout', 0.3))
        self.activation = str(model_config.get('activation', 'gelu')).lower()
        self.max_seq_len = int(model_config.get('max_seq_len', 10000))
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Learned positional embedding
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Dropout on embeddings
        self.embed_dropout = nn.Dropout(self.dropout_rate)
        
        # Transformer decoder layers with causal masking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation=self.activation,
            batch_first=True,
            norm_first=True  # Pre-LN for training stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_decoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # Output projection head
        if self.activation == 'gelu':
            act_fn = nn.GELU()
        elif self.activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()
        
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            act_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.vocab_size)
        )
        
        # Trading signal head (BUY/HOLD/SELL)
        self.trading_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            act_fn,
            nn.LayerNorm(self.d_model // 2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, 3)  # 3 classes: BUY(0), HOLD(1), SELL(2)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized GPTDecoder | vocab={self.vocab_size}, d_model={self.d_model}, "
            f"nhead={self.nhead}, layers={self.num_decoder_layers}, ff={self.dim_feedforward}, "
            f"dropout={self.dropout_rate}, activation={self.activation}, with_trading_head=True"
        )
    
    def _init_weights(self):
        """GPT-style weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_trading: bool = False):
        """
        Forward pass with causal masking
        
        Args:
            x: (B, L) long tensor of token indices
            return_trading: If True, return dict with both next-token and trading logits
            
        Returns:
            If return_trading=False:
                logits: (B, L, vocab_size) next-token prediction logits
            If return_trading=True:
                dict with:
                    'logits': (B, L, vocab_size) next-token logits
                    'trading': (B, 3) BUY/HOLD/SELL logits
        """
        B, L = x.shape
        device = x.device
        
        # Create causal mask (upper triangular, prevents attending to future)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            L, device=device
        )
        
        # Position indices
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        
        # Embed: tokens + positions
        token_embeds = self.token_embedding(x)
        pos_embeds = self.pos_embedding(positions)
        embeddings = token_embeds + pos_embeds
        embeddings = self.embed_dropout(embeddings)
        
        # Decoder with causal masking
        # Note: For decoder-only, we use embeddings as both tgt and memory
        # The tgt_mask ensures causal (left-to-right) attention
        # Use gradient checkpointing to reduce memory for long sequences
        if self.training:
            # Checkpointing during training to save memory
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            hidden = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.decoder),
                embeddings,
                embeddings,
                causal_mask,
                use_reentrant=False
            )
        else:
            hidden = self.decoder(
                tgt=embeddings,
                memory=embeddings,
                tgt_mask=causal_mask
            )
        
        # Project to logits
        logits = self.head(hidden)  # (B, L, vocab_size)
        
        if return_trading:
            # Use last timestep's representation for trading signal
            # Last position has seen the entire sequence
            trading_logits = self.trading_head(hidden[:, -1, :])  # (B, 3)
            return {
                'logits': logits,
                'trading': trading_logits
            }
        
        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        context: torch.Tensor, 
        num_steps: int = 200, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation
        
        Args:
            context: (B, L) initial context tokens
            num_steps: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, nucleus sampling (sample from top-p cumulative probability)
            
        Returns:
            Generated sequence: (B, L+num_steps)
        """
        self.eval()
        
        for _ in range(num_steps):
            # Forward pass (only use last max_seq_len tokens if context is too long)
            if context.size(1) > self.max_seq_len:
                context_input = context[:, -self.max_seq_len:]
            else:
                context_input = context
            
            logits = self(context_input)  # (B, L, vocab_size)
            
            # Get logits for last position
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')
            
            # Optional nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Set filtered logits to -inf
                for batch_idx in range(next_logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_logits[batch_idx, indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to context
            context = torch.cat([context, next_token], dim=1)
        
        return context
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

