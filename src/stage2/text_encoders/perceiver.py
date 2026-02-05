"""
perceiver.py

A modular Perceiver-based Variational Encoder (TextVE) built using
modern Transformer blocks (RMSNorm, SwiGLU) based on the provided DDT
components.

This module is designed to translate variable-length 1D text embeddings
(e.g., CLIP, Llama) into a fixed-length 2D grid of latent representations
(e.g., for RAE/DINO space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable
import copy
import numpy as np
from functools import partial
from timm.models.vision_transformer import Mlp

# --- Helper Modules (Copied from provided DDT code) ---

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# --- Positional Embedding Helpers (from provided DDT code) ---

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    # The (embed_dim // 2 - 1) denominator is a common variant, 
    # but the provided DDT code uses embed_dim / 2.
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2. 
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # use half of dimensions to encode grid_w
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        # This logic seems specific to DINO's register tokens,
        # for our queries, we just need the grid.
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class NormAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype) # rope may change the q,k's dtype
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- Main Perceiver Variational Encoder ---

class PerceiverLayer(nn.Module):
    """
    A single Perceiver layer, combining Cross-Attention and Self-Attention.
    BUILT USING DDT/TIMM COMPONENTS.
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 dropout: float = 0.1,
                 use_rmsnorm: bool = True, 
                 use_swiglu: bool = True,
                 use_qknorm: bool = False,
                 ):
        super().__init__()
        
        NormLayer = partial(RMSNorm, eps=1e-6) if use_rmsnorm else partial(nn.LayerNorm, eps=1e-6)
        # 1. Cross-Attention components
        self.norm_queries = NormLayer(d_model)
        self.norm_context = NormLayer(d_model)
        # Use standard nn.MultiheadAttention for Cross-Attention (Q != K,V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_cross = nn.Dropout(dropout)

        # 2. Self-Attention components (DDT-style Pre-Norm)
        self.norm_sa = NormLayer(d_model)
        self.self_attn = NormAttention(
            dim=d_model,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            attn_drop=dropout,
            proj_drop=dropout,
            norm_layer=NormLayer, # Pass NormLayer constructor
            use_rmsnorm=use_rmsnorm
        )
        self.dropout_sa = nn.Dropout(dropout)
        
        # 3. FFN components (DDT-style Pre-Norm)
        self.norm_ffn = NormLayer(d_model)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # Use SwiGLU FFN. Calculate hidden dim based on DDT's heuristic
            hidden_features = int(d_ff * (2/3))
            self.ffn = SwiGLUFFN(
                in_features=d_model, 
                hidden_features=hidden_features, 
                out_features=d_model, 
                drop=dropout
            )
        else:
            # Fallback to standard GELU Mlp
            self.ffn = Mlp(
                in_features=d_model, 
                hidden_features=d_ff, 
                act_layer=approx_gelu,
                drop=dropout
            )

    def forward(self, 
                queries: torch.Tensor, 
                context: torch.Tensor, 
                context_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        # 1. Cross-Attention (Pre-Norm)
        attn_output, _ = self.cross_attn(
            query=self.norm_queries(queries),
            key=self.norm_context(context),
            value=self.norm_context(context),
            key_padding_mask=context_key_padding_mask
        )
        queries = queries + self.dropout_cross(attn_output) # Residual 1

        # 2. Self-Attention (Pre-Norm, using NormAttention)
        # NormAttention does not apply pre-norm itself, so we apply it
        sa_output = self.self_attn(self.norm_sa(queries), rope=None) 
        queries = queries + self.dropout_sa(sa_output) # Residual 2
        
        # 3. FFN (Pre-Norm)
        ffn_output = self.ffn(self.norm_ffn(queries))
        queries = queries + ffn_output # Residual 3
        
        return queries


class PerceiverVE(nn.Module):
    """
    Perceiver-based Variational Encoder (TextVE).
    Uses modern Transformer blocks (RMSNorm, SwiGLU) from DDT code.
    """
    def __init__(self,
                 in_channels: int,        
                 out_channels: int,       
                 hidden_size: int,        
                 depth: int = 4,          
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,  
                 dropout_prob: float = 0.1,
                 num_queries: int = 256,
                 max_input_len: int = 4096,
                 pos_emb_type: str = 'learned',
                 use_qknorm: bool = False,
                 use_swiglu: bool = True,
                 use_rmsnorm: bool = True,
                 use_variational: bool = True,
                 use_per_token_logvar: bool = False,
                 init_logvar: float = 0.0,
                 layernorm_mean: bool = False,
                 fixed_std : float = None,
                 ):
        """
        Args:
            in_channels (int): Dim of input text embeddings (e.g., 4096 for Llama).
            out_channels (int): Dim of target RAE latent space (e.g., 768 for DINO).
            hidden_size (int): Internal hidden dim of the Perceiver (e.g., 1024).
            depth (int): Number of Perceiver layers.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Factor to expand hidden_size for FFN (e.g., 4.0).
            dropout_prob (float): Dropout rate.
            num_queries (int): Number of output latent tokens (e.g., 256 for 16x16).
            max_input_len (int): Max sequence length for 1D text positional embeddings.
            pos_emb_type (str): Type of 2D pos. embedding for queries: 'learned' or 'fixed'.
            use_qknorm (bool): Whether to use QKNorm in Self-Attention.
            use_swiglu (bool): Whether to use SwiGLU FFN (recommended).
            use_rmsnorm (bool): Whether to use RMSNorm (recommended).
        """
        super().__init__()
            
        d_ff = int(hidden_size * mlp_ratio)
        NormLayer = partial(RMSNorm, eps=1e-6) if use_rmsnorm else partial(nn.LayerNorm, eps=1e-6)
            
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.use_variational = use_variational
        self.use_per_token_logvar = use_per_token_logvar
        self.init_logvar = init_logvar
        self.layernorm_mean = layernorm_mean
        self.fixed_std = fixed_std
        # 1. Input Projection (Adapter)
        self.input_proj = nn.Linear(in_channels, hidden_size)
        
        # 2. Positional Embeddings
        self.input_pos_emb = nn.Parameter(torch.randn(1, max_input_len, hidden_size))
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        
        self.grid_size = int(math.sqrt(num_queries))
        if self.grid_size * self.grid_size != num_queries:
            raise ValueError(f"num_queries ({num_queries}) must be a perfect square.")
        
        self.pos_emb_type = pos_emb_type
        if pos_emb_type == 'learned':
            self.query_pos_emb_2d = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        elif pos_emb_type == 'sincos':
            pos_emb = get_2d_sincos_pos_embed(hidden_size, self.grid_size)
            self.register_buffer('query_pos_emb_2d', torch.from_numpy(pos_emb).float().unsqueeze(0))
        else:
            raise ValueError(f"Unknown pos_emb_type: {pos_emb_type}. Must be 'learned' or 'sincos'.")

        # 3. Perceiver Body
        self.layers = clones(
            PerceiverLayer(
                d_model=hidden_size, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                dropout=dropout_prob, 
                use_rmsnorm=use_rmsnorm, 
                use_swiglu=use_swiglu,
                use_qknorm=use_qknorm
            ),
            depth
        )
        
        # 4. Output Heads
        self.norm = NormLayer(hidden_size)
        self.mean_head = nn.Linear(hidden_size, out_channels)
        if self.use_variational and self.fixed_std is None:
            if use_per_token_logvar:
                self.log_var_head = nn.Linear(hidden_size, 1)
            else:
                self.log_var_head = nn.Linear(hidden_size, out_channels)
        else:
            self.log_var_head = None

        self.init_weights()

    def init_weights(self):
        # Initialize positional embeddings and query tokens
        nn.init.trunc_normal_(self.input_pos_emb, std=0.02)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        if self.pos_emb_type == 'learned':
            nn.init.trunc_normal_(self.query_pos_emb_2d, std=0.02)
        
        # Initialize all Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        if self.use_variational and self.log_var_head is not None:
            nn.init.normal_(self.log_var_head.weight, std=1e-4)
            nn.init.constant_(self.log_var_head.bias, self.init_logvar)

    def reparameterize(self, mean: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        if log_var is not None and self.use_variational:
            std = torch.exp(0.5 * log_var)
            noise = torch.randn_like(mean) # for broadcasting std -> mean
            return mean + noise * std
        
        return mean

    def forward(self, 
                text_tokens: torch.Tensor, 
                text_key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, L_text, _ = text_tokens.shape
        device = text_tokens.device
        
        context = self.input_proj(text_tokens)
        context = context + self.input_pos_emb[:, :L_text, :].to(device)
        
        queries = self.query_tokens.expand(batch_size, -1, -1)
        queries = queries + self.query_pos_emb_2d.to(device)
        
        for layer in self.layers:
            queries = layer(queries, context, text_key_padding_mask)
            
        queries_norm = self.norm(queries)
        mean = self.mean_head(queries_norm)
        if self.layernorm_mean:
            mean = F.layer_norm(mean, mean.shape[-1:])
        if self.use_variational:
            if self.fixed_std is None:
                log_var = self.log_var_head(queries_norm)
            else:
                log_var = torch.full_like(mean, math.log(self.fixed_std ** 2))
        else:
            log_var = None
        
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var


# --- 디버깅 코드 ---
if __name__ == "__main__":
    
    # --- 1. YAML Config와 동일한 설정으로 테스트 ---
    print("--- 1. 시나리오: YAML Config (CLIP(768) -> RAE(768)) ---")
    config = {
        'in_channels': 768,
        'out_channels': 768,
        'hidden_size': 768,
        'depth': 4,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout_prob': 0.1,
        'num_queries': 256,
        'max_input_len': 77,
        'pos_emb_type': 'learned',
        'use_qknorm': False,
        'use_swiglu': True,
        'use_rmsnorm': True
    }
    
    model_clip = PerceiverVE(**config)
    
    total_params_clip = sum(p.numel() for p in model_clip.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters (CLIP Model, SwiGLU, RMSNorm): {total_params_clip / 1_000_000:.2f}M")
    print("-" * 30)

    BATCH_SIZE = 4
    L_TEXT_CLIP = 77
    dummy_clip_tokens = torch.randn(BATCH_SIZE, L_TEXT_CLIP, config['in_channels'])
    
    model_clip.train()
    mean_clip, log_var_clip = model_clip(dummy_clip_tokens)
    z_clip = model_clip.reparameterize(mean_clip, log_var_clip)
    
    print(f"Output mean shape:    {list(mean_clip.shape)}")
    print(f"Sampled z shape:      {list(z_clip.shape)}")
    assert list(mean_clip.shape) == [BATCH_SIZE, config['num_queries'], config['out_channels']]
    print("--- 시나리오 1 통과 ---")

    # --- 2. Llama(4096) -> RAE(768) 시나리오 테스트 (GELU, LayerNorm) ---
    print("\n" + "--- 2. 시나리오: Llama(4096) -> RAE(768) (GELU, LayerNorm) ---")
    
    model_llama = PerceiverVE(
        in_channels=4096,
        out_channels=768,
        hidden_size=1024,
        num_queries=256,
        depth=2, 
        num_heads=16,
        max_input_len=1024,
        pos_emb_type='sincos',
        use_swiglu=False,       # GELU 사용
        use_rmsnorm=False       # LayerNorm 사용
    )
    
    total_params_llama = sum(p.numel() for p in model_llama.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters (Llama Model, GELU, LayerNorm): {total_params_llama / 1_000_000:.2f}M")
    
    L_TEXT_LLAMA = 1024
    dummy_llama_tokens = torch.randn(BATCH_SIZE, L_TEXT_LLAMA, 4096)
    
    model_llama.train()
    mean_llama, log_var_llama = model_llama(dummy_llama_tokens)
    
    print(f"Output mean shape:    {list(mean_llama.shape)}")
    assert list(mean_llama.shape) == [BATCH_SIZE, 256, 768]
    print("--- 시나리오 2 통과 ---")