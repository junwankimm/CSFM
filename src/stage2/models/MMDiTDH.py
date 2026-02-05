### This file contains impls for MM-DiT, the core model component of SD3

import math
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""
    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            dtype=None,
            device=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift, 
        with shift and scale repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift

def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate, 
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scaling_factor=None, offset=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""

    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]

def optimized_attention(qkv, num_heads):
    return attention(qkv[0], qkv[1], qkv[2], num_heads)

class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_mode: str = "xformers",
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        rmsnorm: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(
        self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None
    ):
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
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

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
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning."""

    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device)
        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate="tanh"), dtype=dtype, device=device)
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)

class MMDiTDDTBlock(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=False, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device)
        
        if not rmsnorm:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not swiglu:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate="tanh"), dtype=dtype, device=device)
        else:
            self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6
        else:
            n_mods = 4

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if not self.scale_mod_only:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1) # (B, L, D*6) -> 6 * (B, L, D)
        else:
            shift_msa = None
            shift_mlp = None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)

        # 1. Attention
        attn_in = DDTModulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in) # SelfAttention.forward
        x = x + DDTGate(attn_out, gate_msa)
        
        # 2. MLP
        mlp_in = DDTModulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mlp_in)
        x = x + DDTGate(mlp_out, gate_mlp)
        
        return x
    
class MMDiTDDTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, rmsnorm: bool, total_out_channels: Optional[int] = None, dtype=None, device=None):
        super().__init__()
        if not rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
            if (total_out_channels is None)
            else nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1) # (B, L, D*2) -> 2 * (B, L, D)
        x = DDTModulate(self.norm_final(x), shift, scale) # DDTModulate 사용
        x = self.linear(x)
        return x
    
def block_mixing(context, x, context_block, x_block, c):
    assert context is not None, "block_mixing called with None context"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    x_qkv, x_intermediates = x_block.pre_attention(x, c)

    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    q, k, v = tuple(o)

    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = (attn[:, : context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1] :])

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int] = None, dtype=None, device=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
            if (total_out_channels is None)
            else nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 768,
        depth: int = 28,
        hidden_size: int = 1792,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_config: Optional[Dict] = None,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches = None,
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        dtype = None,
        device = None,
    ):
        super().__init__()
        print(f"mmdit initializing with: {input_size=}, {patch_size=}, {in_channels=}, {depth=}, {mlp_ratio=}, {learn_sigma=}, {adm_in_channels=}, {context_embedder_config=}, {register_length=}, {attn_mode=}, {rmsnorm=}, {scale_mod_only=}, {swiglu=}, {out_channels=}, {pos_embed_scaling_factor=}, {pos_embed_offset=}, {pos_embed_max_size=}, {num_patches=}, {qk_norm=}, {qkv_bias=}, {dtype=}, {device=}")
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size

        if hidden_size != (64 * depth):
            print("[WARNING] MMDiT Expects hidden_size to be 64 * depth. Please check if this is intended.")
        if num_heads != depth:
            print("[WARNING] MMDiT Expects num_heads to be equal to depth. Please check if this is intended.")
        
        # apply magic --> this defines a head_size of 64
        # hidden_size = 64 * depth
        # num_heads = depth

        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=self.pos_embed_max_size is None, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)

        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size, dtype=dtype, device=device)

        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config["target"] == "torch.nn.Linear":
                self.context_embedder = nn.Linear(**context_embedder_config["params"], dtype=dtype, device=device)

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype, device=device))

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # just use a buffer already
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device),
            )
        else:
            self.pos_embed = None

        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=i == depth - 1, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu, qk_norm=qk_norm, dtype=dtype, device=device)
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x: torch.Tensor, c_mod: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.register_length > 0:
            context = torch.cat((repeat(self.register, "1 ... -> b ...", b=x.shape[0]), context if context is not None else torch.Tensor([]).type_as(x)), 1)

        # context is B, L', D
        # x is B, L, D
        for block in self.joint_blocks:
            context, x = block(context, x, c=c_mod)

        x = self.final_layer(x, c_mod)  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        if y is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        context = self.context_embedder(context)

        x = self.forward_core_with_concat(x, c, context)

        x = self.unpatchify(x, hw=hw)  # (N, out_channels, H, W)
        return x
    
    def forward_with_cfg(self, x, t, y=None, context=None, cfg_scale=1.0, cfg_interval=(0,1), **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, context)
        eps, rest = model_out[:, :self.encoder_hidden_size], model_out[:, self.encoder_hidden_size:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, f"cfg_interval should be (min, max) with min < max, but got {cfg_interval}"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def forward_with_autoguidance(self, x, t, y,additional_model_forward, context=None, cfg_scale=1.0, cfg_interval=(-1e4, -1e4), interval_cfg: float = 0.0, **kwargs):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        half = x[: len(x) // 2] # cut the x by half, autoguidance does not need repeated input
        t = t[: len(t) // 2]
        y = y[: len(y) // 2]
        context = context[: len(context) // 2] if context is not None else None
        model_kwargs = { "x": half, "t": t, "y": y, "context": context }
        model_out = self.forward(**model_kwargs)
        ag_model_out = additional_model_forward(**model_kwargs)
        eps, rest = model_out[:, :self.encoder_hidden_size], model_out[:, self.encoder_hidden_size:]
        ag_eps = ag_model_out[:, :self.encoder_hidden_size]
        guid_t_min, guid_t_max = cfg_interval
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        rest = torch.cat([rest, rest], dim=0)
        return torch.cat([eps, rest], dim=1)


class MMDiTwDDTHead(MMDiT):
    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 768,
        depth: List[int] = [28, 2],
        hidden_size: List[int] = [1152, 2048],
        num_heads: List[int] = [16, 16],
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_config: Optional[Dict] = None,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches = None,
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        dtype = None,
        device = None,
        use_pos_proj = False,
        cond_drop_prob = 0.0,
    ):
        
        super().__init__(
            input_size, patch_size, in_channels, depth[0], hidden_size[0], num_heads[0], mlp_ratio, learn_sigma,
            adm_in_channels, context_embedder_config, register_length, attn_mode,
            rmsnorm, scale_mod_only, swiglu, out_channels, pos_embed_scaling_factor,
            pos_embed_offset, pos_embed_max_size, num_patches, qk_norm, qkv_bias,
            dtype, device
        )
        
        print(f"Initializing MMDiTwDDTHead with DDT config:")
        print(f"  {depth=}, {hidden_size=}, {num_heads=}")
        
        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_encoder_blocks = depth[0]
        self.num_decoder_blocks = depth[1]
        self.num_encoder_heads = num_heads[0]
        self.num_decoder_heads = num_heads[1]
        self.use_pos_proj = use_pos_proj
        self.cond_drop_prob = cond_drop_prob
        
        self.encoder_final_norm = self.final_layer.norm_final
        self.encoder_final_modulation = self.final_layer.adaLN_modulation
        
        self.s_projector = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.dec_x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, self.decoder_hidden_size, bias=True, strict_img_size=self.pos_embed_max_size is None, dtype=dtype, device=device)
        
        if self.use_pos_proj:
            self.pos_proj = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=True, dtype=dtype, device=device)
        else:
            self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.decoder_hidden_size, dtype=dtype, device=device))
            
        
        self.decoder_blocks = nn.ModuleList(
            [
                MMDiTDDTBlock(
                    self.decoder_hidden_size,
                    self.num_decoder_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_mode=attn_mode,
                    rmsnorm=rmsnorm,
                    scale_mod_only=scale_mod_only,
                    swiglu=swiglu,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device)
                for _ in range(self.num_decoder_blocks)
            ]
        )
        
        self.final_layer = MMDiTDDTFinalLayer(
            hidden_size=self.decoder_hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
            rmsnorm=rmsnorm,
            dtype=dtype, 
            device=device,
        )
        
        self.initialize_ddt_weights()
    
    def initialize_ddt_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        if hasattr(self, 'y_embedder'):
            nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)
        
        if self.pos_embed is not None:
            grid_size = int(self.x_embedder.num_patches ** 0.5)
            pos_embed_val = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], grid_size
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed_val).float().unsqueeze(0))
        
        for block in self.joint_blocks:
            nn.init.constant_(block.context_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].bias, 0)
            
            nn.init.constant_(block.x_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.encoder_final_modulation[-1].weight, 0)
        nn.init.constant_(self.encoder_final_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        w = self.dec_x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.dec_x_embedder.proj.bias is not None:
            nn.init.constant_(self.dec_x_embedder.proj.bias, 0)
            
        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        if isinstance(self.s_projector, nn.Linear):
            nn.init.xavier_uniform_(self.s_projector.weight)
            if self.s_projector.bias is not None:
                nn.init.constant_(self.s_projector.bias, 0)
        
        if self.use_pos_proj:
            if hasattr(self, 'pos_proj'):
                nn.init.xavier_uniform_(self.pos_proj.weight)
                if self.pos_proj.bias is not None:
                    nn.init.constant_(self.pos_proj.bias, 0)
        else:
            if hasattr(self, 'dec_pos_embed'):
                grid_size = int(self.dec_x_embedder.num_patches ** 0.5)
                dec_pos_embed_val = get_2d_sincos_pos_embed(
                    self.dec_pos_embed.shape[-1], grid_size
                )
                self.dec_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed_val).float().unsqueeze(0))
            
        print("DDT Head weights initialized with adaLN-Zero strategy.")
    
    def cropped_dec_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.dec_x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.dec_pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        y: Optional[torch.Tensor] = None, 
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        hw = x.shape[-2:]

        B = x.shape[0]
        drop_cond = (self.training and self.cond_drop_prob > 0.0) or self.cond_drop_prob == 1.0
        if drop_cond:
            drop_mask = (torch.rand(B, device=x.device) < self.cond_drop_prob)
        else:
            drop_mask = torch.zeros(B, device=x.device, dtype=torch.bool)
        t = self.t_embedder(t, dtype=x.dtype)
        if y is not None:
            if not self.training:
                is_y_null = (y.abs().sum(dim=1, keepdim=True) == 0) #  TODO: assuming 0 null tokens
            y = self.y_embedder(y)
            if drop_cond:
                mask_y = drop_mask.view(B, *([1] * (y.ndim - 1)))
                y = torch.where(mask_y, torch.zeros_like(y), y)
            elif not self.training:
                y = torch.where(is_y_null, torch.zeros_like(y), y)
            #     if drop_mask.any():
            #         y = y.clone()
            #         y[drop_mask] = 0.0 # TODO learnable emb
            enc_c = t + y
        else:
            enc_c = t
        
        if not self.training:
            is_context_null = (context.abs().sum(dim=(1,2), keepdim=True) == 0) # TODO: assuming 0 null tokens
        context_tokens = self.context_embedder(context)
        if drop_cond:
            mask_ctx = drop_mask.view(B, *([1] * (context_tokens.ndim - 1)))
            context_tokens = torch.where(mask_ctx, torch.zeros_like(context_tokens), context_tokens)
        elif not self.training:
            context_tokens = torch.where(is_context_null, torch.zeros_like(context_tokens), context_tokens)
            # if drop_mask.any():
                # context_tokens = context_tokens.clone()
                # context_tokens[drop_mask] = 0.0 # TODO learnable emb
        
        enc_x = self.x_embedder(x)
        enc_pos_embed = self.cropped_pos_embed(hw)
        s = enc_x + enc_pos_embed
        
        if self.register_length > 0:
            context_tokens = torch.cat((repeat(self.register, "1 ... -> b ...", b=x.shape[0]), context_tokens if context_tokens is not None else torch.Tensor([]).type_as(x)), 1)
            
        for block in self.joint_blocks:
            context_tokens, s = block(context_tokens, s, c=enc_c)
        
        # shift, scale = self.main_adaLN_modulation(c_main).chunk(2, dim=1)
        # s = modulate(self.main_norm_final(s), shift, scale)
        shift, scale = self.encoder_final_modulation(enc_c).chunk(2, dim=1)
        s = modulate(self.encoder_final_norm(s), shift, scale)
        s = self.s_projector(s + t.unsqueeze(1))
        
        dec_x = self.dec_x_embedder(x)
        if self.use_pos_proj:
            dec_pos_embed = self.pos_proj(enc_pos_embed)
        else:
            dec_pos_embed = self.cropped_dec_pos_embed(hw)
            dec_x = dec_x + dec_pos_embed
        
        for block in self.decoder_blocks:
            dec_x = block(dec_x, s)
        
        x_out = self.final_layer(dec_x, s)
        x_out = self.unpatchify(x_out, hw=hw)
        
        return x_out

    def forward_with_cfg(self, x, t, y=None, context=None, cfg_scale=1.0, cfg_interval=(0,1), **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, context)
        eps, rest = model_out[:, :self.encoder_hidden_size], model_out[:, self.encoder_hidden_size:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
        

    
        

if __name__ == "__main__":
    from torch.cuda.amp import autocast

    print("\n" + "="*50)
    print("     MMDiTwDDTHead Verification (Dummy Tensors)")
    print("      (Simulating a single text encoder)")
    print("="*50)

    # 1. 설정 (Configuration)
    if torch.cuda.is_available():
        device = "cuda"
        use_autocast = True
        if torch.cuda.is_bf16_supported():
            TEST_DTYPE = torch.bfloat16
            print(f"Device: {device}, Precision: bfloat16 (autocast enabled)")
        else:
            TEST_DTYPE = torch.float16
            print(f"Device: {device}, Precision: float16 (autocast enabled)")
    else:
        device = "cpu"
        TEST_DTYPE = torch.float32
        use_autocast = False
        print(f"Device: {device}, Precision: float32 (autocast disabled)")

    MODEL_DTYPE = torch.float32 
    INPUT_DTYPE = torch.float32

    LATENT_H, LATENT_W = 16, 16 
    IN_CHANNELS = 768
    BATCH_SIZE = 1
    TEXT_EMBED_DIM = 768
    SEQ_LEN = 77
    
    print(f"\nSimulating a single text encoder output:")
    print(f"  - Pooled Embedding Dim ('y'):      {TEXT_EMBED_DIM}")
    print(f"  - Sequence Embedding Dim ('context'): {TEXT_EMBED_DIM}")

    TEST_PATCH_SIZE = 1
    ENCODER_DEPTH = 12
    ENCODER_HIDDEN_SIZE = 768
    ENCODER_HEADS = 12        
    DECODER_DEPTH = 2
    DECODER_HIDDEN_SIZE = 2048 
    DECODER_HEADS = 16
    
    TEST_DEPTH_LIST = [ENCODER_DEPTH, DECODER_DEPTH]
    TEST_HIDDEN_LIST = [ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE]
    TEST_HEADS_LIST = [ENCODER_HEADS, DECODER_HEADS]
    
    MAX_PATCH_GRID_SIZE = LATENT_H // TEST_PATCH_SIZE
    NUM_PATCHES_MAX = MAX_PATCH_GRID_SIZE * MAX_PATCH_GRID_SIZE

    context_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": TEXT_EMBED_DIM, 
            "out_features": ENCODER_HIDDEN_SIZE
        }
    }

    # 2. 모델 초기화
    print(f"\nInitializing MMDiTwDDTHead with max_grid_size={MAX_PATCH_GRID_SIZE}, num_patches={NUM_PATCHES_MAX}")
    model = MMDiTwDDTHead(
        input_size=LATENT_H,
        patch_size=TEST_PATCH_SIZE,
        in_channels=IN_CHANNELS,
        depth=TEST_DEPTH_LIST,
        hidden_size=TEST_HIDDEN_LIST,
        num_heads=TEST_HEADS_LIST,
        adm_in_channels=TEXT_EMBED_DIM,
        context_embedder_config=context_config,
        pos_embed_max_size=MAX_PATCH_GRID_SIZE,
        num_patches=NUM_PATCHES_MAX,
        mlp_ratio=4.0,
        swiglu=True,
        rmsnorm=True,        
        scale_mod_only=False,   
        use_pos_proj=False,
        dtype=MODEL_DTYPE,
        device=device
    ).to(device) 
    
    model.train()
    print("\n -> Model Initialized Successfully (Mode: Train).")
    
    model_param_dtype = next(model.parameters()).dtype
    print(f" -> Model Parameter Dtype: {model_param_dtype}")
    assert model_param_dtype == MODEL_DTYPE

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Total Trainable Parameters: {total_params/1e6:.2f}M")

    # 3. Dummy Data 생성
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=INPUT_DTYPE)
    t = torch.tensor([500] * BATCH_SIZE, device=device)
    y_pooled = torch.randn(BATCH_SIZE, TEXT_EMBED_DIM, device=device, dtype=INPUT_DTYPE)
    context_sequence = torch.randn(BATCH_SIZE, SEQ_LEN, TEXT_EMBED_DIM, device=device, dtype=INPUT_DTYPE)
    
    dummy_target = torch.randn(BATCH_SIZE, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=INPUT_DTYPE)

    print(f" -> Inputs Created (Dtype: {INPUT_DTYPE}):")
    print(f"      x (latent): {x.shape}")
    print(f"      t (timestep): {t.shape}")
    print(f"      y (pooled): {y_pooled.shape}")
    print(f"      context (sequence): {context_sequence.shape}")
    print(f"      dummy_target: {dummy_target.shape}")


    # 4. Forward Pass 실행 (Autocast + Backprop)
    print(f" -> Running Forward Pass (with autocast, target_dtype={TEST_DTYPE})...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    with autocast(enabled=use_autocast, dtype=TEST_DTYPE if use_autocast else None):
        output = model(x, t, y=y_pooled, context=context_sequence)
        
        loss = torch.nn.functional.mse_loss(output.float(), dummy_target.float())

    print(f"Output shape: {output.shape}")
    print(" -> Running Backward Pass...")
    
    loss.backward()
    optimizer.step()

    print(" -> Backward Pass Completed.")
    
    sample_grad = model.final_layer.linear.weight.grad
    if sample_grad is not None:
        print(f" -> Sample Gradient Check (final_layer.linear.weight): PASSED (grad mean: {sample_grad.mean().item()})")
        assert not torch.all(sample_grad == 0), "Gradient is all zeros!"
    else:
        print(" -> Sample Gradient Check: FAILED (grad is None)")
        assert False, "Gradient was not computed"
        
    print("\n" + "="*50)
    print(" -> [Pass] Verification Complete (Backprop OK).")
    print("="*50)