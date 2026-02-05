# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import List, Optional, Tuple

ENABLE_FLASH_ATTN = False
IS_XLA=False
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    ENABLE_FLASH_ATTN = False # For XLA
    IS_XLA = True # Just for ease of use later

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
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

def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

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
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


#############################################################################
#                               Core NextDiT Model                              #
#############################################################################


class JointAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(
            dim,
            (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.qkv.weight)

        self.out = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.out.weight)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def apply_rotary_emb(
        x_in: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            x_in (torch.Tensor): Query or Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    # copied from huggingface modeling_llama.py
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_local_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            x:
            x_mask:
            freqs_cis:

        Returns:

        """
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        
        attn_dtype = xq.dtype
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq = JointAttention.apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = JointAttention.apply_rotary_emb(xk, freqs_cis=freqs_cis)
        
        if IS_XLA:
            xq, xk, xv = xq.to(attn_dtype), xk.to(attn_dtype), xv.to(attn_dtype)
        else:
            xq, xk = xq.to(dtype), xk.to(dtype)

        softmax_scale = math.sqrt(1 / self.head_dim)

        if dtype in [torch.float16, torch.bfloat16] and ENABLE_FLASH_ATTN:
            # begin var_len flash attn
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(xq, xk, xv, x_mask, seqlen)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )
            output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)
            # end var_len_flash_attn

        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = (
                F.scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_local_heads, seqlen, -1),
                    scale=softmax_scale,
                )
                .permute(0, 2, 1, 3)
            )

        output = output.flatten(-2)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        """
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w1.weight)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w2.weight)
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w3.weight)

    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        modulation=True
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int):
            ffn_dim_multiplier (float):
            norm_eps (float):

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    min(dim, 1024),
                    4 * dim,
                    bias=True,
                ),
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor]=None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    freqs_cis,
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )
            )
        else:
            assert adaln_input is None
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                )
            )
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of NextDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                hidden_size,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self, theta: float = 10000.0, axes_dims: List[int] = (16, 56, 56), axes_lens: List[int] = (1, 512, 512)
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = NextDiT.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)

    def __call__(self, ids: torch.Tensor):
        self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            # import torch.distributed as dist
            # if not dist.is_initialized() or dist.get_rank() == 0:
            #     import pdb
            #     pdb.set_trace()
            index = ids[:, :, i:i+1].repeat(1, 1, self.freqs_cis[i].shape[-1]).to(torch.int64)
            result.append(torch.gather(self.freqs_cis[i].unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1)


class NextDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 384,
        dim: int = 2304,
        n_layers: int = 16,
        n_refiner_layers: int = 2,
        n_heads: int = 24,
        n_kv_heads: int  = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 1024,
        axes_dims: List[int] = (32, 32, 32),
        axes_lens: List[int] = (600, 32, 32),
        cond_drop_prob: float = 0.0,
        use_null_text_embed: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.cond_drop_prob = cond_drop_prob
        self.use_null_text_embed = use_null_text_embed

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
        )
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(
                cap_feat_dim,
                dim,
                bias=True,
            ),
        )
        nn.init.trunc_normal_(self.cap_embedder[1].weight, std=0.02)
        # nn.init.zeros_(self.cap_embedder[1].weight)
        nn.init.zeros_(self.cap_embedder[1].bias)

        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                )
                for layer_id in range(n_layers)
            ]
        )
        # self.norm_final = RMSNorm(dim, eps=norm_eps)
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(axes_dims=axes_dims, axes_lens=axes_lens)
        self.dim = dim
        self.n_heads = n_heads

    def unpatchify(self, x: torch.Tensor, img_token_len: int) -> torch.Tensor:
        bsz = x.shape[0]
        pH = pW = self.patch_size
        
        # Calculate H, W from img_token_len
        H_tokens = W_tokens = int(img_token_len ** 0.5)
        assert H_tokens * W_tokens == img_token_len, "img_token_len must be square"
        
        H, W = H_tokens * pH, W_tokens * pW
        
        # Reshape: (B, H_tokens*W_tokens, pH*pW*C) -> (B, H_tokens, W_tokens, pH, pW, C)
        x = x.view(bsz, H_tokens, W_tokens, pH, pW, self.out_channels)
        
        # Permute: (B, H_tokens, W_tokens, pH, pW, C) -> (B, C, H_tokens, pH, W_tokens, pW)
        x = x.permute(0, 5, 1, 3, 2, 4)
        
        # Reshape: (B, C, H_tokens, pH, W_tokens, pW) -> (B, C, H, W)
        x = x.reshape(bsz, self.out_channels, H, W)
        
        return x

    def patchify_and_embed(
        self, x: torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        bsz, C, H, W = x.shape
        pH = pW = self.patch_size
        device = x.device
        
        H_tokens, W_tokens = H // pH, W // pW
        img_token_len = H_tokens * W_tokens
        cap_token_len = cap_feats.shape[1]
        seq_len = cap_token_len + img_token_len
        
        # Vectorized patchify: (B, C, H, W) -> (B, H_tokens, W_tokens, pH, pW, C)
        x = x.view(bsz, C, H_tokens, pH, W_tokens, pW)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H_tokens, W_tokens, pH, pW, C)
        x = x.flatten(3)  # (B, H_tokens, W_tokens, pH*pW*C)
        x = x.flatten(1, 2)  # (B, img_token_len, pH*pW*C)
        
        # Embed images
        img_embed = self.x_embedder(x)
        
        # Refine images with noise refiner
        img_mask = torch.ones(bsz, img_token_len, dtype=torch.bool, device=device)
        
        # Create position IDs for images (vectorized)
        position_ids = torch.zeros(bsz, seq_len, 3, dtype=torch.int32, device=device)
        
        # Caption position IDs: [0:cap_len] in temporal dim
        cap_temporal = torch.arange(cap_token_len, dtype=torch.int32, device=device)
        position_ids[:, :cap_token_len, 0] = cap_temporal.unsqueeze(0)
        
        # Image position IDs: temporal=cap_len, spatial=(row, col)
        row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
        
        position_ids[:, cap_token_len:, 0] = cap_token_len
        position_ids[:, cap_token_len:, 1] = row_ids
        position_ids[:, cap_token_len:, 2] = col_ids
        
        # Get RoPE embeddings
        freqs_cis = self.rope_embedder(position_ids)
        
        # Split freqs for refiners
        cap_freqs_cis = freqs_cis[:, :cap_token_len]
        img_freqs_cis = freqs_cis[:, cap_token_len:]
        
        # Refine context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)
        
        # Refine image with timestep
        for layer in self.noise_refiner:
            img_embed = layer(img_embed, img_mask, img_freqs_cis, t)
        
        # Concatenate caption and image tokens
        full_embed = torch.cat([cap_feats, img_embed], dim=1)
        full_mask = torch.cat([cap_mask, img_mask], dim=1)
        
        return full_embed, full_mask, freqs_cis, img_token_len


    def forward(self, x, t, context, context_mask, null_context=None, null_context_mask=None, **kwargs):
        """
        Forward pass of NextDiT.
        t: (N,) tensor of diffusion timesteps
        cap_feats: (N, L, D) tensor of caption features
        """
        t = self.t_embedder(t)  # (N, D)
        adaln_input = t
        B = x.shape[0]
        
        drop_cond = (self.training and self.cond_drop_prob > 0.0) or self.cond_drop_prob == 1.0
        if drop_cond:
            drop_mask = torch.rand(B, device=x.device) < self.cond_drop_prob
        else:
            drop_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        
        cap_feats = context
        cap_mask = context_mask
        
        if drop_cond and self.use_null_text_embed:
            assert null_context is not None and null_context_mask is not None
            mask_ctx = drop_mask.view(B, 1, 1)
            cap_feats = torch.where(mask_ctx, torch.zeros_like(cap_feats), cap_feats)
            mask_mask_ctx = drop_mask.view(B, 1)
            cap_mask = torch.where(mask_mask_ctx, null_context_mask, cap_mask)
            
        cap_feats = self.cap_embedder(cap_feats)  # (N, L, D)
        
        if drop_cond and not self.use_null_text_embed:
            mask_ctx = drop_mask.view(B, 1, 1)
            cap_feats = torch.where(mask_ctx, torch.zeros_like(cap_feats), cap_feats)

        # Fixed: patchify_and_embed returns 4 values
        x, mask, freqs_cis, img_token_len = self.patchify_and_embed(x, cap_feats, cap_mask, t)
        freqs_cis = freqs_cis.to(x.device)

        # Encoder layers
        for layer in self.layers:
            x = layer(x, mask, freqs_cis, adaln_input)

        # Final layer on full sequence
        x = self.final_layer(x, adaln_input)
        
        # Extract only image tokens for unpatchify
        cap_token_len = cap_feats.shape[1]
        x_img = x[:, cap_token_len:cap_token_len+img_token_len]
        
        # Fixed: unpatchify takes (x, img_token_len)
        x = self.unpatchify(x_img, img_token_len)

        return x

    def forward_with_cfg(self, x, t, cfg_scale, context, context_mask, null_context=None, null_context_mask=None, cfg_trunc=100, renorm_cfg=0, **kwargs):
        half = x[: len(x) // 2]
        if t[0] < cfg_trunc:
            combined = torch.cat([half, half], dim=0) # [2, 16, 128, 128]
            model_out = self.forward(combined, t, context, context_mask, null_context, null_context_mask)
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)  
            if float(renorm_cfg) > 0.0: 
                raise NotImplementedError("Renormalization with cfg is not implemented yet.")
        else:
            raise NotImplementedError("CFG without truncation is not implemented for NextDiTwDDTHead.")

        output = torch.cat([half_eps, half_eps], dim=0)
        return output

    @staticmethod
    def precompute_freqs_cis(
        dim: List[int],
        end: List[int],
        theta: float = 10000.0,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (list): Dimension of the frequency tensor.
            end (list): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """
        freqs_cis = []
        for i, (d, e) in enumerate(zip(dim, end)):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#############################################################################
#                                 NextDiT Configs                               #
#############################################################################

def NextDiT_2B_GQA_patch2_Adaln_Refiner(**kwargs):
    return NextDiT(
        patch_size=2,
        dim=2304,
        n_layers=26,
        n_heads=24,
        n_kv_heads=8,
        axes_dims=[32, 32, 32],
        axes_lens=[300, 512, 512],
        **kwargs
    )

def NextDiT_3B_GQA_patch2_Adaln_Refiner(**kwargs):
    return NextDiT(
        patch_size=2,
        dim=2592,
        n_layers=30,
        n_heads=24,
        n_kv_heads=8,
        axes_dims=[36, 36, 36],
        axes_lens=[300, 512, 512],
        **kwargs,
    )

def NextDiT_4B_GQA_patch2_Adaln_Refiner(**kwargs):
    return NextDiT(
        patch_size=2,
        dim=2880,
        n_layers=32,
        n_heads=24,
        n_kv_heads=8,
        axes_dims=[40, 40, 40],
        axes_lens=[300, 512, 512],
        **kwargs,
    )

def NextDiT_7B_GQA_patch2_Adaln_Refiner(**kwargs):
    return NextDiT(
        patch_size=2,
        dim=3840,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        axes_dims=[40, 40, 40],
        axes_lens=[300, 512, 512],
        **kwargs,
    )