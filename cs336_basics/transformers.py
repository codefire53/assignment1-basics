from jaxtyping import Float, Array, Integer, Bool
from torch import nn
import torch
from typing import Optional, Union
from einops import einsum, rearrange
import math

def SiLU(x: torch.Tensor)-> torch.Tensor:
    return x * torch.sigmoid(x)

def scaled_dot_product_attention(q: Float[Array, '... seq_len d_model'],
                                 k: Float[Array, '... seq_len d_model'],
                                 v: Float[Array, '... seq_len d_model'],
                                 mask: Optional[Float[Array, '... seq_len seq_len']] = None) -> Float[Array, '... seq_len d_model']:
    attn_scores: Float[Array, '... q_len k_len'] = einsum(q, k, "... i d_model, ... j d_model -> ... i j") / (q.shape[-1] ** 0.5)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(mask == False, float("-inf"))
        # the mask contains inf
        else:
            attn_scores = attn_scores + mask
    
    attn_weights: Float[Array, '... q_len k_len'] = torch.softmax(attn_scores, dim=-1)
    attn_output: Float[Array, '... q_len d_model'] = einsum(attn_weights, v, "... i j, ... j d_model -> ... i d_model")
    return attn_output

# def scaled_dot_product_attention(
#     Q: torch.Tensor,
#     K: torch.Tensor,
#     V: torch.Tensor,
#     mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """
#     计算缩放点积注意力。
#     """
#     # 1. 计算 Q 和 K 的转置的点积，然后缩放
#     d_k = Q.size(-1)
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

#     # 2. 应用掩码 (如果提供)
#     if mask is not None:
#         # --- 这是需要修改的地方 ---
#         # 检查 mask 的类型，以决定如何应用它
#         if mask.dtype == torch.bool:
#             # 如果是布尔掩码，使用 masked_fill
#             scores = scores.masked_fill(mask == False, float('-inf'))
#         else:
#             # 如果是浮点数掩码 (带有 -inf)，直接相加
#             scores = scores + mask

#     # 3. 对缩放后的分数应用你自己的 softmax 函数
#     attn_weights = torch.softmax(scores, dim=-1)

#     # 4. 将注意力权重与 V 相乘
#     output = torch.matmul(attn_weights, V)

#     return output

    

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # initialize weight
        mean = 0.0
        std = (2/(in_features + out_features))**0.5
        left_bound = -3*std
        right_bound = 3*std
        nn.init.trunc_normal_(self.W, mean=mean, std=std, a=left_bound, b=right_bound)

    def forward(self, x: Float[Array, '... in_dim']) -> Float[Array, '... out_dim']:
        return einsum(x, self.W, "... i, j i->... j")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        
        # initialize weight
        mean = 0.0
        std = 1.0
        left_bound = -3*std
        right_bound = 3*std
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=left_bound, b=right_bound)

    def forward(self, token_ids: Integer[Array, '... seq_len']) -> Float[Array, '... seq_len embed_dim']:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Array, '... d_model']) -> Float[Array, '... d_model']:
        # to ensure numerical stability, we do the operation on float32
        original_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        normalized_x = x*rms*self.weights
        return normalized_x.to(original_dtype)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ffn, device=device, dtype=dtype)
        self.linear2 = Linear(d_ffn, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ffn, device=device, dtype=dtype)
    

    def forward(self, x: Float[Array, '... d_model']) -> Float[Array, '... d_model']:
        return self.linear2(SiLU(self.linear1(x))*self.linear3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        inv_freq: Float[Array, "d_k//2"] = 1.0/(self.theta ** (2*torch.arange(d_k//2, device=device, dtype=dtype)/d_k))
        pos: Integer[Array, "max_seq_len"] = torch.arange(max_seq_len, device=device, dtype=dtype)
        rotation_mat: Float[Array, "max_seq_len d_k//2"] = torch.einsum("i,j->ij", pos, inv_freq)

        # store the cos and sin components
        self.register_buffer('rot_cos', rotation_mat.cos(), persistent=False)
        self.register_buffer('rot_sin', rotation_mat.sin(), persistent=False)

    def forward(self, x: Float[Array, "... seq_len d_model"], token_positions: Integer[Array, '... seq_len']) -> Float[Array, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        if token_positions is not None:
            cos = self.rot_cos[token_positions]
            sin = self.rot_sin[token_positions]
        else:
            cos = self.rot_cos[:seq_len]
            sin = self.rot_sin[:seq_len]

        even_pos_x: Float[Array, '... d_model//2'] = x[..., 0::2]
        odd_pos_x: Float[Array, '... d_model//2'] = x[..., 1::2]

        first_component = even_pos_x*cos - odd_pos_x*sin
        second_component = even_pos_x*sin + odd_pos_x*cos

        transformed_x: Float[Array, "... seq_len d_model 2"] = torch.stack((first_component, second_component), dim=-1)
        return rearrange(transformed_x, "... d_model two -> ... (d_model two)")   


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: Optional[nn.Module] = None
                 , device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope
    
    def forward(self, x: Float[Array, '... seq_len d_model'], token_positions: Optional[Integer[Array, '... seq_len']] = None, mask: Optional[torch.Tensor] = None) -> Float[Array, '... seq_len d_model']:
        seq_len = x.shape[-2]
    
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... seq_len (n_comps n_heads head_dim) -> n_comps ... n_heads seq_len head_dim", n_comps=3, n_heads=self.n_heads)

        # Apply RoPE
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # Create causal mask
        if mask is None:
            q_pos: Integer[Array, '... seq_len 1'] = torch.arange(seq_len, device=x.device, dtype=q.dtype).unsqueeze(1)
            k_pos: Integer[Array, '... 1 seq_len'] = torch.arange(seq_len, device=x.device, dtype=q.dtype).unsqueeze(0)
            mask = q_pos >= k_pos

        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # Concatenate heads and project output
        attn_output = rearrange(attn_output, "... n_heads seq_len head_dim -> ... seq_len (n_heads head_dim)", n_heads=self.n_heads, head_dim=self.head_dim)
        return self.out_proj(attn_output)


class TransformersBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: Optional[nn.Module] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
    
    def forward(self, x: Float[Array, '... seq_len d_model'], token_positions: Optional[Integer[Array, '... seq_len']] = None, mask: Optional[torch.Tensor] = None) -> Float[Array, '... seq_len d_model']:
        # Multi-head self-attention
        attn_residual = self.norm1(x)
        attn_residual = self.attention(attn_residual, token_positions=token_positions, mask=mask)
        x = x + attn_residual

        # Feed-forward network
        ffn_residual = self.norm2(x)
        ffn_residual = self.ffn(ffn_residual)
        x = x + ffn_residual
        return x

class TransformersLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
                 , rope_theta: float = 10000.0, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model//num_heads, context_length, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformersBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformersBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
    
    def forward(self, token_ids: Integer[Array, 'batch_size seq_len'], token_positions: Optional[Integer[Array, 'batch_size seq_len']] = None) -> Float[Array, 'batch_size seq_len vocab_size']:
        x = self.embeddings(token_ids)
        
        # create mask
        q_pos = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype).unsqueeze(1)
        k_pos = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype).unsqueeze(0)
        mask = q_pos >= k_pos

        for block in self.transformer_blocks:
            x = block(x, token_positions=token_positions)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits