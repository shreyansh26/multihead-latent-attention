import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from model_config import ModelConfig
from rope import precompute_freqs_cis, apply_rotary_emb

"""
Variable suufix nomenclature:

b: batch size
q: number of query heads
k: number of key/value heads
s: sequence length (query)
l: sequence length (key/value)
d: model dimension
h: head dimension
"""

class MHA(nn.Module):
    """
    Class to implement Multi-Head Attention. 
    This can also implement MQA or GQA by setting the number of key/value heads 
    to be different from the number of query heads.
    """
    def __init__(self, model_config: ModelConfig, dtype=torch.bfloat16):
        super().__init__()
        assert model_config.d_model % model_config.num_heads == 0
        assert model_config.d_model % model_config.num_kv_heads == 0

        self.d_model = model_config.d_model
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads

        self.num_kv_groups = model_config.num_heads // model_config.num_kv_heads
        self.head_dim = model_config.d_model // model_config.num_heads
        freqs_cis = precompute_freqs_cis(self.head_dim, model_config.max_seq_len)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model,bias=False, dtype=dtype)

    def forward(self, x_bsd, return_torch_ref=False):
        batch_size, seq_len, d_model = x_bsd.shape
        new_shape = (batch_size, seq_len, -1, self.head_dim) # -1 because num_heads or num_kv_heads
        q_bsqh = self.q_proj(x_bsd).view(new_shape)
        k_blkh = self.k_proj(x_bsd).view(new_shape)
        v_blkh = self.v_proj(x_bsd).view(new_shape)

        q_bsqh = apply_rotary_emb(q_bsqh, self.freqs_cis)
        k_blkh = apply_rotary_emb(k_blkh, self.freqs_cis)

        q_bqsh = q_bsqh.transpose(1, 2)
        k_bklh = k_blkh.transpose(1, 2)
        v_bklh = v_blkh.transpose(1, 2)
        
        # Repeat K/V across query head groups
        k_bqlh = k_bklh.repeat_interleave(self.num_kv_groups, dim=1)
        v_bqlh = v_bklh.repeat_interleave(self.num_kv_groups, dim=1)

        attn_bqsl = einsum(q_bqsh, k_bqlh, "b q s h, b q l h -> b q s l")
        attn_bqsl = attn_bqsl / math.sqrt(self.head_dim)
        attn_bqsl = F.softmax(attn_bqsl, dim=-1)
        out_bqsh = einsum(attn_bqsl, v_bqlh, "b q s l, b q l h -> b q s h")
        out_bsqh = out_bqsh.transpose(1, 2).contiguous()
        out_bsd = out_bsqh.view(batch_size, seq_len, self.d_model)
        out_bsd = self.o_proj(out_bsd)

        # Torch implementation
        if return_torch_ref:
            out_bsd_torch = torch.nn.functional.scaled_dot_product_attention(
                q_bqsh,
                k_bklh,
                v_bklh,
                enable_gqa=self.num_kv_groups > 1,
                is_causal=False,
            )
            out_bsd_torch = out_bsd_torch.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            out_bsd_torch = self.o_proj(out_bsd_torch)
            return out_bsd, out_bsd_torch

        return out_bsd

if __name__ == "__main__":
    model_config_mha = ModelConfig(
        d_model=8192,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=4096,
    )

    model_config_gqa = ModelConfig(
        d_model=8192,
        num_heads=8,
        num_kv_heads=4,
        max_seq_len=4096,
    )

    model_config_mqa = ModelConfig(
        d_model=8192,
        num_heads=8,
        num_kv_heads=1,
        max_seq_len=4096,
    )

    dtype = torch.float32
    batch_size = 32
    device = "cuda"
    seq_len = 1024

    # MHA
    model = MHA(model_config_mha, dtype=dtype).to(device)
    x_bsd = torch.randn(batch_size, seq_len, model_config_mha.d_model, dtype=dtype).to(device)
    out_bsd, out_bsd_torch = model(x_bsd, return_torch_ref=True)

    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MHA: All check passed.")

    # del model, x_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()

    # GQA
    model = MHA(model_config_gqa, dtype=dtype).to(device)
    x_bsd = torch.randn(batch_size, seq_len, model_config_gqa.d_model, dtype=dtype).to(device)
    out_bsd, out_bsd_torch = model(x_bsd, return_torch_ref=True)

    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("GQA: All check passed.")

    # del model, x_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()

    # MQA
    model = MHA(model_config_mqa, dtype=dtype).to(device)
    x_bsd = torch.randn(batch_size, seq_len, model_config_mqa.d_model, dtype=dtype).to(device)
    out_bsd, out_bsd_torch = model(x_bsd, return_torch_ref=True)

    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MQA: All check passed.")

    # del model, x_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()