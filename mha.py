import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

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
    def __init__(self, d_model, num_heads, num_kv_heads, dtype=torch.bfloat16):
        super().__init__()
        assert d_model % num_heads == 0
        assert d_model % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model,bias=False, dtype=dtype)

    def forward(self, h_bsd, return_torch_ref=False):
        batch_size, seq_len, d_model = h_bsd.shape
        new_shape = (batch_size, seq_len, -1, self.head_dim) # -1 because num_heads or num_kv_heads
        q_bqsh = self.q_proj(h_bsd).view(new_shape).transpose(1, 2)
        k_bklh = self.k_proj(h_bsd).view(new_shape).transpose(1, 2)
        v_bklh = self.v_proj(h_bsd).view(new_shape).transpose(1, 2)
        
        # Repeat K/V across query head groups
        k_bqlh = k_bklh.repeat_interleave(self.num_kv_groups, dim=1)
        v_bqlh = v_bklh.repeat_interleave(self.num_kv_groups, dim=1)

        attn_bqsl = einsum(q_bqsh, k_bqlh, "b q s h, b q l h -> b q s l")
        attn_bqsl = attn_bqsl / math.sqrt(self.head_dim)
        attn_bqsl = F.softmax(attn_bqsl, dim=-1)
        out_bqsh = einsum(attn_bqsl, v_bqlh, "b q s l, b q l h -> b q s h")
        out_bsd = out_bqsh.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
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
            out_bsd_torch = out_bsd_torch.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            out_bsd_torch = self.o_proj(out_bsd_torch)
            return out_bsd, out_bsd_torch

        return out_bsd

if __name__ == "__main__":
    batch_size = 32
    d_model = 8192
    seq_len = 1024
    num_heads = 8
    num_kv_heads = 4
    dtype = torch.float32
    device = "cuda"

    # MHA
    model = MHA(d_model, num_heads=num_heads, num_kv_heads=num_heads, dtype=dtype).to(device)
    h_bsd = torch.randn(batch_size, seq_len, d_model, dtype=dtype).to(device)
    out_bsd, out_bsd_torch = model(h_bsd, return_torch_ref=True)

    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MHA: All check passed.")

    # del model, h_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()

    # GQA
    model = MHA(d_model, num_heads=num_heads, num_kv_heads=num_kv_heads, dtype=dtype).to(device)
    h_bsd = torch.randn(batch_size, seq_len, d_model, dtype=dtype).to(device)
    out_bsd, out_bsd_torch = model(h_bsd, return_torch_ref=True)

    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("GQA: All check passed.")

    # del model, h_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()

    # MQA
    model = MHA(d_model, num_heads=num_heads, num_kv_heads=1, dtype=dtype).to(device)
    h_bsd = torch.randn(batch_size, seq_len, d_model, dtype=dtype).to(device)
    out_bsd, out_bsd_torch = model(h_bsd, return_torch_ref=True)

    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MQA: All check passed.")

    # del model, h_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()