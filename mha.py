import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import ModelConfig
from rope import precompute_freqs_cis, apply_rotary_emb
from attention import naive_attention, sdpa_attention
from cache import KVCacheMHA

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
        self.head_dim = model_config.head_dim
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads

        self.num_kv_groups = model_config.num_heads // model_config.num_kv_heads

        freqs_cis = precompute_freqs_cis(self.head_dim, model_config.max_seq_len)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model,bias=False, dtype=dtype)

    def forward(self, x_bsd, is_causal=False, kv_cache=None, return_torch_ref=False):
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

        if kv_cache is not None:
            k_bklh_updated, v_bklh_updated = kv_cache.update(k_bklh, v_bklh)
            k_bklh = k_bklh_updated
            v_bklh = v_bklh_updated
        
        out_bsd = naive_attention(q_bqsh, k_bklh, v_bklh, is_causal=is_causal)
        # out_bsd = sdpa_attention(q_bqsh, k_bklh, v_bklh, is_causal=is_causal)
        out_bsd = self.o_proj(out_bsd)

        # Torch implementation
        if return_torch_ref:
            out_bsd_torch = sdpa_attention(q_bqsh, k_bklh, v_bklh, is_causal=is_causal)
            out_bsd_torch = self.o_proj(out_bsd_torch)
            return out_bsd, out_bsd_torch

        return out_bsd

if __name__ == "__main__":
    model_config_mha = ModelConfig(
        d_model=4096,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        max_seq_len=4096
    )

    model_config_gqa = ModelConfig(
        d_model=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=4096
    )

    model_config_mqa = ModelConfig(
        d_model=4096,
        num_heads=32,
        num_kv_heads=1,
        head_dim=128,
        max_seq_len=4096
    )

    dtype = torch.float32
    batch_size = 32
    device = "cuda"
    seq_len = 1024

    # MHA
    model = MHA(model_config_mha, dtype=dtype).to(device)
    x_bsd = torch.randn(batch_size, seq_len, model_config_mha.d_model, dtype=dtype).to(device)
    
    out_bsd, out_bsd_torch = model(x_bsd, is_causal=False, return_torch_ref=True)
    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MHA: All check passed for non-causal case.")
    
    out_bsd, out_bsd_torch = model(x_bsd, is_causal=True, return_torch_ref=True)
    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MHA: All check passed for causal case.")

    # del model, x_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()

    # GQA
    model = MHA(model_config_gqa, dtype=dtype).to(device)
    x_bsd = torch.randn(batch_size, seq_len, model_config_gqa.d_model, dtype=dtype).to(device)
   
    out_bsd, out_bsd_torch = model(x_bsd, is_causal=False, return_torch_ref=True)
    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("GQA: All check passed for non-causal case.")

    out_bsd, out_bsd_torch = model(x_bsd, is_causal=True, return_torch_ref=True)
    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("GQA: All check passed for causal case.")

    # del model, x_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()

    # MQA
    model = MHA(model_config_mqa, dtype=dtype).to(device)
    x_bsd = torch.randn(batch_size, seq_len, model_config_mqa.d_model, dtype=dtype).to(device)
    
    out_bsd, out_bsd_torch = model(x_bsd, is_causal=False, return_torch_ref=True)
    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MQA: All check passed for non-causal case.")

    out_bsd, out_bsd_torch = model(x_bsd, is_causal=True, return_torch_ref=True)
    torch.testing.assert_close(out_bsd, out_bsd_torch, atol=1e-4, rtol=1e-4)
    print("MQA: All check passed for causal case.")

    # del model, x_bsd, out_bsd, out_bsd_torch
    # torch.cuda.empty_cache()

    ## Inference simulation using KV Cache
    prefill_len = 128
    decode_steps = 16

    model = MHA(model_config_gqa, dtype=dtype).to(device)
    kv_cache = KVCacheMHA(batch_size, model_config_gqa.max_seq_len, model_config_gqa, dtype=dtype, device=device)

    # Prefill
    x_prefill = torch.randn(batch_size, prefill_len, model_config_gqa.d_model, dtype=dtype).to(device)
    _ = model(x_prefill, is_causal=True, kv_cache=kv_cache)

    # Decode loop (sequence length = 1 per step)
    for _ in range(decode_steps):
        x_decode = torch.randn(batch_size, 1, model_config_gqa.d_model, dtype=dtype).to(device)
        _ = model(x_decode, is_causal=True, kv_cache=kv_cache)

    assert kv_cache.pos == prefill_len + decode_steps
    print("KV-cache inference simulation: prefill + decode loop completed.")