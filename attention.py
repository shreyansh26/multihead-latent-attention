import math
import torch

def naive_attention(q_bqsh, k_bklh, v_bklh, is_causal=False):
    """
    Naive attention implementation.
    """
    batch_size, seq_len = q_bqsh.shape[0], q_bqsh.shape[2]
    head_dim = q_bqsh.shape[-1]
    num_heads = q_bqsh.shape[1]
    num_kv_heads = k_bklh.shape[1]
    num_kv_groups = num_heads // num_kv_heads

    k_bqlh = k_bklh.repeat_interleave(num_kv_groups, dim=1)
    v_bqlh = v_bklh.repeat_interleave(num_kv_groups, dim=1)

    attn_bqsl = torch.einsum("b q s h, b q l h -> b q s l", q_bqsh, k_bqlh)
    attn_bqsl = attn_bqsl / math.sqrt(head_dim)
    if is_causal:
        mask = torch.triu(torch.ones_like(attn_bqsl, dtype=torch.bool), diagonal=1)
        attn_bqsl = attn_bqsl.masked_fill(mask, float("-inf"))
    attn_bqsl = torch.softmax(attn_bqsl, dim=-1)
    out_bqsh = torch.einsum("b q s l, b q l h -> b q s h", attn_bqsl, v_bqlh)
    out_bsqh = out_bqsh.transpose(1, 2).contiguous()
    out_bsd = out_bsqh.view(batch_size, seq_len, -1)
    return out_bsd

def sdpa_attention(q_bqsh, k_bklh, v_bklh, is_causal=False):
    """
    SDPA attention implementation.
    """
    batch_size, seq_len = q_bqsh.shape[0], q_bqsh.shape[2]
    num_heads = q_bqsh.shape[1]
    num_kv_heads = k_bklh.shape[1]
    num_kv_groups = num_heads // num_kv_heads

    attn_bqsl = torch.nn.functional.scaled_dot_product_attention(q_bqsh, k_bklh, v_bklh, is_causal=is_causal, enable_gqa=num_kv_groups > 1)
    out_bqsh = attn_bqsl.transpose(1, 2).contiguous()
    out_bsd = out_bqsh.view(batch_size, seq_len, -1)
    return out_bsd

if __name__ == "__main__":
    b = 1
    q = 1024
    s = 1024
    h = 128
    l = 1024
    q_bqsh = torch.randn(b, q, s, h, device="cuda")
    k_bqlh = torch.randn(b, q, l, h, device="cuda")
    v_bqlh = torch.randn(b, q, l, h, device="cuda")

    is_causal = False
    out_bsd = naive_attention(q_bqsh, k_bqlh, v_bqlh, is_causal)
    out_bsd_sdpa = sdpa_attention(q_bqsh, k_bqlh, v_bqlh, is_causal)
    torch.testing.assert_close(out_bsd, out_bsd_sdpa)
    print("Naive and SDPA attention are close for non-causal case")

    is_causal = True
    out_bsd = naive_attention(q_bqsh, k_bqlh, v_bqlh, is_causal)
    out_bsd_sdpa = sdpa_attention(q_bqsh, k_bqlh, v_bqlh, is_causal)
    torch.testing.assert_close(out_bsd, out_bsd_sdpa)
    print("Naive and SDPA attention are close for causal case")