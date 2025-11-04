import torch
from mla import MLA, MLAFused, ModelConfigMLA

torch.manual_seed(0)
cfg = ModelConfigMLA(
    dim=7168, q_lora_rank=1536, kv_lora_rank=512,
    qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
    num_key_value_heads=128, num_attention_heads=128, max_seq_len=4096,
)
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

mla = MLA(cfg, dtype=dtype).to(device)
mlag = MLAFused(cfg, dtype=dtype).to(device)

# Copy/fuse weights
with torch.no_grad():
    mlag.w_dkv_kr.weight.copy_(torch.cat([mla.w_dkv.weight, mla.w_kr.weight], dim=0))
    mlag.w_uk_uv.weight.copy_(torch.cat([mla.w_uk.weight, mla.w_uv.weight], dim=0))
    mlag.w_qr_uq.weight.copy_(torch.cat([mla.w_qr.weight, mla.w_uq.weight], dim=0))
    mlag.w_o.weight.copy_(mla.w_o.weight)

B, S = 2, 256
x = torch.randn(B, S, cfg.dim, dtype=dtype, device=device)
out_a = mla(x, is_causal=False)
out_b = mlag(x, is_causal=False)

# Allow tiny numeric noise; bfloat16 may need a higher tol
diff = (out_a.float() - out_b.float()).abs().max().item()
print("max_abs_diff:", diff)
assert diff < 1e-1, f"Mismatch: {diff}"