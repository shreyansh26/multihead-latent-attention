import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import ModelConfigMLA
from rope import precompute_freqs_cis, apply_rotary_emb
from attention import naive_attention

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

class MLA(nn.Module):
    """
    Class to implement Multi-Head Latent Attention. 
    """
    def __init__(self, model_config: ModelConfigMLA, dtype=torch.bfloat16):
        super().__init__()
        self.qk_rope_head_dim = model_config.qk_rope_head_dim
        self.qk_nope_head_dim = model_config.qk_nope_head_dim
        self.v_head_dim = model_config.v_head_dim
        self.seq_len = model_config.max_seq_len
        self.num_key_value_heads = model_config.num_key_value_heads
        self.num_attention_heads = model_config.num_attention_heads
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads

        freqs_cis_qk = precompute_freqs_cis(self.qk_rope_head_dim, self.seq_len)
        self.register_buffer('freqs_cis_qk', freqs_cis_qk, persistent=False)

        self.w_dkv = nn.Linear(model_config.dim, model_config.kv_lora_rank, bias=False, dtype=dtype)
        self.w_kr = nn.Linear(model_config.kv_lora_rank, self.qk_rope_head_dim, bias=False, dtype=dtype)
        self.w_uk = nn.Linear(model_config.kv_lora_rank, self.num_key_value_heads * self.qk_nope_head_dim, bias=False, dtype=dtype)
        self.w_uv = nn.Linear(model_config.kv_lora_rank, self.num_key_value_heads * self.v_head_dim, bias=False, dtype=dtype)

        self.w_dq = nn.Linear(model_config.dim, model_config.q_lora_rank, bias=False, dtype=dtype)
        self.w_qr = nn.Linear(model_config.q_lora_rank, self.num_attention_heads * self.qk_rope_head_dim, bias=False, dtype=dtype)
        self.w_uq = nn.Linear(model_config.q_lora_rank, self.num_attention_heads * self.qk_nope_head_dim, bias=False, dtype=dtype)

        self.w_o = nn.Linear(self.num_attention_heads * self.v_head_dim, model_config.dim, bias=False, dtype=dtype)
        

    def forward(self, x_bsd, is_causal=False):
        batch_size, seq_len, d_model = x_bsd.shape
        
        c_kv = self.w_dkv(x_bsd) # [B, S, kv_lora_rank]
        c_q = self.w_dq(x_bsd) # [B, S, q_lora_rank]

        # Not needed to be done explicitly
        # k_c = self.w_uk(c_kv)
        # v_c = self.w_uv(c_kv)

        k_r = self.w_kr(x_bsd) # [B, S, qk_rope_head_dim]
        k_r = k_r.view(batch_size, seq_len, 1, self.qk_rope_head_dim)
        k_r = apply_rotary_emb(k_r, self.freqs_cis_qk)
        k_r = k_r.transpose(1, 2) # [B, 1, S, qk_rope_head_dim]
        k_r = k_r.repeat_interleave(self.num_key_value_heads, dim=1) # [B, num_key_value_heads, S, qk_rope_head_dim]

        k_n = self.w_uk(c_kv) # [B, S, num_key_value_heads * qk_nope_head_dim]
        k_n = k_n.view(batch_size, seq_len, self.num_key_value_heads, self.qk_nope_head_dim)
        k_n = k_n.transpose(1, 2) # [B, num_key_value_heads, S, qk_nope_head_dim]

        k = torch.cat([k_r, k_n], dim=-1) # [B, num_key_value_heads, S, (qk_rope_head_dim + qk_nope_head_dim)]

        q_r = self.w_qr(c_q) # [B, S, num_attention_heads * qk_rope_head_dim]
        q_r = q_r.view(batch_size, seq_len, self.num_attention_heads, self.qk_rope_head_dim)
        q_r = apply_rotary_emb(q_r, self.freqs_cis_qk)
        q_r = q_r.transpose(1, 2) #[B, num_attention_heads, S, qk_rope_head_dim]

        q_n = self.w_uq(c_q) # [B, S, num_attention_heads * qk_nope_head_dim]
        q_n = q_n.view(batch_size, seq_len, self.num_attention_heads, self.qk_nope_head_dim)
        q_n = q_n.transpose(1, 2) #[B, num_attention_heads, S, qk_nope_head_dim]
        
        q = torch.cat([q_r, q_n], dim=-1) # [B, num_attention_heads, S, (qk_rope_head_dim + qk_nope_head_dim)]

        v = self.w_uv(c_kv) # [B, S, num_key_value_heads * v_head_dim]
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.v_head_dim)
        v = v.transpose(1, 2) # [B, num_key_value_heads, S, v_head_dim]

        # k = k.repeat_interleave(self.num_kv_groups, dim=1)
        # v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out_bsd = naive_attention(q, k, v, is_causal=is_causal)
        out_bsd = self.w_o(out_bsd)

        return out_bsd

if __name__ == "__main__":
    model_config_mla = ModelConfigMLA(
        dim=7168,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        num_key_value_heads=129,
        num_attention_heads=128,
        max_seq_len=163840,
    )

    dtype = torch.bfloat16
    batch_size = 4
    device = "cuda"
    seq_len = 1024

    # MLA
    model = MLA(model_config_mla, dtype=dtype).to(device)
    x_bsd = torch.randn(batch_size, seq_len, model_config_mla.dim, dtype=dtype).to(device)
    out_bsd = model(x_bsd)
    print(out_bsd.shape)