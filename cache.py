import torch
import torch.nn as nn

from model_config import ModelConfig

class KVCacheMHA():
    def __init__(self, batch_size, max_tokens, model_config: ModelConfig, dtype=torch.bfloat16, device="cuda"):
        cache_shape = (batch_size, model_config.num_kv_heads, max_tokens, model_config.head_dim)
        self.k_cache = torch.empty(cache_shape, device=device)
        self.v_cache = torch.empty(cache_shape, device=device)
        self.pos = 0
        
    def update(self, k, v):
        len_seq = k.shape[2]
        self.k_cache[:, :, self.pos: self.pos + len_seq] = k
        self.v_cache[:, :, self.pos: self.pos + len_seq] = v
        self.pos += len_seq
        return self.k_cache[:, :, :self.pos], self.v_cache[:, :, :self.pos]

class _CompressedKVCache():
    def __init__(self, batch_size, max_tokens, kv_lora_rank, dtype=torch.bfloat16, device="cuda"):
        self.cache = torch.empty((batch_size, max_tokens, kv_lora_rank), dtype=dtype, device=device)
        self.pos = 0
        
    def update(self, compressed_kv):
        len_seq = compressed_kv.shape[1]
        self.cache[:, self.pos: self.pos + len_seq] = compressed_kv
        self.pos += len_seq
        return self.cache[:, :self.pos]

class _KRopeCache():
    def __init__(self, batch_size, max_tokens, qk_rope_head_dim, dtype=torch.bfloat16, device="cuda"):
        self.cache = torch.empty((batch_size, 1, max_tokens, qk_rope_head_dim), dtype=dtype, device=device)
        self.pos = 0
        
    def update(self, k_rope):
        len_seq = k_rope.shape[2]
        self.cache[:, :, self.pos: self.pos + len_seq] = k_rope
        self.pos += len_seq
        return self.cache[:, :, :self.pos]

class CacheMLA():
    def __init__(self, batch_size, max_tokens, model_config, dtype=torch.bfloat16, device="cuda"):
        self.compressed_kv = _CompressedKVCache(batch_size, max_tokens, model_config.kv_lora_rank, dtype=dtype, device=device)
        self.k_rope = _KRopeCache(batch_size, max_tokens, model_config.qk_rope_head_dim, dtype=dtype, device=device)
        
    @property
    def pos(self):
        return self.compressed_kv.pos