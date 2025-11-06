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