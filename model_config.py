from pydantic.dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int

@dataclass
class ModelConfigMLA:
    dim: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    num_key_value_heads: int
    num_attention_heads: int
    max_seq_len: int