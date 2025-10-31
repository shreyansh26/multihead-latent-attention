from pydantic.dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    num_heads: int
    num_kv_heads: int
    max_seq_len: int