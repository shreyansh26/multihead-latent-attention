## Multi-Head Latent Attention (MLA)

**Blog Link** - [https://shreyansh26.github.io/post/2025-11-08_multihead-latent-attention/](https://shreyansh26.github.io/post/2025-11-08_multihead-latent-attention/)

A small, self-contained reference implementation of:
- **MHA/GQA/MQA** in `mha.py`
- **MLA** plus fused and absorbed variants in `mla.py`

Both use Rotary Positional Embeddings (RoPE), support causal/non‑causal attention, and include simple cache-based decode simulations.

### Repository layout

```text
multihead-latent-attention/
├── attention.py          # naive_attention, sdpa_attention
├── cache.py              # KVCacheMHA, CacheMLA
├── mha.py                # MHA/GQA/MQA
├── mla.py                # MLA, MLAFused, MLAFusedAbsorbed
├── model_config.py       # ModelConfig, ModelConfigMLA
└── rope.py               # RoPE utilities
```

## Requirements

- Python 3.10+
- PyTorch >= 2.6 (CUDA build recommended for GPU)
- GPU with sufficient memory for the example shapes

## Installation

Install PyTorch per your CUDA setup, then any local deps:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126  # choose the right CUDA wheel
```

## Quickstart

From the project root directory:

```bash
python mha.py
python mla.py
```

Each script:
- Builds a model with a reasonable demo config
- Runs a forward pass
- Demonstrates a prefill + decode loop using the cache utilities

## MHA (including GQA/MQA)

`mha.py` exposes class `MHA` with configurable query heads and KV heads.

Key shapes (b=batch, s=query len, l=kv len, h=head dim):
- q: `[b, num_heads, s, h]`
- k/v: `[b, num_kv_heads, l, h]`

Constructor arguments are provided via `ModelConfig` in `model_config.py`:

```python
from mha import MHA
from model_config import ModelConfig

cfg = ModelConfig(
    d_model=4096,
    num_heads=32,
    num_kv_heads=8,   # =32 for MHA, <32 for GQA, =1 for MQA
    head_dim=128,
    max_seq_len=4096,
)
model = MHA(cfg, dtype=torch.bfloat16).to("cuda")
```

Forward usage:

```python
out = model(x_bsd, is_causal=True, kv_cache=kv_cache)  # kv_cache optional
```

## MLA (and fused/absorbed variants)

`mla.py` contains three modules:
- `MLA`: baseline decomposition with separate projections
- `MLAFused`: fuses some projections to reduce ops/memory traffic
- `MLAFusedAbsorbed`: absorbs `W^{UK}`/`W^{UV}` to avoid materializing decompressed K/V during inference

Configuration is via `ModelConfigMLA` in `model_config.py`. Typical fields:
- `dim`, `q_lora_rank`, `kv_lora_rank`
- `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`
- `num_key_value_heads`, `num_attention_heads`, `max_seq_len`

Example:

```python
from mla import MLA, MLAFused, MLAFusedAbsorbed
from model_config import ModelConfigMLA

cfg = ModelConfigMLA(
    dim=7168,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    num_key_value_heads=128,
    num_attention_heads=128,
    max_seq_len=163840,
)
model = MLAFusedAbsorbed(cfg, dtype=torch.bfloat16).to("cuda")
```

Forward usage mirrors MHA:

```python
out = model(x_bsd, cache=cache, is_causal=True)  # cache optional
```

### Caching: prefill + decode

Both `mha.py` and `mla.py` include minimal, runnable examples of:
1) A prefill pass over the prompt
2) A decode loop with `seq_len=1` per step

## Attention backends

Two interchangeable implementations exist in `attention.py`:
- `naive_attention`: straightforward reference
- `sdpa_attention`: PyTorch SDPA path for speed

Each file shows how to toggle the backend (comment/uncomment one line).

## dType and devices

- Default examples use `torch.bfloat16` on CUDA for speed.
- You can switch to `torch.float32` if you’re on CPU or debugging numerical issues.

## Troubleshooting

- Out of memory (OOM): reduce `batch_size`, `seq_len`, head counts, or ranks.
- CUDA errors: verify the installed PyTorch wheel matches your CUDA runtime.