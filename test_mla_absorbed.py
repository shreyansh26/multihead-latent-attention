import torch
from mla import MLA, MLAFusedAbsorbed
from model_config import ModelConfigMLA
from cache import CacheMLA

def fuse_weights_for_absorption(mla, mla_absorbed):
    """
    Fuse weights from MLA into MLAFusedAbsorbed.
    
    Key insight:
    - w_uq_absorbed = (W^{UK})^T @ W^{UQ} for each attention head
    - w_o_absorbed = W^O @ W^{UV} with proper head mapping
    """
    with torch.no_grad():
        # Copy shared weights
        mla_absorbed.w_dkv.weight.copy_(mla.w_dkv.weight)
        mla_absorbed.w_kr.weight.copy_(mla.w_kr.weight)
        mla_absorbed.w_dq.weight.copy_(mla.w_dq.weight)
        mla_absorbed.w_qr.weight.copy_(mla.w_qr.weight)
        
        # Fuse w_uq_absorbed: for each attention head, compute (W^{UK})^T @ W^{UQ}
        qk_nope_head_dim = mla.qk_nope_head_dim
        kv_lora_rank = mla_absorbed.kv_lora_rank
        num_attention_heads = mla.num_attention_heads
        num_kv_groups = mla.num_kv_groups
        
        w_uq_absorbed_weight = torch.zeros_like(mla_absorbed.w_uq_absorbed.weight)
        
        for head_i in range(num_attention_heads):
            kv_head_j = head_i // num_kv_groups
            
            # Extract W^{UQ} slice for attention head i: [qk_nope_head_dim, q_lora_rank]
            w_uq_slice = mla.w_uq.weight[head_i * qk_nope_head_dim:(head_i + 1) * qk_nope_head_dim, :]
            
            # Extract W^{UK} slice for kv_head j: [qk_nope_head_dim, kv_lora_rank]
            w_uk_slice = mla.w_uk.weight[kv_head_j * qk_nope_head_dim:(kv_head_j + 1) * qk_nope_head_dim, :]
            
            # Compute absorbed weight: (W^{UK})^T @ W^{UQ} = [kv_lora_rank, q_lora_rank]
            w_absorbed = w_uk_slice.T @ w_uq_slice  # [kv_lora_rank, q_lora_rank]
            
            # Assign to w_uq_absorbed
            w_uq_absorbed_weight[head_i * kv_lora_rank:(head_i + 1) * kv_lora_rank, :] = w_absorbed
        
        mla_absorbed.w_uq_absorbed.weight.copy_(w_uq_absorbed_weight)
        
        # Fuse w_o_absorbed: for each attention head, compute W^O @ W^{UV}
        v_head_dim = mla.v_head_dim
        dim = mla.w_o.weight.shape[0]
        
        w_o_absorbed_weight = torch.zeros_like(mla_absorbed.w_o_absorbed.weight)
        
        for head_i in range(num_attention_heads):
            kv_head_j = head_i // num_kv_groups
            
            # Extract W^O slice for attention head i: [dim, v_head_dim]
            w_o_slice = mla.w_o.weight[:, head_i * v_head_dim:(head_i + 1) * v_head_dim]
            
            # Extract W^{UV} slice for kv_head j: [v_head_dim, kv_lora_rank]
            w_uv_slice = mla.w_uv.weight[kv_head_j * v_head_dim:(kv_head_j + 1) * v_head_dim, :]
            
            # Compute absorbed weight: W^O @ W^{UV} = [dim, kv_lora_rank]
            w_absorbed = w_o_slice @ w_uv_slice  # [dim, kv_lora_rank]
            
            # Assign to w_o_absorbed
            w_o_absorbed_weight[:, head_i * kv_lora_rank:(head_i + 1) * kv_lora_rank] = w_absorbed
        
        mla_absorbed.w_o_absorbed.weight.copy_(w_o_absorbed_weight)

def test_forward_no_cache():
    """Test forward pass without cache."""
    print("\n=== Test 1: Forward pass without cache ===")
    
    torch.manual_seed(42)
    cfg = ModelConfigMLA(
        dim=512,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        num_key_value_heads=4,
        num_attention_heads=8,
        max_seq_len=4096,
    )
    
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create models
    mla = MLA(cfg, dtype=dtype).to(device)
    mla_absorbed = MLAFusedAbsorbed(cfg, dtype=dtype).to(device)
    
    # Fuse weights
    fuse_weights_for_absorption(mla, mla_absorbed)
    
    # Test forward
    B, S = 2, 64
    x = torch.randn(B, S, cfg.dim, dtype=dtype, device=device)
    
    with torch.no_grad():
        out_mla = mla(x, is_causal=False)
        out_absorbed = mla_absorbed(x, is_causal=False)
    
    diff = (out_mla.float() - out_absorbed.float()).abs().max().item()
    print(f"Max absolute difference: {diff:.6e}")
    
    # Use a reasonable tolerance for bfloat16
    assert diff < 1e-1, f"Forward pass mismatch: {diff}"
    print("✓ Forward pass without cache: PASSED")

def test_forward_with_cache():
    """Test forward pass with cache (prefill + decode)."""
    print("\n=== Test 2: Forward pass with cache ===")
    
    torch.manual_seed(42)
    cfg = ModelConfigMLA(
        dim=512,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        num_key_value_heads=4,
        num_attention_heads=8,
        max_seq_len=4096,
    )
    
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create models
    mla = MLA(cfg, dtype=dtype).to(device)
    mla_absorbed = MLAFusedAbsorbed(cfg, dtype=dtype).to(device)
    
    # Fuse weights
    fuse_weights_for_absorption(mla, mla_absorbed)
    
    # Test with cache
    B = 2
    prefill_len = 32
    decode_steps = 8
    
    # Create separate caches
    cache_mla = CacheMLA(B, prefill_len + decode_steps, cfg, dtype=dtype, device=device)
    cache_absorbed = CacheMLA(B, prefill_len + decode_steps, cfg, dtype=dtype, device=device)
    
    # Prefill
    x_prefill = torch.randn(B, prefill_len, cfg.dim, dtype=dtype, device=device)
    
    with torch.no_grad():
        out_mla_prefill = mla(x_prefill, cache=cache_mla, is_causal=True)
        out_absorbed_prefill = mla_absorbed(x_prefill, cache=cache_absorbed, is_causal=True)
    
    diff_prefill = (out_mla_prefill.float() - out_absorbed_prefill.float()).abs().max().item()
    print(f"Prefill max absolute difference: {diff_prefill:.6e}")
    assert diff_prefill < 1e-1, f"Prefill mismatch: {diff_prefill}"
    
    # Decode loop
    max_decode_diff = 0.0
    for step in range(decode_steps):
        x_decode = torch.randn(B, 1, cfg.dim, dtype=dtype, device=device)
        
        with torch.no_grad():
            out_mla_decode = mla(x_decode, cache=cache_mla, is_causal=True)
            out_absorbed_decode = mla_absorbed(x_decode, cache=cache_absorbed, is_causal=True)
        
        diff = (out_mla_decode.float() - out_absorbed_decode.float()).abs().max().item()
        max_decode_diff = max(max_decode_diff, diff)
    
    print(f"Decode max absolute difference: {max_decode_diff:.6e}")
    assert max_decode_diff < 1e-1, f"Decode mismatch: {max_decode_diff}"
    
    assert cache_mla.pos == prefill_len + decode_steps
    assert cache_absorbed.pos == prefill_len + decode_steps
    
    print("✓ Forward pass with cache (prefill + decode): PASSED")

def test_large_config():
    """Test with larger, more realistic configuration."""
    print("\n=== Test 3: Large configuration ===")
    
    torch.manual_seed(42)
    cfg = ModelConfigMLA(
        dim=7168,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        num_key_value_heads=128,
        num_attention_heads=128,
        max_seq_len=4096,
    )
    
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create models
    mla = MLA(cfg, dtype=dtype).to(device)
    mla_absorbed = MLAFusedAbsorbed(cfg, dtype=dtype).to(device)
    
    # Fuse weights
    fuse_weights_for_absorption(mla, mla_absorbed)
    
    # Test forward
    B, S = 2, 64
    x = torch.randn(B, S, cfg.dim, dtype=dtype, device=device)
    
    with torch.no_grad():
        out_mla = mla(x, is_causal=False)
        out_absorbed = mla_absorbed(x, is_causal=False)
    
    diff = (out_mla.float() - out_absorbed.float()).abs().max().item()
    print(f"Max absolute difference: {diff:.6e}")
    assert diff < 1e-1, f"Large config mismatch: {diff}"
    print("✓ Large configuration test: PASSED")

if __name__ == "__main__":
    print("Testing MLAFusedAbsorbed implementation")
    print("=" * 50)
    
    test_forward_no_cache()
    test_forward_with_cache()
    test_large_config()
    
    print("\n" + "=" * 50)
    print("All tests PASSED! ✓")
    print("\nMLAFusedAbsorbed correctly implements the absorption optimization.")
    print("Keys and values remain in compressed latent space during inference.")