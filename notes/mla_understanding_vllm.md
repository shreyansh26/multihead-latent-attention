# Understanding multihead latent attention in vllm
_Exported on 25/10/2025 at 0:05:46 GMT-7 from Cursor (1.7.54)_

---

**User**

In @triton_mla.py - I don't see any multihead latent attention speciifc logic - the complicated processing of q, k, v etc. Where is that implemented in vllm? I want to understand how MLA is used and (the underflywing flow which makes it work) in vllm

---

**Cursor**

I'll scan the repo to locate the MLA-specific implementation and the q/k/v processing path, then pull the key files and symbol definitions so I can cite exact code locations.

I'll pull the exact code for the MLA flow: where q/kv are formed in the model, where the common MLA forward splits prefill/decode and computes ql_nope and caches, and where the Triton decode kernel is called. Then I'll cite the code locations directly.

I searched the vLLM tree for MLA and traced the end-to-end path. I’ll point you to the exact files and lines where q/k/v are formed, cached, transformed, and where the Triton kernel is invoked.

### Where the MLA math actually lives
- The “complicated processing” is in the common MLA impl, not the small `triton_mla.py` wrapper. See: weight processing (`W_UK_T`, `W_UV`), prefill computation of K/V from latent, decode formation of `ql_nope`, cache concat, attention calls, and the final up-projection.

Code references:
```960:1003:vllm/v1/attention/backends/mla/common.py
        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0)
```

```1084:1118:vllm/v1/attention/backends/mla/common.py
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = self._run_prefill_new_tokens(
            prefill=attn_metadata.prefill,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # unpad if necessary
        if self._pad_v:
            output = output[..., :v.shape[-1]]

        return output.flatten(start_dim=-2)
```

```1200:1226:vllm/v1/attention/backends/mla/common.py
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            if fp8_attention:
                ql_nope_shape = decode_ql_nope.shape
                decode_ql_nope, _ = ops.scaled_fp8_quant(
                    decode_ql_nope.reshape([
                        ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]
                    ]), layer._q_scale)
                decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
                q_pe_shape = decode_q_pe.shape
                decode_q_pe, _ = ops.scaled_fp8_quant(
                    decode_q_pe.reshape(
                        [q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
                    layer._q_scale)
                decode_q_pe = decode_q_pe.reshape(q_pe_shape)

            output[:num_decode_tokens] = self._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, attn_metadata, layer)
```

```945:951:vllm/v1/attention/backends/mla/common.py
    def _v_up_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        return x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
```

- In short:
  - Prefill: build per-head `k_nope` and `v` from latent `kv_c_normed` using `kv_b_proj`; concatenate `k = [k_nope, k_pe]`; run FA on `q = [q_nope, q_pe]` vs `k`, `v`.
  - Decode: convert `q_nope → ql_nope = q_nope · W_UK_T`; then attention over `q = [ql_nope, q_pe]` vs cache `[kv_c, k_pe]` with values `kv_c` (MQA-like); finally up-project `o` via `W_UV`.

### Where the Triton kernel is called (decode)
- The Triton backend’s `_forward_decode` is a thin wrapper that concatenates `[ql_nope, q_pe]`, gathers the paged cache, and calls the Triton decode kernel; then applies `_v_up_proj`.

```138:174:vllm/v1/attention/backends/mla/triton_mla.py
        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        num_kv_splits = 4  # TODO: heuristic

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                self.num_heads,
                num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA
        decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o,
                             attn_metadata.decode.block_table,
                             attn_metadata.decode.seq_lens, attn_logits,
                             num_kv_splits, self.scale, PAGE_SIZE)

        return self._v_up_proj(o)
```

- The kernel interface shows the dispatch. When there’s a single KV group (MLA uses kv_group_num = 1), it selects the MHA/MQA path accordingly.

```631:659:vllm/vllm/attention/ops/triton_decode_attention.py
def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
```

### Where q/kv_c/k_pe are produced in the model
- The model builds latent `kv_c` and decoupled `k_pe`, applies rotary to `q[..., rope]`, and hands these to the MLA backend:

```505:534:vllm/v1/model_executor/models/deepseek_v2.py
        if self.q_lora_rank is not None:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_local_heads * self.v_head_dim))
        return self.o_proj(attn_out)[0]
```

### Where KV cache entries are written (layout is [kv_c | k_pe])
- During forward, `concat_and_cache_mla` writes latent K/V and K’s PE into the paged cache:

```1181:1190:vllm/v1/attention/backends/mla/common.py
        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )
```

- The CUDA kernel signature confirms the two parts being concatenated:

```354:369:vllm/csrc/cache_kernels.cu
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
```

### If you’re looking outside v1
- The non-v1 path has the same structure in `vllm/attention/backends/mla/common.py` (weight processing, prefill/decode split, `_v_up_proj`), and backend variants like `flashmla.py`, `cutlass_mla.py`, `rocm_aiter_mla.py` implement `_forward_decode` with different kernels.

Status: I located and cited the precise MLA flow: model projection, cache layout, prefill attention, decode formation of ql_nope, Triton decode call, and up-projection.

- MLA logic is centralized in `vllm/v1/attention/backends/mla/common.py` (weight prep, prefill/decode, cache ops, up-proj).
- Backend `vllm/v1/attention/backends/mla/triton_mla.py` only wires decode to `triton_decode_attention.decode_attention_fwd`.
- Model builds `q`, `kv_c_normed`, `k_pe` in `vllm/v1/model_executor/models/deepseek_v2.py` and passes to the MLA backend.
- KV cache stores `[kv_c | k_pe]` via `concat_and_cache_mla` (kernel in `csrc/cache_kernels.cu`).

```python
        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0)
```

```python
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = self._run_prefill_new_tokens(
            prefill=attn_metadata.prefill,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # unpad if necessary
        if self._pad_v:
            output = output[..., :v.shape[-1]]

        return output.flatten(start_dim=-2)
```

```python
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            if fp8_attention:
                ql_nope_shape = decode_ql_nope.shape
                decode_ql_nope, _ = ops.scaled_fp8_quant(
                    decode_ql_nope.reshape([
                        ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]
                    ]), layer._q_scale)
                decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
                q_pe_shape = decode_q_pe.shape
                decode_q_pe, _ = ops.scaled_fp8_quant(
                    decode_q_pe.reshape(
                        [q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
                    layer._q_scale)
                decode_q_pe = decode_q_pe.reshape(q_pe_shape)

            output[:num_decode_tokens] = self._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, attn_metadata, layer)
```

```python
    def _v_up_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        return x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
```

```python
        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        num_kv_splits = 4  # TODO: heuristic

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                self.num_heads,
                num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA
        decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o,
                             attn_metadata.decode.block_table,
                             attn_metadata.decode.seq_lens, attn_logits,
                             num_kv_splits, self.scale, PAGE_SIZE)

        return self._v_up_proj(o)
```

```python
def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
```

```python
        if self.q_lora_rank is not None:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_local_heads * self.v_head_dim))
        return self.o_proj(attn_out)[0]
```

```python
        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )
```

```plaintext
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
```

