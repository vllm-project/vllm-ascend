# SFA MLA Prolog V3 With KV-Quant Sparse Attention

This document describes the SFA path that connects `npu_mla_prolog_v3`
with `npu_kv_quant_sparse_flash_attention`.

## Goal

The feature is enabled by:

```bash
VLLM_ASCEND_ENABLE_SFA_PROLOG_V3=1
VLLM_ASCEND_ENABLE_SFA_KV_QUANT_SPARSE_ATTENTION=1
```

It is independent from `enable_fa_quant`. The KV cache quantization is
produced by `npu_mla_prolog_v3` with per-tile packed cache mode, then consumed
directly by `npu_kv_quant_sparse_flash_attention`.

## Data Flow

```text
hidden_states
    |
    v
npu_mla_prolog_v3
    |                 \
    |                  \-- q_norm -> DSA indexer -> sparse_indices
    |
    +-- q_nope + q_rope
    |
    +-- packed int8 KV cache
            |
            v
npu_kv_quant_sparse_flash_attention
    |
    v
latent attention output -> W_UV -> o_proj
```

## Cache Layout

The packed KV cache is stored in `kv_cache[0]`. `kv_cache[1]` is an empty
`kr_cache` placeholder because rope and per-tile scales are combined into the
packed cache. DSA indexer cache still uses the later tuple entries.

```text
kv_cache tuple

normal SFA:
  [0] kv_lora        bf16/fp16
  [1] k_rope         bf16/fp16
  [2] dsa_index_key  bf16/fp16

SFA KV-quant sparse attention:
  [0] packed_kv      int8
      = kv_lora_int8
        + k_rope_bf16.view(int8)
        + tile_scales_fp32.view(int8)
  [1] empty kr_cache bf16, last_dim = 0
  [2] dsa_index_key  bf16/fp16 or int8 when Sparse C8 indexer is enabled
  [3] dsa_index_scale only when Sparse C8 indexer is enabled
```

For the common MLA shape, the packed KV last dimension is:

```text
kv_lora_rank + qk_rope_head_dim * 2 + (kv_lora_rank / 128) * 4

512 + 64 * 2 + 4 * 4 = 656
```

## Operator Parameters

`npu_mla_prolog_v3` uses:

```text
kv_cache_quant_mode=3
ckvkr_repo_mode=1
quant_scale_repo_mode=1
tile_size=128
query_quant_mode=0
weight_quant_mode=2
```

`npu_kv_quant_sparse_flash_attention` uses:

```text
query = cat(q_nope, q_rope)
key = packed_kv
value = packed_kv
layout_query="TND"
layout_kv="PA_BSND"
key_quant_mode=2
value_quant_mode=2
attention_mode=2
quant_scale_repo_mode=1
tile_size=128
rope_head_dim=qk_rope_head_dim
```

The attention output last dimension is `kv_lora_rank`, so the existing MLA
`W_UV` up-projection remains unchanged.

## Context Parallel

For DSA-CP or PCP, packed KV and DSA indexer cache are communicated separately.
This avoids concatenating tensors with different dtypes and preserves the
packed int8 cache format before writing back to paged cache.

```text
local prolog packed_kv(int8)  --gather--> scatter to kv_cache[0]
local dsa_index_key          --gather--> scatter to kv_cache[2]
```

## Current Scope

The int8 path is enabled for non-A5 devices. A5 keeps the existing Sparse C8
FP8 CKV behavior. If the SFA prolog v3 weight and layernorm requirements are
not met while this feature is enabled, initialization fails fast instead of
falling back to an incompatible unpacked cache layout.
