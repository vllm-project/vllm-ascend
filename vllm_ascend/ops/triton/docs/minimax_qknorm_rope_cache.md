# MiniMax-M3 QK norm, RoPE and KV cache insertion

## Description

- **Function**: `minimax_qknorm_rope_cache` applies four Gemma RMSNorms, partial NeoX RoPE, and three cache writes in one Triton kernel. Only Q and index Q are returned. K, V, and index K are written directly into their paged caches.
- **Location**: `vllm_ascend/ops/triton/msa/minimax_qknorm_rope_cache.py` — `_minimax_qknorm_rope_cache_kernel`, wrapper `minimax_qknorm_rope_cache`.
- **Algorithm flow**:
  1. Each program handles one token and vectorizes Q/K and index Q/K along contiguous head rows.
  2. Cos/sin is loaded from the RoPE cache by `positions`. No external gather, slice-contiguous, or cache-scatter kernel is launched.
  3. The model calls this through the opaque op `minimax_m3_fused_sparse_forward`, then runs the indexer and sparse attention. The original cache-insertion path is not called again.
- **Supported modes**: BF16 NPU input, main and index head dimension 128, at most 512 tokens including graph padding. Packed `[Q | K | V | index Q | index K]` projections and separate QKV/index projections are both supported. Other shapes keep the existing prepare path.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type |
| --- | --- | --- | --- |
| `packed` | Input | Main QKV projection, or packed QKV plus index when `index_packed` is omitted | bfloat16 |
| `index_packed` | Input | Separate index Q/K projection. Used by W8A8 QKV with BF16 index weights | bfloat16 |
| `cos_sin_cache` | Input | NeoX RoPE table, indexed by `positions` | floating |
| `positions` | Input | Token positions, rank 1 | integer |
| `q_weight`, `k_weight`, `index_q_weight`, `index_k_weight` | Input | Gemma RMSNorm weights. All four epsilons must match | floating |
| `key_cache`, `value_cache` | Output | Main cache `[num_blocks, block_size, heads, head_dim]` | bfloat16 or float8_e4m3fn |
| `index_cache` | Output | Index cache `[num_blocks, block_size, head_dim]` | bfloat16 or float8_e4m3fn |
| `slot_mapping`, `index_slot_mapping` | Input | Independent main and index slots. Negative slots are not written | integer |
| query, index query | Output | Materialized queries only | bfloat16 / index cache dtype |

## Constraints

- `num_blocks` is the runtime cache capacity. It is not hardcoded and is not a kernel compile parameter. Main and index block sizes may differ.
- Cache writes use the actual tensor strides, including block padding.
- Slots outside `num_actual_tokens`, and negative slots, do not update a cache. The caller must keep valid slots in range and must not write the same destination twice in one call.
- Normalization accumulates in FP32 and rounds back to BF16 before RoPE. Main K is rounded to the model dtype before the cache store. FP8 E4M3 caches are clamped to ±448 before the store.
- A forward without attention metadata only zeroes the attention output and does not touch the cache.
- No new environment variable. MoE, FlashComm, and speculative-decoding policy are unchanged.

## Origin and Differences

- **Origin**: Ascend PR #15772 and vLLM `fusedMiniMaxQNormRopeKVInsertKernel`. This implementation is independent of #15772 and targets current Ascend main.
- **Differences**: The operator lives in `ops/triton/msa` with the other MiniMax-M3 Triton kernels. Separate index projections are accepted without a concat or copy. Projection GEMMs stay on the original model path.

## Test Cases

```bash
pytest -q tests/ut/models/minimax_m3
pytest -q tests/e2e/pull_request/one_card/test_minimax_qknorm_rope_cache.py
```

Unit tests cover the fusion guard, fallback, dummy and fake forwards, separate index projections, and cache-insertion order. NPU tests cover 1 to 513 tokens, several head layouts, partial and full RoPE, shuffled and negative slots, untouched cache regions, non-contiguous inputs, empty inputs, and graph replay after `positions` or slots change. Dynamic-cache cases use 3, 64, 5392, and 8192 blocks. Two FP8 cases are skipped when A5 is not available.
