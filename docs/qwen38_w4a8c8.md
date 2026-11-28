# Qwen3.8-Flash-Next W4A8C8

This feature consumes ModelSlim mixed static/dynamic INT8 FA3 checkpoints for
Qwen3.8-Flash-Next on Ascend 910B/A3. Use the vLLM commit verified by this
vLLM Ascend checkout and CANN 9.3.0 with GBSA. The documented configuration uses eight Ascend 910B3 devices.

## Configuration

Use the existing `--quantization ascend --kv-cache-dtype auto` options and
Model Runner V2. The checkpoint must carry per-layer ModelSlim descriptions:

```json
{
  "model.language_model.layers.11.self_attn.quant_type": "QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL",
  "model.language_model.layers.11.self_attn.indexer.quant_type": "QK_INT8_DYNAMIC",
  "mtp.layers.0.self_attn.quant_type": "QK_INT8_DYNAMIC_V_INT8",
  "mtp.layers.0.self_attn.indexer.quant_type": "QK_INT8_DYNAMIC"
}
```

The usual model weight mapper translates these names into the runtime
namespace. These markers are independent of the W4A8/W8A8 linear schemes.
A global `kv_cache_type` or `fa_quant_type` alone does not enable this path.
Unmarked owners retain their existing cache precision, including MTP layers
excluded from FA3 by the calibration recipe. Unsupported per-layer FA3 modes
raise a construction error naming the owner and requested mode.

Q/K use dynamic symmetric INT8. Each marked main or MTP attention layer
selects its own static V granularity. `QK_INT8_DYNAMIC_V_INT8` uses one
FP32 scale per KV head, stored as `[num_kv_heads, 1]`.
`QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL` uses one scale per KV head and channel.
Per-head scales are broadcast across channels once at load time for GBSA.
The checkpoint must include
`<self_attn>.fa_v.scale` and zero INT8 `<self_attn>.fa_v.offset`. The loader
validates scales and handles KV-head sharding/replication under TP. Dynamic-V
`INT8_DYNAMIC` checkpoints require re-quantization with static V; they are
rejected by this native path. MTP FA3 remains excluded when calibration does
not execute MTP. The raw indexer compression ring stays BF16.

## Storage and compute

The QSA cache stores INT8 K/V codes in scheduler-managed pages. FP32 K
scales use a dense `[physical page, KV head, token]` view of the same
allocation. V uses the layer's calibrated scales. Mixed BF16/C8 owners honor
their physical page strides and retain scheduler block IDs. Each indexer
page stores dense INT8 keys and FP16 scales for A2/A3 QLI v2.

Main attention calls native INT8 GBSA (`quant_mode=7`) with INT8 Q/K/V,
FP32 Q/K/V scales, probability quantization multiplier 127, BF16 output
and FP32 softmax. GBSA follows the scheduler page table to read the original
K/V codes and K scales. On A3, the native INT8 indexer shares each request's
page table across its query tokens during uniform multi-token decode. Prefill
and single-token decode use explicit per-row visibility.

GDN state and BF16 draft layers affect total memory savings. INT8 QSA
attention and indexer caches require prefill context parallel size 1.
External KV connectors and offload need separate integration validation.

## Verification

Run CPU interface, numerical and cache-budget tests:

```bash
python -m pytest -q \
  tests/ut/models/qwen4_exp/test_c8_quantization.py \
  tests/ut/models/qwen4_exp/test_c8_attention.py \
  tests/ut/models/test_qwen4_exp_cache_config.py
```

On a reserved device, run synthetic store, indexer, attention and changed-input
graph replay tests without loading model weights:

```bash
python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_qwen38_c8.py
```

CPU mocks and offline kernel compilation do not establish device numerical
correctness. Before production use, validate the complete quantized model's
loading, prefill/decode, chunked prefill, prefix reuse, MTP acceptance,
tensor parallelism, graph replay, accuracy and performance.
