# MiniMax-M3 A8C4 dual-format cache (experimental)

## Function and interface

This model-specific path targets MiniMax-M3/M3.1 sparse attention on Ascend 950.
Enable it with `--kv-cache-dtype fp8` and
`--additional-config '{"sparse_attention_quant_config":{"enable_a8c4":true}}'`.
The typed additional configuration defaults to `false`: ordinary FP8 and BF16
cache behavior is preserved. An FP8 dtype alone cannot select a different page
layout, so this experimental layout needs a separate explicit opt-in.

`a8c4_cache_views` aliases matching FP8 K/V planes of shape
`[blocks, 128, kv_heads, 256]`. Logical head size remains 128. Each plane stores
an FP8 region, packed FP4 E2M1 region and BF16 per-token/group-16 scales, followed
by padding. Per token/head, K+V payload is 416 bytes and reservation is 512 bytes.
This intentionally retains FP8 for prefill; it does **not** reduce the cache
reservation relative to the native FP8 layout.

`dual_cache_scatter` accepts BF16 K/V `[tokens, kv_heads, 128]` and int64 slots.
Negative slots are padding. Nonnegative slots must fit the allocated block range,
and active slots must be unique. It mutates only the selected FP8/FP4/scale rows.
Input values must be finite. Group scale is `amax / 6`, rounded to BF16, with
scale 1 for all-zero groups. Quantization uses nearest-even E2M1 rounding; two
adjacent logical values occupy the low and high nibble of each uint8 byte.

The normal fused `minimax_qknorm_rope_cache` producer calls the same inlined FP4
store, and emits FP8 Q for pure decode. Thus the fused path adds no separate
packing/scaling launch. The unfused model path uses `dual_cache_scatter`.
Mixed prefill/decode batches retain BF16 Q and convert the decode slice to FP8.

## GBSA and compatibility

Decode passes FP8 Q, packed uint8 K/V, separate BF16 scales, quantization mode 8
and BF16 output to CANN GBSA. CANN's PyTorch bridge must explicitly wrap packed
uint8 storage as `ACL_FLOAT4_E2M1`, doubling the final logical dimension and
converting element strides/offsets to nibble units. Ordinary uint8 tensor dispatch
is insufficient. Requires the A8C4 GBSA/metadata operator from
[CANN ops-transformer PR 16153](https://gitcode.com/cann/ops-transformer/pull/16153)
at commit `25e53d6c502f7a8c20e258ad4448e35a33a15d4e`, including its mode-8
PyTorch bridge. No private filesystem paths are used.

Prefill reads the FP8 region. The dense prefill fallback materializes contiguous
FP8 pages, so prefill memory/time overhead remains to be measured. Union prefill
is disabled with this option. PP other than 1, CP, multi-node execution and KV
transfer connectors are rejected. The dual-format layout is a temporary
single-node integration, pending a compact native FP4 cache and PD transfer
contract; it is not a validated PD format.

## Validation

CPU tests check allocation accounting, alias boundaries, nonzero storage offsets,
multiple KV heads, malformed layouts, default behavior and configuration guards:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
  tests/ut/ops/triton/msa/test_msa_m3_a8c4_cache.py \
  tests/ut/test_sparse_attention_quant_config.py
```

NPU producer tests use an independent CPU nearest-even reference, including zero
groups, invalid slots, block boundaries, untouched padding and pure-decode token
counts 1/8/224. Packed integers and scales from standalone scatter require exact
equality. Fused normalization permits 2% packing-boundary mismatches and 2% scale
relative error, with FP8 Q absolute tolerance 0.25. These producer checks do not
establish model-level FP4 accuracy.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_a8c4_cache.py
```

Prior r114 single-node validation used TP4/DP2, DSpark K7, input 131072/output
1920, C56/N109. All 109 requests completed; a 4.0009386-second stable window
contained 4442 mode-8 GBSA and 4442 fused producer calls on every rank, with zero
standalone FP4 packing/scattering or Q-shaped Cast/Clip calls. This is dispatch
evidence, not a baseline speedup measurement. The rebased PR and opt-in guards
still require fresh NPU/graph replay, model accuracy, prefill cost and performance
regression validation before merge. BF16 versus FP4 model accuracy and PD are
unverified.

## JIT specialization

Head counts, head dimension, block strides, cache dtype and the A8C4/query-FP8
flags specialize the existing fused producer. Tensor data and slot contents are
runtime inputs; changing token count/stride/alignment may select another variant.
The fallback scatter specializes head counts and strides. No new warmup registry
or mutable module state is introduced. Startup-versus-serving compilation counts
on a model evaluation workload remain to be collected; a kernel launch count is
not a compilation count.
