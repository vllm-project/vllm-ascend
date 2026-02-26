# FP8-on-NPU Lessons

## 1) Recommended debug order

1. Start with `--load-format dummy` to quickly verify architecture path.
2. Run with real weights to validate weight mapping and load-time stability.
3. If blocked by fp8 execution limits on NPU, use fp8->bf16 dequantization loading path.
4. Validate `/v1/models`, then one text request, then one VL request (if multimodal).

## 2) FP8 checkpoint on NPU

Common symptom:

- `fp8 quantization is currently not supported in npu`.

Recommended pattern:

- do not force fp8 execution kernels on NPU;
- dequantize fp8 weights to bf16 during loading using paired tensors:
    - `*.weight`
    - `*.weight_scale_inv`
- keep strict unpaired scale/weight checks to avoid silent corruption.

## 3) Typical real-only risks (dummy may not expose)

- missing fp8 scale keys during real shard loading;
- wrong weight remap path only triggered by real checkpoints;
- KV/QK norm sharding mismatch under TP + replicated KV heads.

## 4) KV replication + TP pitfalls

Typical symptom:

- shape mismatch like `128 vs 64` when `tp_size > num_key_value_heads`.

Recommended pattern:

- detect KV-head replication explicitly;
- use local norm/shard loader path for replicated KV heads;
- avoid assuming uniform divisibility for all head dimensions.

## 5) ACLGraph stability for fp8-origin checkpoints

Recommended pattern:

- prefer `HCCL_OP_EXPANSION_MODE=AIV` when using graph mode;
- keep practical capture sizes and re-test from small, stable shapes;
- use `--enforce-eager` only as temporary isolation fallback.

## 6) Reporting discipline

Always report both:

- what dummy validated (fast gate), and
- what only real weights validated (mandatory gate).

Do not sign off fp8-on-NPU adaptation with dummy-only evidence.
