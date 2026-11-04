# Fused GDN decode

The A3 single-token decode path fuses Q/K L2 normalization, softplus decay,
sigmoid beta, recurrent state update, and attention output reduction into one Triton
launch. It avoids materializing normalized Q/K and the two gate tensors.

`BaseDeviceAdaptor.try_fused_gdn_decode` selects this path for BF16 inputs,
128-dimensional key/value heads, FP32/BF16 state with contiguous head/value/key dimensions, and at most 32
decode tokens. Other hardware, larger batches, unsupported state strides, prefill,
mixed batches, and speculative verification retain their existing paths.

State has layout `[pool, value_heads, value_dim, key_dim]`. Distinct live
requests must use distinct state slots. Slot zero is valid. Negative indices
and tokens outside the device-side request boundaries produce zero output and
never write state; this also handles padded graph replays without host reads.
The layerwise KV connector wait/save hooks run for both implementations.

The kernel explicitly rounds normalized Q/K and sigmoid beta to BF16 before
the FP32 recurrence. Integer round-to-nearest-even preserves the rounding
boundaries of the original operators: the NPU compiler can fold a register-only
FP32/BF16/FP32 cast sequence. State tiles use 64 value rows and disable
multibuffering to stay within the vector core's unified buffer.

## Validation

Numerical tests cover FP32/BF16 state, grouped value heads, packed strided
projections, strided gates/indices, batch sizes 1/8/32/128, padding, slot zero,
32 recurrent steps, extreme gates, empty inputs, and graph replay with changed
indices and request boundaries. CPU tests cover dispatch, fallback, connector
hooks, and invalid shapes.

```bash
pytest -q tests/ut/ops/test_fused_gdn_decode.py
pytest -q tests/ut/ops/test_gdn_mixed_batch.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_decode.py
```

The benchmark compares the complete existing FLA path, including normalization
and gating, against the fused kernel. It checks output and final state before
timing. Sixteen sequential calls per captured graph amortize Python replay
overhead; reported latency is per call, using the median of five samples.
Extra allocator bytes refer to the complete captured graph, not a single call.
Batch 128 is included as a diagnostic even though the default dispatch keeps
the FLA implementation above 32 tokens.

Hybrid KV-cache pages may have padding between state slots. The kernel uses the
state slot stride directly and accepts non-overlapping padded pages; it does not
copy the state pool or write the padding. Inner head/value/key strides must
remain contiguous.
