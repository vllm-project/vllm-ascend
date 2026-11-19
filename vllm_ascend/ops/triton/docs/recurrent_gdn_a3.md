# A3 recurrent GDN fusion — PR #21655

Upstream head: b03394cdc92571441ad79edede46b2e3bc22ff0a.
Applied on top of PR #22010, the minimal #21820 prepare fusion, and #21678.

## A3 adaptation

Enable FLA_GDN_RECURRENT_FUSION for A3 and retain A5. A2/310P remain disabled.
The installed A3 FLA 26.10.0+main.dev2ef0735 wheel supports the extended recurrent API.
No additional wheel or operator build was needed for this change.

For eligible BF16 recurrent calls with K/V head dimensions 128, gate computation
(-exp(A_log) * softplus(a + dt_bias)), beta sigmoid, and Q/K L2 normalization
run inside recurrent GDN. External gating and two L2Norm launches are removed.
The cached float32 bias is refreshed in place after successful weight reload or
weight-transfer completion, preserving graph addresses.

Keep API probing and existing fallback: unsupported dtype/dimensions/API,
explicit upstream recurrence, or speculative state-table width greater than 8.
Prefill continues to use the separate #21678 path.

## Validation

- CPU: 375 passed, 15 dependency warnings.
- A3 logical die 6 (physical NPU3 reserved with die7): 8 shapes, 24 eager
  comparisons and 16 NPUGraph replays against external preprocessing.
- Q/K heads 16/32 (TP1 model shape) and 8/16; sequence widths 1, 4, 8;
  one/two sequences, varying accepted-token positions, repeated state updates.
- Maximum relative L2: output 0.00373518, float32 state 0.00215254;
  threshold 0.01. No NaNs; unselected state rows unchanged.
- Graph replay also changes raw gates and refreshes bias in the original buffer.
- This is operator parity coverage, not a model benchmark or quality evaluation.

## Integration coverage

The original hardware-profile expectations are updated for A3/A5 recurrent
fusion. Worker weight-refresh tests are in
`tests/ut/ops/test_gdn_weight_reload.py`.

A3 service smoke tests passed with Qwen3.6-35B-A3B W4A8, TP1/DP1, 1P1D,
MTP3, prefill eager and decode FULL_DECODE_ONLY, prefix caching disabled,
and automatic KV-cache dtype. These checks do not establish model accuracy
or end-to-end throughput improvement.
