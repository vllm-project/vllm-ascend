# Qwen PLE PyTorch backend

The Ascend Qwen PLE short convolution uses PyTorch NPU operators directly,
without an additional configuration option.

The implementation supports prefill, decode and speculative decoding, including
token remapping, initial history and state writeback. All modes use the same
bounded token chunks and vectorized state writeback. The existing model
adaptation replaces the function bound by the upstream PLE layer; this binding
should be removed when upstream exposes a platform dispatch hook for the
convolution.

## Validation

Run on a compatible Qwen-enabled vLLM checkout with torch-npu installed:

```bash
pytest -q tests/e2e/singlecard/test_torch_ple.py
```

The tests compare output and the entire state bank with upstream Triton, including
BF16/FP16, residual addition, null slots, mixed initial-state flags, token mapping,
speculative acceptance counts, and 8192-token prefill. The output threshold is
99.9% elementwise equality; state equality is exact.

The generic implementation passed 129 CPU comparison cases against the prior
implementation, with exact output and state across two consecutive calls per case.
NPU accuracy and performance must be remeasured for this revision before using
the previous single-request prefill benchmark as a performance claim.
