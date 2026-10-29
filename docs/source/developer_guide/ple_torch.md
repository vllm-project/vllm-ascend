# Qwen PLE PyTorch backend

The Ascend Qwen PLE short convolution uses PyTorch NPU operators directly,
without an additional configuration option.

The implementation supports prefill, decode and speculative decoding, including
token remapping, initial history and state writeback. Single-request prefill uses
contiguous history slices and FP32 accumulation with `addcmul`. Other cases use
bounded token chunks. The existing model adaptation replaces the function bound
by the upstream PLE layer; this binding should be removed when upstream exposes
a platform dispatch hook for the convolution.

## Validation

Run on a compatible Qwen-enabled vLLM checkout with torch-npu installed:

```bash
pytest -q tests/e2e/singlecard/test_torch_ple.py
```

The tests compare output and the entire state bank with upstream Triton, including
BF16/FP16, residual addition, null slots, mixed initial-state flags, token mapping,
speculative acceptance counts, and 8192-token prefill. The output threshold is
99.9% elementwise equality; state equality is exact.

On Ascend 950DT, a controlled TP4 prefill + TP4 decode deployment through the same
proxy measured the following with profiling disabled, prefix caching disabled,
8192 batched tokens, MTP3, concurrency 1 and four identical 16436-input/1-output
requests after a separate warmup:

| Backend | Mean TTFT | Median TTFT | Range |
| --- | --- | --- | --- |
| Triton | 2730.0 ms | 2730.3 ms | 2716.5–2742.7 ms |
| PyTorch | 2317.1 ms | 2314.6 ms | 2313.8–2325.2 ms |

The mean reduction was 15.1% in this small controlled experiment. Separately
captured profiles showed approximately 199–201 ms for the upstream convolution
and 5.72–5.86 ms for the PyTorch call including state writeback at 8192 x 10240.
The latter launched 51 NPU operators with the serving residual and state layout.
These measurements do not establish performance on other hardware or workloads.
