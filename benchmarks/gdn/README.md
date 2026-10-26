# GDN decode fusion benchmark

Run `python benchmarks/gdn/benchmark_fused_decode.py` with the matching A3 FLA
NPU wheel available. The comparison includes the existing gating kernel, both
L2 normalization kernels, and FLA recurrent update. Output and final state are
checked before timing. These are operator measurements, not model TPOT results.

## Reference measurement

Measured on October 26, 2026:

- Ascend910_9382, CANN 9.1.0, torch 2.10.0, torch_npu 2.10.0.post4,
  Triton 3.2.0.
- vLLM Ascend baseline `041cfb328c0628b1fc17db5c55b8d374e6363466`;
  installed upstream vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`.
- FLA A3 wheel `26.10.0+main.deved987cd`.
- BF16 input, FP32 state, 16 key heads, 32 value heads, head dimension 128.
- Sixteen calls per captured graph, 100 replays per sample, median of five
  samples. This amortizes Python graph-replay overhead.

| Decode tokens | Existing path (us) | Fused path (us) | Speedup |
| --- | --- | --- | --- |
| 1 | 33.41 | 10.06 | 3.32x |
| 8 | 72.56 | 34.13 | 2.13x |
| 32 | 202.53 | 118.76 | 1.71x |
| 128 | 734.85 | 534.56 | 1.37x |

The default dispatch uses fusion only up to 32 tokens. Batch 128 is diagnostic:
its margin was smaller in single-call graph measurements, so larger batches
retain FLA pending broader workload validation.

For batch 1, additional peak allocated memory for the captured 16-call graph
was 16,806,912 bytes for the existing path and 18,432 bytes for the fused path.
This includes graph allocations and operator workspace; it is not a per-call
workspace requirement or total process HBM usage.

## Correctness and integration checks

- 21 NPU tests cover outputs, state updates, BF16 gate rounding, padding,
  strided inputs, multiple steps, empty input, and graph replay.
- 53 focused unit tests cover dispatch/fallback, connector hooks, decode QKV
  layout, and state indices. These used isolated upstream vLLM
  `256d8b40d0864aa694e3dc236d9b351886a4a94a` because the installed upstream
  lacks APIs required by the current Ascend source.
- The broader mixed-batch suite hits an existing weight-reload incompatibility:
  `replace_parameter()` does not accept `prefer_copy`. Model end-to-end
  validation remains pending in a matching upstream environment.

Ruff, Codespell, Typos, Gitleaks, ShellCheck, and the repository's Python
structure/import checks passed. Markdownlint 0.45.0 passed separately. The
available full-repository hooks passed with
`SKIP=clang-format,markdownlint,actionlint bash format.sh ci`; the unmodified
C++ and workflow hooks remain unrun because their tools are unavailable in
the container.
