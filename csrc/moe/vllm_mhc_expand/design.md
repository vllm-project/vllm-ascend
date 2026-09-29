# mHC forward expansion

## Contract

`npu_mhc_expand(x, mult)` copies contiguous FP16/BF16 `[tokens, hidden]`
into a new contiguous `[tokens, mult, hidden]` tensor on Ascend A2.
Each output stream is a bitwise copy of the input. `mult` must be positive.
Empty tensors return without launching a kernel. This is an inference-only op.
The Python helper preserves the native path for gradients, other devices,
unsupported dtypes, noncontiguous input, widths not divisible by 16, and
trivial/empty expansion. Nonaligned widths remain supported by the raw operator
for correctness, but the helper avoids its measured single-core slowdown.

## Algorithm

Adapted from the local mHC Expand competition implementation: copy each input
tile into UB once, then emit all streams using DMA. Small aligned rows use
up to 16 rows per batch; full aligned rows use two-row strided output DMA.
When there are no more tokens than launched cores, each core handles one row
to avoid leaving half the cores idle.
Larger rows are split into 32-element-aligned tiles of at most 8192 elements.
GM offsets use 64-bit arithmetic. Unaligned rows use one core to avoid
neighboring cores sharing a partial 32-byte output block. Explicit MTE events
protect load-to-store dependencies and UB reuse. No numerical conversion occurs.
The conservative original tile budget is retained for this initial port.

## Integration and evaluation

Register the ACLNN op, Torch PrivateUse1 and symbolic Meta implementations.
Initially build and select this implementation on A2 only. Route GLM mHC
expansion through the helper; preserve its mean-based contraction.

Correctness covers both dtypes, empty tensors, multipliers 1/2/4/8,
unaligned and tile-boundary widths, changed inputs, special bit patterns,
and graph replay. Baselines are `expand().contiguous()` and `repeat()`.
Performance cases include decode token counts 1/4/16 and prefill 128/1024,
widths 4096/7168, and small/tail widths. Use separate torch_npu profiler runs
per case and implementation with five warmup and five active steps.
Sum all device operator times per active invocation. Report raw profiler paths,
hardware/software versions and speedups. These are independently designed cases
(用例为自行设计，非 testcase-gen 产出). Performance claims require hardware measurements.

## Running the checks

In an A2 environment with the repository's supported CANN, PyTorch and vLLM
versions, build/install this branch using the normal project installation flow.
Then run:

```bash
pytest -q tests/ut/ops/test_mhc_expand.py tests/ut/device/test_hardware_profile.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_mhc_expand.py
python benchmarks/mhc_expand.py --output /tmp/mhc-expand-profile
```

Use a new output directory for each profile. Record the CANN toolkit and driver
versions alongside `report.md`. Inspect per-case results before enabling the
custom path for additional hardware or model call sites.
