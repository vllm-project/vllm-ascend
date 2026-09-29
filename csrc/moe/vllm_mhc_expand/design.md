# mHC forward expansion

## Contract

`npu_mhc_expand(x, mult)` copies contiguous FP16/BF16 `[tokens, hidden]`
into a new contiguous `[tokens, mult, hidden]` tensor on Ascend A2.
Each output stream is a bitwise copy of the input. `mult` must be positive.
Empty tensors return without launching a kernel. This is an inference-only op.
The Python helper preserves the native path for gradients, other devices,
unsupported dtypes, noncontiguous input, widths not divisible by 16, and
empty expansion and stream counts other than the measured GLM multiplier 4.
Nonaligned widths remain supported by the raw operator
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

`DataCopyPad` supports UB-to-GM stores on A2. For `DataCopyExtParams`, GM strides
are in bytes and UB strides are in 32-byte blocks. The output stride is therefore
`(mult - 1) * hidden * sizeof(uint16_t)` bytes. The source stride is zero.
Tail stores discard padding instead of writing past the logical output.
See the [official nonaligned-copy guide](https://www.hiascend.com/developer/techArticles/20250627-1).

## Integration and evaluation

Register the ACLNN op, Torch PrivateUse1 and symbolic Meta implementations.
Initially build and select this implementation on A2 only. Route GLM mHC
expansion through the helper; preserve its mean-based contraction.
The GLM call is at the first layer of each mHC forward, rather than at every
decoder layer. Device-only microbenchmarks do not establish model throughput.

Correctness covers both dtypes, empty tensors, multipliers 1/2/4/8,
unaligned and tile-boundary widths, changed inputs, special bit patterns,
and graph replay. Baselines are `expand().contiguous()` and `repeat()`.
NPU tests exercise both the helper and the actual GLM `hc_expand` entry point,
including native fallback and gradients, with the normal extension loader.
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

## Repeated production-entry measurements

The additional GLM cases use hidden size 4096, multiplier 4 and token counts
1/4/16/17/128/1024/3500/8192 for both supported dtypes. The cases are independently
designed from the GLM mHC contract. Run each timing mode into a fresh directory:

```bash
python benchmarks/mhc_expand.py --path glm --rounds 3 --cases benchmarks/mhc_expand_glm_cases.jsonl --timing profiler --output /tmp/mhc-glm-profiler
python benchmarks/mhc_expand.py --path glm --rounds 3 --cases benchmarks/mhc_expand_glm_cases.jsonl --timing wall --output /tmp/mhc-glm-wall
python benchmarks/mhc_expand.py --path glm --rounds 3 --cases benchmarks/mhc_expand_glm_cases.jsonl --timing graph --output /tmp/mhc-glm-graph
```

The implementation order alternates by round. Report all rounds, their medians
and the minimum/maximum speedup. Wall measurements include Python enqueue and
final device completion, averaged over 100 invocations; they are not isolated
per-request latency. Graph capture and correctness checks are outside timing.
Profiler measurements keep five warmup and five active calls per trace.
