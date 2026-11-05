# A3 INT8 indexer quant/cache fusion

The A3 Kimi K3.1 indexer can write its INT8 K and FP16 scale caches with one
Triton launch. This replaces the split DynamicQuant, scale Cast, two sets of
slot-to-page index arithmetic, and two ScatterNdUpdate operations. The v1 A3
profile contained 16 kernels in this sequence for 48 rows of width 128.
Projection, normalization, Hadamard and query quantization are outside this
fusion.

## Activation and fallback

The existing `process_weights_after_loading` hook selects the implementation
automatically for the A3 native INT8 sparse adaptor when Triton is available.
No environment variable, C++ rebuild or CANN extension is added.

The supported pipeline is the base `AscendSFAIndexerBackend` for Kimi K3.1,
C8 INT8 keys with FP16 scales, head dimension 128, no indexer RoPE, and no LoRA.
At runtime, inputs must be BF16 with 1–128 rows. K and scale caches have shapes
`[pages, page_size, 1, 128]` and `[pages, page_size, 1, 1]`, respectively.
Rows are contiguous inside each plane; page strides may contain padding.
Both view pointers and the K page stride must be 32-byte aligned.
The page size and scale page stride must also be multiples of 16 FP16 elements,
so that each full scale block and its corresponding K rows stay inside a page.

PCP cache gathering retains its existing implementation. DCP
alone does not disable fusion: the existing sharded DCP path with local cache
writes remains eligible. Batches larger than 128, unsupported cache layouts,
or missing Triton use the split path. A5 continues using its existing CANN
FP8 implementation.

## Kernel behavior

- Quantization uses the per-row FP32 `max(abs(x)) / 127` scale, round-to-even,
  and symmetric INT8 bounds. The scale is converted to FP16 only for storage.
  Zero rows write zero K and zero stored scale without dividing by zero.
- INT32 and INT64 flat slots are consumed directly. Negative padding slots
  and out-of-range slots do not write. Valid destination slots must be unique,
  matching the scatter-cache input contract.
- All page offsets use the actual view strides and 64-bit slot arithmetic.
  Pointers retain storage offsets, including when K and scale share one hybrid
  allocation. There is no cache flattening or temporary quantized tensor.
- The first input row targeting a scale DMA block elects its sole writer.
  That program maps all 16 destination lanes to their input rows, then updates
  a full 32-byte scale block and the corresponding contiguous 2048-byte K block.
  Untouched lanes retain their previous values. The stores are unmasked;
  neither destination is expressed as a discrete scatter store.
- Each participating program scans the small input batch. The 128-token limit
  deliberately bounds this overhead; this is a decode optimization rather
  than a large-prefill kernel.

## Validation entry points

Implementation and test cases were prepared locally. **No tests, NPU
compilation, numerical validation or performance measurements were run for
this change, as requested.** The supplied service log shows that the original
version failed during graph capture; the corrected version still requires NPU
verification. One launch is the code path's structure, not a measured latency
or end-to-end speedup claim.

From the repository root in the A3 runtime:

```bash
pytest -q tests/ut/attention/test_kimi_indexer_pipeline_cpu.py --confcutdir=tests/ut/attention
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_indexer_quant_cache_int8.py
python tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_indexer_quant_cache_int8.py --tokens 48 --trials 30
```

The NPU cases compare INT8 values and stored FP16 scales exactly against the
installed `torch_npu.npu_dynamic_quant` implementation. They also compare full
hybrid backing allocations, including untouched rows and padding. Cases cover
both slot dtypes, strided inputs, row-tile boundaries, adjacent destination
scales, invalid slots, small magnitudes, rounding boundaries and graph replay
with updated values and slots. Exact agreement with the installed CANN build
is intentionally checked rather than assumed from the mathematical formula.
The capture regression explicitly uses GLOBAL mode, with repeated launches
after eager warmup, and covers 1–128 tokens and both slot dtypes. A separate
compiler regression requires `sync_block_lock_layout == 0`; warming a kernel
alone does not prevent a per-launch synchronous workspace initialization.

Direct execution runs a paired graph microbenchmark against the production
split writer. It checks numerical agreement first, warms both paths, alternates
measurement order, and prints raw samples plus median microseconds per call.
Graph measurements contain 100 calls per replay. Use the same device and
runtime as the production profile; the result measures this cache-write chain,
not overall decode throughput.

For a service profile, the eligible chain should appear as
`_indexer_quant_cache_int8_kernel` instead of the 16 split kernels. Confirm
this activation before comparing latency, since ineligible layouts retain the
original implementation.

## Graph-capture failure and correction

The supplied October 31 startup log first names
`_indexer_quant_cache_int8_kernel` at line 6080. Lines 6086–6087 report a
synchronous memory copy rejected during GLOBAL capture. The failure later
surfaces at a DCP `all_gather` because device work is asynchronous. Subsequent
`capture_end` error 107025 and RNG capture-state errors follow the first failed
capture; they do not establish independent root causes.

The original kernel used discrete masked K/FP16-scale stores. Source analysis
of Triton-Ascend shows that discrete masked stores can introduce cross-block
locks. On affected launchers, unordered-lock initialization performs a
synchronous host-to-device copy on each launch. This is consistent with the
reported failure, though the deployment's compiled IR and launcher were not
provided and have not been inspected.

The correction removes the discrete destination stores by assigning each
aligned block one writer and preserving holes with full-block read/modify/write.
It retains one kernel launch and does not weaken capture mode, disable compiler
synchronization, patch the installed launcher, or require a dependency upgrade.
Unsupported page alignment uses the existing split fallback. The source-level
proof of nonoverlapping blocks and the compiler/graph regressions must both hold.

Upstream evidence: [Triton-Ascend issue #2815](https://github.com/triton-lang/triton-ascend/issues/2815)
and the affected [launcher implementation](https://github.com/triton-lang/triton-ascend/blob/d4df76a/third_party/ascend/backend/driver.py#L763-L785).
Numerical agreement, absence of generated lock workspace, full service startup,
and the performance impact of the additional block reads remain to be verified
on the deployment's A3/CANN/Triton stack.
