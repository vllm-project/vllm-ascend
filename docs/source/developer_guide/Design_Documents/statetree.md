# StateTree weight representation

StateTree is an optional lossless representation for eligible BF16 linear
weights on Ascend 950. See the [feature guide](../../user_guide/feature_guide/statetree.md)
for configuration, supported layers, and fallback behavior.

## Source layout

| Location | Responsibility |
| --- | --- |
| `vllm_ascend/ops/statetree/__init__.py` | Read the typed Ascend feature switch. |
| `vllm_ascend/ops/statetree/config.py` | Define tile limits and select weight and input-row tiles. |
| `vllm_ascend/ops/statetree/codec.py` | Pack BF16 weights on the host and provide independent CPU reconstruction. |
| `vllm_ascend/ops/statetree/compiler.py` | Resolve native sources and compile immutable kernel specializations. |
| `vllm_ascend/ops/statetree/native.py` | Allocate outputs and launch kernels on the current NPU stream. |
| `vllm_ascend/ops/statetree/operator.py` | Register the graph-visible custom operator and select the execution path. |
| `vllm_ascend/ops/statetree/methods.py` | Install packed weights through vLLM linear and embedding methods. |
| `csrc/statetree/statetree.asc` | Implement Vector reconstruction, Cube GEMM, and direct-layout decoding. |

The implementation uses the existing linear and embedding extension points.
It does not require model-runner patches or a model-specific dispatch table.
The configuration field belongs to `AscendConfig`; new environment switches
should not be added in the operator modules.

## Weight lifecycle

At model load time, the method checks the layer's shape, dtype, layout contract,
and weight sharing. Eligible weights are packed into one contiguous byte tensor.
A device reconstruction must match the original BF16 bits before the dense
parameter can be released. Incompressible weights retain the normal BF16 method.

Packed metadata is immutable. The payload is registered as a layer buffer, so
its lifetime follows the layer and a captured graph can retain its address.
Compilation and verification occur before graph capture. The execution path
uses device pointers and the current stream; it does not copy weights to the
host or read device scalars for dispatch.

## Dispatch and hardware buffers

`M` is the number of input elements divided by the weight's `K` dimension.
All leading input dimensions contribute, including mixed prefill/decode rows.

For `M <= 64`, the kernel uses the existing small-row tiles and 20 Cube groups.
For `65 <= M <= 128`, it uses a 128-row tile; for `129 <= M < 256`, a 256-row tile.
Both prefill tiles use all 28 Cube cores. Each Cube has two associated Vector
cores. Reconstructed weights remain in UB/L1 and are reused across token rows.
The largest FP32 accumulator occupies the 256-KiB L0C buffer.

For `M >= 256`, the operator retains native BF16 GEMM because its reduction
schedule can differ from the fused kernel. The decoder uses separate buffers
to overlap compressed reads, reconstruction, and stores. DMA loops scatter
fragments directly into the final row-major weight. Row bounds exclude padded
N tiles, and DMA strides are start-to-start byte offsets. Buffer reuse waits
for the relevant producer and consumer events; DMA loop state is reset after
each store.

Compressed weight reads use `NOTALLOC_KEEP`. Direct-layout stores use
`NOTALLOC_CLEAN`. The large-row path needs one temporary full BF16 weight per
call, but no persistent dense copy or cross-request mutable workspace.

## Compilation and packaging

The canonical Ascend C source lives in `csrc/statetree`. `setup.py` maps that
directory to the `vllm_ascend.ops.statetree.kernels` package and includes its
`.asc` resource in source and wheel distributions. Editable checkouts resolve
the canonical source directly; installed wheels resolve their bundled resource.
There is no duplicate checked-in kernel under the Python operator directory.

This is a shape-specialized JIT kernel, not an ACLNN operator. It is compiled
with CANN `bisheng` for `dav-3510` during weight preparation rather than added
to the ACLNN operator registry. The compiler cache resides under
`VLLM_CACHE_ROOT/ascend/statetree`. A cache entry identifies the source content,
M/N/K tiles, and exceptional-pair capacity. A per-entry file lock and atomic
rename prevent readers from loading a partially written library. Failed builds
remove the temporary output. Toolchain upgrades require a clean compiler cache.

All serving tile variants are warmed before graph capture. The fused launch
ABI takes stream, payload, input, output, and M/N/K. The direct decoder ABI takes
stream, payload, output, record count, N, and K. Changes to either ABI must update
the ctypes signatures and hardware tests together.

## Regression coverage

CPU tests follow the module layout under `tests/ut/ops/statetree/` and use
`unittest`. They cover the packed representation, invalid inputs, tile
boundaries, graph metadata, dispatch, source discovery, and compiler failures.
CLI configuration checks remain in `tests/ut/test_ascend_config.py`.

```bash
OMP_NUM_THREADS=8 python -m pytest -q tests/ut/ops/statetree/
python -m pytest -q tests/ut/test_ascend_config.py -k statetree
```

Device execution belongs to the nightly single-card operator suite. The tests
require Ascend 950 and skip on other devices. They cover BF16 reconstruction,
fused and native-GEMM boundaries, layer integration, padding, and graph replay
after input or payload updates.

```bash
OMP_NUM_THREADS=8 python -m pytest -q \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_statetree.py
```

The A5 nightly configuration already includes this operator directory. Keep
benchmark output and machine-specific measurements in review artifacts rather
than source files or correctness tests. Report hardware validation in the PR
according to the [testing requirements](../contribution/testing.md#pr-testing-requirements).
