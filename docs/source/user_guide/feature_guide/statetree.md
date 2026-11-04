# Lossless StateTree Weight Compression

StateTree is an experimental, opt-in weight representation for BF16 dense
linear layers on Ascend 950. It stores eligible weights in a lossless packed
format and fuses packed-weight reads, BF16 reconstruction, and Cube matrix
multiplication for decode and moderate prefill batches. The fused path never writes a dense
weight to global memory.

The implementation is adapted from
[StateTree-Representation](https://github.com/ammarali32/StateTree-Representation).
It is disabled by default and does not change models that do not explicitly
enable it.

## Enable the feature

For online serving, use the typed Ascend additional-config switch:

```bash
vllm serve /path/to/model \
    --dtype bfloat16 \
    --additional-config.enable_statetree true
```

The equivalent JSON form is:

```bash
vllm serve /path/to/model \
    --dtype bfloat16 \
    --additional-config '{"enable_statetree": true}'
```

For offline inference:

```python
from vllm import LLM

llm = LLM(
    model="/path/to/model",
    dtype="bfloat16",
    additional_config={"enable_statetree": True},
)
```

## Layer selection and fallback

Selection is based on the rank-local weight, not a model-name whitelist. An
eligible layer currently has all of the following properties:

- a two-dimensional, bias-free BF16 weight on NPU;
- `N` divisible by 128 and `K` divisible by 256;
- at least 2,097,152 weight elements;
- the normal vLLM quantization-method forward path (no layer-specific custom
  operator that reads the dense weight directly);
- no specialized Conv1D, `wo_a`, FP32-precast, or keep-dense layout contract;
- for an LM head, a weight that is not tied to the token embedding.

Layers outside this validated subset automatically retain the existing Ascend
BF16 implementation. An incompressible packed result also falls back to BF16.
Consequently, Qwen and other model families can use the feature when their
rank-local shapes match; the feature is not tied to one exact model or one
global checkpoint shape.

The operator selects a path from the actual number of input token rows `M`,
including rows from mixed prefill/decode scheduling:

| Input rows | Path |
| --- | --- |
| `M <= 64` | Existing decode fusion with 16/32/64-row tiles and 20 Cube/Vector groups. |
| `65 <= M <= 128` | Fused prefill with a 128-row tile and all 28 Cube cores. |
| `129 <= M < 256` | Fused prefill with a 256-row tile and all 28 Cube cores. |
| `M >= 256` | Pipelined direct-to-row-major weight reconstruction followed by native BF16 GEMM. |

Fused prefill reuses reconstructed weights in UB/L1 across more tokens. The
largest 256-by-256 FP32 accumulator fits the 950's 256-KiB L0C buffer. The
existing decode weight-read cache policy is preserved.

From 256 rows onward, native GEMM can choose a different reduction schedule;
retaining native GEMM avoids changing its output rounding. The large-prefill
path uses 3510 DMA loops to write the final row-major layout directly, replacing
the former full-sized NZ intermediate and layout-copy passes. Separate buffers
overlap compressed reads, Vector reconstruction, and writes. This path still
uses one temporary BF16 weight; no persistent BF16 copy or cross-request mutable
workspace is retained. It is not a port of CUDA's concurrent `tgx` ring-buffer
implementation.

## Correctness and startup behavior

During model loading, every selected weight is encoded on the host and decoded
once on the device. vLLM compares the reconstructed BF16 bit patterns with the
original weight and aborts startup on any mismatch. It then compiles all
decode and fused-prefill kernel variants before graph capture and releases the original
dense parameter.

Compiled Ascend C specializations are cached under
`$VLLM_CACHE_ROOT/ascend/statetree`. The cache key includes the kernel source,
tile sizes, and exceptional-pair capacity. Remove that directory after a CANN
toolchain upgrade if a clean rebuild is required.

## Current limitations

- Ascend 950 only; the kernel target is `dav-3510`. Current hardware validation
  is on 950PR.
- BF16 unquantized models only.
- Sleep mode, RL mode, and runtime weight transfer are rejected at startup.
- Tied LM heads stay in BF16 because embedding lookup consumes the shared dense
  parameter.
- Bias-bearing and otherwise unsupported layers fall back to BF16.

The representation is lossless, but end-to-end speedup depends on the fraction
of model time spent in eligible GEMMs and the distribution of token-row counts. Scheduler, attention, MTP,
collectives, sampling, and fallback layers are unchanged.

For module responsibilities, kernel packaging, and regression commands, see the
[StateTree implementation design](../../developer_guide/Design_Documents/statetree.md).
