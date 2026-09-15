# FA3 inference with device-side tiling

The opt-in FA3 inference backend uses `flash_attn_varlen_func` and
`flash_attn_with_kvcache` from
[flash-attention-npu](https://github.com/MinghuasLab/flash-attention-npu).
It is separate from the [RL training-consistency backend](flash_attention.md).

## Installation and selection

Build the v3 operator for Ascend 910B/C with the accompanying
[varlen metadata interface patch](fa3_varlen_scheduler_metadata.patch).
Replace `/path/to/vllm-ascend` below with this checkout's absolute path:

```bash
git clone --recursive https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git checkout d2edb6d7c3587fb7f58b4c91e1f3aedd6e602ca0
git apply /path/to/vllm-ascend/docs/source/user_guide/feature_guide/fa3_varlen_scheduler_metadata.patch
source /usr/local/Ascend/cann/set_env.sh
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python setup.py install
```

The Python module must provide `flash_attn_npu_3.get_scheduler_metadata`,
`flash_attn_varlen_func`, and `flash_attn_with_kvcache`. The integration was
developed against operator revision `d2edb6d7c3587fb7f58b4c91e1f3aedd6e602ca0`.
The patch adds the optional `scheduler_metadata` argument to the varlen Python
interface; it leaves the C++ ABI and kernels unchanged. The unpatched revision
does not accept this argument, so update the operator package together with this
backend. Calls without the argument retain the operator's internal tiling path.

Enable the backend through the Ascend platform's attention selector:

```bash
vllm serve Qwen/Qwen3-8B \
    --additional-config '{"enable_fa3": true}' \
    --compilation-config '{"cudagraph_mode": "FULL"}' \
    --enable-chunked-prefill \
    --enable-prefix-caching
```

`enable_fa3` defaults to `false`. No new environment variable or model patch
is required. With the option disabled, existing attention selection is preserved.

## Attention and graph execution

- Uncached prefill uses the variable-length interface. Cached prefill, prefix
  hits, mixed prefill/decode batches, and multi-token verification use paged KV
  attention with right-aligned causal masking.
- KV writes use the existing Ascend cache writer and vLLM slot mapping. The FA3
  call receives the total KV length **after** those writes; it does not append KV
  a second time. Shared prefix pages remain read-only when writing a new suffix.
- The metadata builder consumes device-side query offsets and sequence lengths.
  It does not copy lengths to the CPU or infer decode causality from query length.
- `get_scheduler_metadata` executes device-side tiling once per distinct layer
  layout and interface per batch. Varlen and paged metadata are stored separately
  because their KV layouts differ. Layers reuse tiling, while each layer still
  computes attention from its own Q/K/V. Omitting metadata from the paged
  interface would select host tiling and introduce device synchronization.
- For captured token sizes, tiling and request metadata have stable addresses.
  The builder refreshes these buffers before model graph replay. Tiling runs
  outside the model graph because the operator's AICPU auxiliary-stream events
  are not capturable on every CANN release. Attention itself is captured, with
  no per-layer host task updates.
- Graph capture uses the paged interface, including uncached prefill. An uncached
  batch at a capturable token size prepares both kinds of tiling because eager
  warmup uses varlen while capture uses paged attention. Cached prefill, decode,
  chunked prefill, and speculative verification all use the paged interface;
  mixed prefill/decode batches do not require separate calls.
- Request dimensions include an extra padding row. Zero-length queries exclude
  padding, including a dummy request inserted by the FIA-oriented runner.

The standard Ascend graph dispatcher remains responsible for choosing FULL,
FULL_DECODE_ONLY, or piecewise execution. Model runner and `attention_v1.py`
changes are not required.

## Scope and speculative decoding

This backend targets dense decoder self-attention with FP16/BF16 KV cache and
128-token pages, including GQA/MQA. C8, other quantized KV caches, MLA, sparse
attention, context parallelism, sliding windows, ALiBi, and attention sinks are
outside its supported scope and are rejected.

Multi-token causal verification is tested independently of a draft algorithm.
Query offsets and KV lengths stay on device so that future EAGLE3, MTP, and
DSpark integration can consume accepted-token corrections. Distinct draft input
buffers retain separate graph metadata. This does **not** constitute end-to-end
validation of those speculative algorithms: their draft graph lifetimes,
accept/reject cache updates, and any tree or non-causal masks require additional
integration tests before enabling them in production.

## Validation

The varlen metadata sharing update has CPU regression coverage for sharing tiling
across layers while computing each layer's output separately. Its paired operator
patch has been checked with isolated CPU mocks for default and supplied metadata,
parameter mismatch rejection, and backward return arity. NPU and serving accuracy
validation of this update are pending; earlier results do not validate it.

```bash
pytest -q tests/ut/attention/test_flash_attention_v3.py
pytest -q tests/e2e/pull_request/one_card/test_flash_attention_v3.py
```

The NPU tests compare against FP32 CPU attention for FP16/BF16, GQA ratios 2
and 16, shared prefix pages, page-boundary writes, mixed queries, and multi-token
causal verification. Graph tests change request counts, query lengths, KV lengths,
and physical page tables between replays and assert that no host update tasks
were registered. Tests enter through the inherited full `forward()` method,
reject any call to its original FIA/PA branches, and count calls to the real FA3
wrappers during eager execution and graph capture. Serving accuracy must
additionally be evaluated with real model weights and the same dataset and
generation settings used for the FIA baseline.

Run the fixed-shape GQA microbenchmark on an otherwise idle NPU:

```bash
python benchmarks/benchmark_fa3.py --num-layers 4 --output fa3-benchmark.json
```

It checks FA3 against FIA before measuring eager FIA, eager FA3, and captured
FA3. FA3 timings include one device-side tiling invocation per group of layers;
graph timings also include copying the tiling result before replay. These are
operator timings, not model throughput or a comparison against captured FIA.
