# FA3 inference with device-side tiling

The Qwen3 FA3 inference backend uses `flash_attn_with_kvcache` for both
prefill and decode, with automatic KV splitting (`num_splits=0`), from
[flash-attention-npu](https://github.com/MinghuasLab/flash-attention-npu).
It is separate from the [RL training-consistency backend](flash_attention.md).

## Installation and selection

Build the v3 operator for Ascend 910B/C:

```bash
git clone --recursive https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git checkout 7ce2a8926a2c10c92fd05c195cde33c49b32fc0f
source /usr/local/Ascend/cann/set_env.sh
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python setup.py install
```

The Python module must provide `flash_attn_npu_3.get_scheduler_metadata`
and `flash_attn_with_kvcache`. The varlen metadata interface patch is no longer
required by this backend.

Enable the backend through the Ascend platform's attention selector:

```bash
vllm serve Qwen/Qwen3-8B \
    --compilation-config '{"cudagraph_mode": "FULL"}' \
    --enable-chunked-prefill \
    --enable-prefix-caching
```

On A2/A3, for `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM`, the platform automatically
selects this backend when `flash_attn_npu_3` is installed and the attention
configuration is supported. No additional user switch is required. If the
package is absent, selection falls back to FIA. Other architectures, C8 or
quantized KV cache, context parallelism, and 310P/A5 hardware
retain their existing backends. Explicit RL training-consistency selection
also retains its existing backend.

## Attention and graph execution

- Uncached prefill, cached prefill, decode, prefix hits, mixed prefill/decode
  batches, and multi-token verification all use paged KV attention with
  right-aligned causal masking.
- KV writes use the existing Ascend cache writer and vLLM slot mapping. The FA3
  call receives the total KV length **after** those writes; it does not append KV
  a second time. Shared prefix pages remain read-only when writing a new suffix.
- The metadata builder consumes device-side query offsets and sequence lengths.
  It does not copy lengths to the CPU or infer decode causality from query length.
  The runner's existing CPU query offsets identify the active request prefix for
  tiling, without synchronizing device lengths.
- `get_scheduler_metadata` executes device-side tiling once per distinct layer
  layout per batch. Layers reuse tiling, while each layer still
  computes attention from its own Q/K/V. Omitting metadata from the paged
  interface would select host tiling and introduce device synchronization.
- For captured token sizes, tiling and request metadata have stable addresses.
  The runner updates request buffers and the builder refreshes tiling before
  model graph replay. Tiling runs
  outside the model graph because the operator's AICPU auxiliary-stream events
  are not capturable on every CANN release. Attention itself is captured, with
  no per-layer host task updates.
- Eager execution and graph capture use the same paged interface and metadata.
  Both tiling construction and attention execution pass `num_splits=0`, allowing
  the operator to select splitting. Mixed prefill/decode batches do not require
  separate calls.
- Request tensors reuse the runner-owned buffers without extra allocation,
  copying, or padding cleanup. Tiling receives the active batch rather than
  buffer capacity:
  inactive requests must not disable FlashDecode through its minimum query
  length and task-count checks. A dummy forward retains one empty request.
- FA3 does not consume `attn_state`: all scheduling states use the same paged
  interface. The capture entry point retains the argument for runner compatibility.

The standard Ascend graph dispatcher remains responsible for choosing FULL,
FULL_DECODE_ONLY, or piecewise execution. Model runner and `attention_v1.py`
changes are not required.

## Scope and speculative decoding

This backend targets dense decoder self-attention with FP16/BF16 KV cache and
128-token pages, including GQA/MQA. C8, other quantized KV caches, MLA, sparse
attention, context parallelism, sliding windows, ALiBi, and attention sinks are
outside its supported scope and retain their original attention backend.
Unsupported per-layer settings are excluded during backend selection, before
constructing FA3 metadata or implementation instances.

Multi-token causal verification is tested independently of a draft algorithm.
Query offsets and KV lengths stay on device so that future EAGLE3, MTP, and
DSpark integration can consume accepted-token corrections. Distinct draft input
buffers retain separate graph metadata. This does **not** constitute end-to-end
validation of those speculative algorithms: their draft graph lifetimes,
accept/reject cache updates, and any tree or non-causal masks require additional
integration tests before enabling them in production.

## Validation

CPU regression tests cover shared paged tiling, active batch counts, padding,
and empty forwards. NPU regression tests cover long-KV graph replay while
changing the active batch and switching between FlashDecode-eligible and
short-KV attention workloads. Historical serving accuracy results do not
validate subsequent backend changes.

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
