# NOVA-KV 2-Bit KV Cache (Experimental)

NOVA-KV quantizes full-attention KV history to 2-bit payloads. Keys use
calibrated group vector quantization: each four-dimensional group has one
8-bit codebook index, equivalent to two payload bits per component. Values
use packed INT2 scalar quantization after rotation. At D256, each KV head
uses 134 bytes per token: 64 key-code bytes, 64 value-code bytes and three
BF16 parameters. Sink tokens and a recent-token ring additionally retain
original BF16 K/V. Hybrid linear-attention state uses the existing Mamba
allocator.

The implementation currently supports Ascend 950. Compressed decode
reconstructs KV tiles on chip and combines their attention output with the
BF16 exact windows. No persistent full-history BF16 cache is retained.

## Supported Configuration

The integration targets calibrated dense `qwen3_5_text` models with D256 full
attention, one to six query heads per KV head, at most four active requests
and at most 96K context. Artifacts must match the model and TP layout. Cache
pages must be a positive multiple of 128 tokens. The feature is disabled
unless `nova_kv` appears in additional configuration.

Compressed inference uses Model Runner V2, synchronous scheduling and PP1.
Uniform single-token decode supports `NONE` and `FULL_DECODE_ONLY` graph
modes; prefill and mixed batches run eagerly. Calibration capture uses BF16
attention and requires `--enforce-eager`.

Prefix caching, speculative decoding, KV transfer, context parallelism,
FlashComm sequence parallelism and NZ KV layout are unsupported. Cross-layer
KV sharing, sliding windows, ALiBi, learnable attention sinks, logits soft
caps and unequal K/V dimensions are also unsupported.

## Capture and Calibrate

Capture representative prompts using BF16 attention:

```bash
vllm serve /models/Qwen3.5 \
  --no-enable-prefix-caching --enforce-eager \
  --additional-config '{"nova_kv":{"capture_path":"/data/nova-capture"}}'
```

Capture writes a reservoir and query-moment file for each layer and TP rank.
Use a new directory for each model, TP layout and dataset. Keep at least
256 samples; all rank files must come from the same capture run. Calibration
can run on CPU:

```bash
python examples/nova_kv/calibrate.py --capture /data/nova-capture \
  --output /data/nova-artifacts --device cpu --iterations 25
```

## Serve With Compressed History

Install native Ascend TileLang in the worker environment for the fused
compressed-decode kernel. The original implementation used version 0.1.15.
The disabled and BF16 capture paths do not require this dependency, and no
standalone attention library build is needed.

```bash
vllm serve /models/Qwen3.5 \
  --tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --max-model-len 91024 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --block-size 128 --no-enable-prefix-caching --no-async-scheduling \
  --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4]}' \
  --additional-config '{"nova_kv":{"artifact_path":"/data/nova-artifacts","sink_tokens":64,"recent_tokens":256}}'
```

These are the complete `nova_kv` options. Only the presence of the `nova_kv`
object enables the feature.

| Option | Default | Constraint |
| --- | --- | --- |
| `artifact_path` | unset | Required for compression; excludes `capture_path` |
| `capture_path` | unset | Required for BF16 capture; excludes `artifact_path` |
| `sink_tokens` | 192 | Nonnegative |
| `recent_tokens` | 256 | Positive; ring includes one extra current-token position |
| `codec_batch_size` | 256 | Positive; bounds reference-encoder VQ workspace |
| `sample_tokens` | 8192 | Positive; capture reservoir size |

Prefill uses an 8,192-token query budget and reconstructs history in
32,768-token chunks. These are implementation constants. Set the engine's
`max_num_batched_tokens` to at most 8,192. Decode uses 128-token tiles and
1,024-token splits, or 4,096-token splits for batch 4 at configured contexts
of at least 65,536 tokens.

## Runtime and Memory

Model Runner V2 retains exact-window slots by request ID independently of
permanent request-state rows. Each step gathers slots into a stable device
buffer in input-batch order. Graph padding and capture rows use `-1` and
cannot write real cache pages or windows. Finished and preempted requests
release their slots. Before each scheduling step, workers report missing
windows; the NOVA scheduler preempts affected requests and recomputes their
history. This adds one host worker RPC per active scheduling step.

For `L` full-attention layers, local KV heads `H`, active tokens `T`, maximum
concurrency `B`, sink size `S` and recent size `R`, compressed records use
`L * H * T * 134` bytes, versus `L * H * T * 1024` bytes for BF16 K+V.
Exact windows additionally allocate
`L * (B + 1) * (S + R + 1) * H * 256 * 4` bytes. The extra slot is a sentinel.

Further costs include page padding, codebooks, projections, request metadata,
split outputs, exact-window tiles and prefill temporaries. Prefill may
temporarily reconstruct the active layer's history for one request; there
is no persistent full-history BF16 mirror. The 7.64x record ratio is not a
whole-device memory ratio. Higher concurrency requires new profiling and tests.

## Validation

Unit tests cover packet encoding, allocator aliases, backend selection,
request ownership, recovery and mixed-batch prefill dispatch. Ascend 950
tests cover cold attention against independent reconstruction, exact-window
boundaries, cache writes, padded requests and dynamic graph replay.

```bash
pytest -q tests/ut/quantization/methods/kv_cache/test_nova_kv.py tests/ut/test_nova_config.py
pytest -q tests/ut/attention/test_nova_request_state.py tests/ut/worker/v2/test_nova_kv.py
pytest -q tests/ut/worker/test_model_runner_v2_mamba.py
pytest -q tests/ut/ops/test_nova_kv.py tests/ut/patch/worker/test_patch_nova_kv.py
pytest -q tests/ut/patch/platform/test_patch_engine_core.py
pytest -q tests/e2e/pull_request/one_card/test_nova_kv.py
pytest -q tests/e2e/pull_request/one_card/test_nova_kv_decode.py
pytest -q tests/e2e/pull_request/one_card/test_nova_kv_exact.py
```

Full model/task accuracy, long-running memory behavior, broader hardware support and
performance require validation on the selected model and runtime.
