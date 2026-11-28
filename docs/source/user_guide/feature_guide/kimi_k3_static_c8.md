# Kimi K3 Static FP8 MLA Cache

## Overview

Static C8 reduces the memory used by the Kimi K3 MLA KV cache by storing
its latent and auxiliary key in FP8 E4M3 instead of BF16. The MLA payload
falls from 1152 to 576 bytes per token per layer, leaving more KV memory
available for concurrent requests. Fused Quant MLA Prolog and Quant Flash MLA
reduce decode preparation and KV-read overhead.

KDA/Mamba states and DSpark draft caches retain their existing formats.
The 50% reduction applies to the target MLA payload, not the entire hybrid
KV cache or total model memory.

## Prerequisites

- Use a Kimi K3 NoPE MLA checkpoint with latent/key dimensions of 512/64
  and W8A8_MXFP8 projection weights. See the
  [Quantization Guide](quantization.md) for model quantization information.
- Use Model Runner V2 on an NPU and software stack that support MXFP8
  projections and the required C8 operators.
- Install the matching C8 operator packages before starting the server:
    - The arena-net DSL 0.8 package must provide `ops.quant_mla_prolog`.
    - The `cann_ops_transformer` package must provide
      `quant_flash_mla_metadata` and `quant_flash_mla`.
- For P/D disaggregation, install compatible operator packages and use the
  same C8 cache format on both engines.

C8 uses the Kimi recipe's fixed per-tensor KV descale of `1.0`.
The checkpoint does not need additional attention-cache scale entries.

## Usage

### Enable C8

Set `mla_cache_mode` in `--additional-config`. This is the only C8-specific
configuration option; its default is `C16`, which keeps the existing MLA
execution and cache configuration.

The following example runs a TP8/DCP8 instance with decode Graph enabled:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve /path/to/kimi-k3-mxfp8 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --block-size 1664 \
  --cp-kv-cache-interleave-size 1664 \
  --kv-cache-dtype auto \
  --additional-config '{"mla_cache_mode": "C8"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Replace the model path with your quantized checkpoint. If your deployment
already uses `--additional-config`, add `mla_cache_mode` to that JSON object.
To use eager execution, replace `--compilation-config` with `--enforce-eager`.

The block size and interleave above are existing framework settings. Use
1664 for this C8 deployment; 128 is an internal Prolog workspace size, not
the FA cache block size. See [Context Parallel](context_parallel.md) and
[Graph Mode](graph_mode.md) for the general framework options.

### P/D disaggregation

Add `mla_cache_mode=C8` to both engines in your existing P/D deployment.
Use `MooncakeConnectorV2` or `MooncakePullConnector`, and apply these settings:

| Setting | Prefill producer | Decode consumer |
| --- | --- | --- |
| `mla_cache_mode` | `C8` | `C8` |
| `decode_context_parallel_size` | `1` | `1` or `8` |
| `block_size` | `1664` | `1664` |
| `cp_kv_cache_interleave_size` | Not used with DCP1 | `1664` with DCP8 |
| Execution | `--enforce-eager` | Eager or `FULL_DECODE_ONLY` |
| Connector role | `kv_producer` | `kv_consumer` |

For example, retain your existing connector options and select the roles
through `--kv-transfer-config`:

```json
{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer"}
```

```json
{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_consumer"}
```

Both engines must be updated together when changing the C8 cache format.
Existing P/D routing and transport configuration still apply.

### DSpark

C8 can be combined with DSpark. Configure the draft model through the
existing `--speculative-config` option described in
[Speculative Decoding](speculative_decoding.md#speculating-using-dspark).
Draft caches keep their own BF16 layout; enabling C8 affects the target
MLA cache.

## Limitations

- Supported configurations use Model Runner V2, PCP=1, and DCP=1 or DCP=8.
- DCP8 requires `block_size=1664` and `cp_kv_cache_interleave_size=1664`.
- P/D producers use DCP1 and eager prefill. Decode supports eager execution
  or `FULL_DECODE_ONLY`; other Graph modes are not supported by this path.
- When speculative decoding is enabled, only DSpark is supported.
- The fused Prolog path requires matching MXFP8 weights and operator ABI.
  Other supported projection geometries use native projections with the
  same C8 cache format.
- Increased request capacity depends on the complete hybrid-cache layout,
  prefix reuse, and request lengths. MLA payload savings do not imply a
  fixed whole-model memory reduction or throughput gain.

## Implementation Overview

Prefill computes the current chunk in BF16 and quantizes new KV entries
into the persistent C8 cache. Cached history is dequantized when needed.
Decode uses fused Quant MLA Prolog and Quant Flash MLA. FA reads native
1664-token PA_NZ pages; Prolog writes a preallocated 128-token workspace
whose new KV entries are copied into those pages.

Native-page allocation accounts for each layer's actual storage, including
BF16 draft caches and complete Mamba states. DCP preserves request-level
batches and merges attention contributions before the local V projection.
Mooncake transfers the selected KV spans using each endpoint's physical
layout.
