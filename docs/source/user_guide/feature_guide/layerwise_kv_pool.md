# Layerwise KV Pool Configuration Contract

`AscendStoreConnector` saves and loads external KV one layer at a time. This
page is the common configuration reference for
[Ascend Store](kv_cache_pool_ascend_store.md),
[Mooncake layerwise pooling](mooncake_layerwise_adaptation_and_optimization.md),
and [shared-buffer offloading](layerwise_and_sparse_kv_cache_offloading.md).
The latter combines a Store connector with a separate P/D connector and has
additional deployment requirements.

## Support Matrix {#support-matrix}

This matrix describes the paths in current `main`. “Supported” means the
listed implementation and configuration are available; hardware validation
must cover the actual model, backend and parallel topology before deployment.
An attention hook or CPU configuration test alone is insufficient hardware
evidence. The PR introducing this reference records its validation scope and
requests maintainer confirmation of the release matrix.

| Configuration | Memcache / GVA | Mooncake / block-key | Yuanrong / key-based |
| --- | --- | --- | --- |
| `kv_producer` | Supported | Supported | Unsupported |
| `kv_consumer` | Supported; explicitly enable pool reads/writes | Supported; explicitly enable pool reads/writes | Unsupported |
| `kv_both` | Supported | Supported | Unsupported |
| `attention_v1`, `mla_v1`, `sfa_v1`, `dsa_v1` with compatible cache layouts | Supported | Supported | Unsupported |
| `attention_cp`, `mla_cp`, `sfa_cp`, `dsa_cp` with the topology constraints below | Supported | Supported | Unsupported |
| Single cache group | Supported | Supported | Unsupported |
| Multiple attention cache groups | Supported; groups retain their own layouts | Supported | Unsupported |
| MTP cache layers | Supported; included in physical-layer addressing | Supported; included in stage-local objects | Unsupported |
| Sparse/indexer attention cache | Supported; preserve component layouts | Supported; preserve component layouts | Unsupported |
| Non-`AttentionSpec` state with cross-layer buffer reuse | Unsupported | Unsupported; no NPU buffer reuse | Unsupported |
| Aligned recurrent state without buffer reuse | Supported subject to the model's cacheability policy | Supported; see the hybrid guide | Unsupported |
| Cross-layer NPU buffer reuse | Supported for compatible attention specs | Unsupported | Unsupported |
| Matching TP | Supported | Supported | Unsupported |
| Producer/consumer TP mismatch | Unsupported | Unsupported | Unsupported |
| PCP | Supported with matching producer/consumer shard layout | Supported with matching topology; no KVPP combination | Unsupported |
| DCP greater than one | Supported with matching producer/consumer shard layout | Unsupported | Unsupported |
| PP | Supported with global layer addressing | Supported with matching partition; no empty stage-local groups | Unsupported |

Attention integration includes `attention_v1`, `mla_v1`, `sfa_v1`, and
`dsa_v1`, together with the corresponding `attention_cp`, `mla_cp`, `sfa_cp`,
and DSA CP implementations. Their presence does not override backend, cache
layout, or parallelism restrictions in the matrix. Use the model's deployment
guide for graph mode, quantization and hardware requirements.

All sharing instances must agree on weights, cache format, block sizes,
transfer mode and namespace. Mooncake readers and writers additionally require
matching TP/PP/PCP topology and PP layer partitions. For hybrid cache-state
constraints, see [Mooncake hybrid attention](mooncake_hybrid_attention.md).

Yuanrong and unknown layerwise backends fail during Connector, Scheduler or
Worker initialization with the backend and conflicting data plane in the
error. Mooncake rejects GVA-only buffer options instead of ignoring them.
Existing guards reject TP mismatch, Mooncake DCP, empty PP groups, and
unsupported reuse layouts before transfer threads begin.

## Parameters and Defaults {#parameters-and-defaults}

Set these options in `kv_transfer_config.kv_connector_extra_config`, including
inside the Store child of `MultiConnector`. They are not `--additional-config`
options.

| Parameter | Default | Contract |
| --- | --- | --- |
| `use_layerwise` | `false` | Enable per-layer save/load for a supported backend. |
| `layerwise_prefetch_layers` with Memcache/GVA | `min(layerwise_num_shared_buffers, 8)` | Positive integer prefetch distance. The omitted buffer budget resolves to the number of cache-bearing layers. |
| `layerwise_prefetch_layers` with Mooncake/block-key | `2` | Preserve the current `main` default. Set `1` explicitly for the historical key-based default. |
| `discard_partial_chunks` | `true` | Identical default in ordinary and layerwise pooling; explicit supported overrides are retained. |
| `layerwise_num_shared_buffers` | Cache-bearing layer count | GVA only. A smaller budget enables compatible cross-layer reuse. Independent layers keep separate buffers. |
| `layerwise_independent_layers` | `[0]` | GVA only. Dedicated layers; `"all"` disables cross-layer reuse. |

Every reachable layerwise path uses the same prefetch parser. It accepts a
positive Python/JSON integer or a losslessly parsed decimal integer string,
such as `"3"`, `"003"`, or `"+3"`; surrounding whitespace is stripped.
Booleans, floats (including `2.0`), null, empty/non-decimal strings, zero and
negative values raise `ValueError` naming `layerwise_prefetch_layers` before a
backend client or transfer thread is created. Explicit null is an invalid
value, rather than an omitted setting. There is no additional upper clamp on
an explicit positive prefetch distance.

A consumer enables pool loads with `consumer_is_to_load=true` and pool writes
with `consumer_is_to_put=true`. Setting `use_layerwise=true` alone does not
enable these operations. Decode-only `save_decode_cache=true` without
`consumer_is_to_put=true` remains incompatible with layerwise transfer.

## Compatibility and Resources

The historical task specification used a key-based prefetch default of `1`.
Current `main` uses `2` for Mooncake block-key transfer; this change retains
that value. GVA defaults, explicit valid overrides and
`discard_partial_chunks=true` are unchanged. Invalid values previously accepted
by `int()` (particularly floats and booleans), unsupported backends, and
silently ignored GVA-only options now fail at initialization.

Prefetch controls when loads are submitted. It does not by itself allocate
extra NPU KV cache buffers; buffer counts follow cache planning and GVA reuse
settings. A wider window can change in-flight transfer metadata, read sessions,
leases and backend resources. Measure actual peak device/host memory and
latency/throughput with the selected backend and workload before tuning it.
No new memory or performance guarantee follows from the prefetch value.

## Validation

`tests/ut/distributed/ascend_store/test_layerwise_config.py` covers role and
default contracts, both parser paths, explicit overrides, unsupported backends,
GVA-only options and rejection before client/thread creation. Existing pool,
hybrid, PCP, pipeline and cache-layout tests cover the corresponding paths.
Representative hardware evidence must identify NPU model/card count, software
versions, TP/CP/PP topology, backend, model and command, and distinguish actual
backend/model execution from mocked tests run on an NPU host.
