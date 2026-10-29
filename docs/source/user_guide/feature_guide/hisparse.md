# HiSparse on Ascend

## Overview

HiSparse keeps sparse MLA KV pages in host memory and uses NPU memory for resident pages and a per-request hot cache. During decode, the sparse indexer selects logical tokens. The NPU resolves those tokens to resident or hot-cache rows and reads missing rows from mapped host memory. This allows the main KV history to exceed the resident device cache.

This experimental implementation requires the matching vLLM platform-backend changes. It has no released version yet. The development environment uses Ascend 950, CANN 9.2, torch-npu 2.10, and Triton-Ascend 3.6 development builds.

```text
Sparse indexer -> logical top-k tokens -> NPU residency resolver
                                             |
                               resident row / hot-cache hit / host miss
                                             |
                              mapped-host gather for missing KV rows
                                             |
                              pack latent and RoPE rows -> sparse attention
```

The resolver protects the union of tokens selected by all MTP query rows before choosing eviction slots. It uses device sorting and Triton-Ascend kernels with persistent scratch buffers. Decode does not copy top-k indices to the CPU for transfer planning.

The adapter reuses upstream scheduling, prefix-cache ownership, page spills, and host-mirror completion. Its current attention bridge moves invalid selections to the end while preserving valid KV order and duplicates, then copies selected rows into the split latent/RoPE layout accepted by Ascend SFA. Preserving the order avoids additional BF16 reduction differences. The valid count excludes padding from softmax. Sorting and device copying are part of the execution cost; a throughput or latency benefit must be measured for the intended workload.

KV preprocessing uses the same fused RMSNorm/RoPE operator as native SFA before copying the result into HiSparse's packed cache. A reusable split cache preserves the native operator's BF16 rounding and is allocated during profiling, before graph capture. Local temporary slots include padding; the final packed write masks invalid slots and inactive rows.

## Requirements and limits

| Area | Requirement |
| --- | --- |
| Hardware | Ascend 950 with AIV access to registered host memory; runtime capability is checked |
| Runner | Model Runner V2 |
| Model path | Sparse MLA/DSA with a sparse-index group |
| KV formats | BF16 indexer KV and unquantized main KV; model-weight quantization is independent |
| Context parallelism | Prefill and decode context-parallel sizes must both be 1 |
| Sparse selection | SFA requires top-k divisible by 128 |
| Hot cache | Representation supports at most 32768 aligned rows; NPU tests cover capacities through 10240. Capacity must cover the complete query-row union |
| Host allocation | Private pool per worker rank; account for the sum across TP ranks |
| Native extension | Build the Ascend extension for page DMA and sparse attention |

Mapped-host allocations remain registered until work using their addresses has completed. Host pools must remain alive across graph replay. Initialization fails if the hardware lacks the required mapping capability.

## Configuration

Set `VLLM_USE_V2_MODEL_RUNNER=1`, then add these options to a compatible model's launch command:

```bash
--block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

The example requests 4 GiB of host pool per rank, or 32 GiB across TP8. Size the pool for the model's layers and intended context capacity. Select the model's usual weight-quantization option separately.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `kv_connector_extra_config.host_pool_gib` | Required, positive | Host pool capacity in GiB per rank |
| `attention_config.indexer_kv_dtype` | Model dependent | Set `bf16` for this implementation |
| `attention_config.hisparse_config.device_buffer_size` | Derived from top-k and query-row count | Per-request hot-cache capacity, rounded to the cache block size |
| `attention_config.hisparse_config.eager_host_mirror` | `true` | Mirror decode KV writes during forward; prefill writes are always mirrored |

Omit `device_buffer_size` initially so the upstream allocator accounts for the number of MTP query rows. Changing MTP length changes the required union capacity.

## Operator validation

From the source checkout, with the Ascend environment and project virtual environment active:

```bash
cd /path/to/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest \
  --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_ops.py \
  tests/e2e/pull_request/one_card/test_hisparse_attention.py \
  tests/e2e/pull_request/one_card/test_hisparse_prefill_staging.py \
  tests/e2e/pull_request/one_card/test_hisparse_selection_order.py \
  tests/e2e/pull_request/one_card/test_hisparse_kv_preprocess.py -v
```

These tests compare state transitions and KV payloads with a CPU reference. They cover resident hits, hot-cache hits, host misses, reordered request state, padded rows, physical cache strides, compact gathers, invalidation after KV writes, and ACL graph replay. Large cases include top-k 2048 and four query rows with an 8192-row union. The attention tests independently compute QK, masked softmax, and weighted values in FP32 on CPU, including duplicate selections and empty rows. `--confcutdir` keeps these tests independent of the model-download fixtures. Passing operator tests alone does not establish full-model accuracy or performance.

The prefill staging tests check resident and host rows across shared-layer cache views and different resident page sizes. The selection-order tests check interleaved padding, duplicates, empty rows, and changed selections during graph replay.

The KV preprocessing tests include a fixed BF16 activation row that exposes a rounding difference between standalone and fused RMSNorm. They require native fused results to match exactly, including strided packed destinations, invalid slots, padding, and graph replay with changed inputs.


## Platform backend dependency

This draft targets vLLM PR [#62592](https://github.com/vllm-project/vllm/pull/62592),
commit `d9f25459725c8fd9db07b7d9571a3f5be39db592`. Pin the paired revisions
when validating the interface.

`NPUPlatform.get_hisparse_backend_cls()` selects `AscendHiSparseBackend`.
The common runtime owns scheduling, TP coordination, allocation rollback and
synchronization before Host release. The plugin owns NPU operations and view
adaptation through `validate_kv_cache_config()` and `prepare_kv_cache_views()`.
The previous patch of `init_hisparse_kv_cache` is removed: indexer views are now
prepared before common handle binding, inside its rollback scope.

`gather_resident_rows()` copies complete payloads while preserving page strides
and layer offsets. FP8-typed payloads use `uint8` views before indexing, avoiding
unsupported NPU FP8 indexing without numerical conversion. This primitive does
not add end-to-end attention C8 support to this BF16 feature PR. Native attention
C8 integration remains a separate experimental branch. The CUDA `fp8_ds_mla`
prefill conversion path is explicitly rejected rather than reinterpreting its
layout as Ascend storage.

Host pools remain private per worker rank. Local event creation accepts
`interprocess=False`; NPU IPC event creation/export/import reject calls until
that protocol is implemented and validated. Existing context-parallel and model
restrictions remain.

CPU checks for this interface update cover opaque BF16/E4M3/E5M2 payloads,
strided pages, layer offsets, repeated/padded IDs, geometry rejection and the
actual paired common Prefill staging entry point. NPU graph replay tests with
changed IDs and payloads are included but have not run on this updated PR head.
Historical integration evidence does not qualify this new pair. Complete test
collection, standard CI, final-head NPU/model validation and deferred accuracy
remain outstanding; no performance improvement is claimed.


### Speculative Decode group prefetch

The Ascend backend opts into the common runtime's
`supports_speculative_group_prefetch` capability. A pure Decode batch may stage
the shared index group's missing historical rows for its followers, including
MTP verification with multiple query tokens. Prefill, mixed batches and batches
exceeding the resolver row capacity keep the existing staging path.

This eligibility is separate from the single-token resident-write and Host-mirror
policy. Each follower retains its own completion event and writes its current
tentative KV before attention consumes it. This change does not establish a
measured speedup, complete attention C8 support, or full-model accuracy.
