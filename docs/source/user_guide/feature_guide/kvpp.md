# KVPP (Experimental)

## Overview

KVPP (KV pipeline parallelism) distributes historical KV caches that would otherwise be replicated across TP ranks by layer for non-hybrid MLA/SFA models. This reduces persistent cache storage per rank, allowing the same HBM capacity to accommodate more context tokens or concurrent requests.

KVPP can be combined with TP, EP, PP, and PCP.

The pipeline parallelism in KVPP refers to KV cache storage and communication, while model computation keeps the parallel configuration described above. For the underlying layer-wise KV storage concept, see **LayerSplit** in Z.ai's [Scaling Pain of Coding Agent Serving: Lessons from Debugging GLM-5 at Scale](https://z.ai/blog/scaling-pain).

## Use Cases

KVPP targets long-context inference and concurrent serving workloads where duplicated KV caches limit the available cache capacity. It can also be combined with prefix caching and KV pooling for workloads with reusable long prefixes.

On 950DT Products, the measured KV cache capacity was **6.25×** with GLM-5.2-W4A8C8 on one eight-device node using TP8 + EP, and **5.35×** with GLM-5.2-W8A8C8-mxfp8 on two eight-device nodes using TP8 + PP2 + EP (38/40 layers). Each comparison uses the same model, hardware, and memory budget with KVPP off and on. These capacity ratios are specific to the configurations in the performance section; they are not throughput multipliers or guarantees for other models.

## Supported Scenarios

### Supported Models

KVPP supports non-hybrid MLA and SFA models.

| Attention backend | KVPP support | Model scope |
| --- | --- | --- |
| MLA | ✅ Supported | Non-hybrid models using MLA |
| SFA | ✅ Supported | Non-hybrid models using SFA, such as GLM-5.2 |

Models must also meet the hardware, quantization, and attention-backend requirements of vLLM Ascend.

### Supported Features

The following table lists individual feature combinations with KVPP. It does not imply that all listed features can be enabled together.

| Feature or combination | Support | Conditions and limitations |
| --- | --- | --- |
| Graph mode | ✅ Supported | Eager and `PIECEWISE`; full graph modes are not supported. |
| TP | ✅ Supported | KV caches are assigned by layer within each cache-replica group. |
| EP | ✅ Supported | Requires an MoE model that supports EP. |
| PP | ✅ Supported | Use eager or `PIECEWISE`; see the [PP guide](pipeline_parallel.md) for the current E2E output-mismatch caveat. |
| Chunked prefill | ✅ Supported | Can be combined with KVPP. |
| Prefix caching | ✅ Supported | Can be combined with KVPP. |
| Asynchronous scheduling | ✅ Supported | Can be combined with KVPP. |
| LI-C8 and SFA-C8 cache layouts | ✅ Supported | Allocation follows the actual KV cache specifications. |
| Speculative decoding | ✅ Supported | Fixed-K speculative decoding, including MTP, DSpark, and Eagle3, when the selected method is supported by the model and attention backend. Draft-model caches remain independently allocated and must not share target KV caches; Dynamic-K speculative decoding is not supported. |
| PCP | ✅ Supported | Caches are shared across PCP × TP ranks. |
| DCP | ❌ Not supported | Cannot currently be combined with KVPP. |
| P/D disaggregation | ✅ Supported | Uses `MooncakeConnectorV2` (Experimental); enable KVPP only on the prefill node. |
| KV pooling | ✅ Supported | Memcache with `AscendStoreConnector`, `kv_producer` or `kv_both`, and whole-block loading. |
| Mooncake layerwise KV pooling | 🟠 Experimental | `AscendStoreConnector` with `backend="mooncake"`, `use_layerwise=true`, and `load_async=false`. GLM-5.2 validation pending. |

- ✅ **Supported**: The feature combination is supported under the stated conditions.
- 🟠 **Experimental**: The implementation is available, but the model-specific validation listed in the table is incomplete.
- ❌ **Not supported**: The combination is not currently supported.

## Usage

Enable KVPP through `enable_kvpp` in `--additional-config`:

```bash
vllm serve <model-path> \
    --tensor-parallel-size 2 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --additional-config '{"enable_kvpp": true}'
```

Replace `<model-path>` with a supported MLA/SFA model path and select TP according to model size and available devices. The KVPP group size follows TP × PCP and requires no separate setting.

If the launch command already contains `--additional-config`, merge `enable_kvpp` into the existing JSON object.

The example selects `PIECEWISE` graphs explicitly. To use eager execution, replace `--compilation-config` with `--enforce-eager`. Full graph modes, including `FULL_DECODE_ONLY`, are not supported with KVPP. Hybrid models and DCP are also not supported.

### Combining Features

KVPP can be combined with EP, PP, chunked prefill, prefix caching, and asynchronous scheduling. The following example uses an MoE model that supports EP:

```bash
vllm serve <model-path> \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 12 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --additional-config '{"enable_kvpp": true}'
```

Adjust the token budget and maximum number of sequences to device capacity and workload.

For PCP, add `--prefill-context-parallel-size` to the launch configuration. KVPP shares caches across PCP × TP ranks.

For PP, add `enable_kvpp` to the existing PP launch configuration.

## Configuration Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `additional_config.enable_kvpp` | `false` | Enables KVPP; the group size follows TP × PCP. |

## KV Transfer Configuration

### PD Disaggregation

KVPP supports P/D disaggregation through `MooncakeConnectorV2` (**Experimental**).

Use `MooncakeConnectorV2` with `kv_producer` on the prefill node and `kv_consumer` on the decode node. Enable KVPP only on the prefill node.

```bash
# Prefill node: include {"enable_kvpp": true} in --additional-config.
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":37000}'

# Decode node: leave KVPP disabled.
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":37010}'
```

Each worker uses a handshake port derived from `kv_port` and its parallel rank. Choose non-overlapping port ranges when both nodes run on one host.

### Memcache Pooling

KVPP supports `kv_producer` and `kv_both` for Memcache pooling. `kv_consumer` has not been tested and is outside the validated support scope.

Configure the memcache SDK and MetaService as described in [KV Cache Pool (Ascend Store)](kv_cache_pool_ascend_store.md), then add:

```bash
--kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache","use_layerwise":false,"load_async":true}}'
```

The example uses `kv_producer`; you can also set `kv_role` to `kv_both`. Both roles save and load prefixes in this pooling configuration. Keep `use_layerwise=false` and `discard_partial_chunks=true` (the default). KV events and consumer write-back are not supported with KVPP.

### Mooncake Layerwise Pooling

Experimental layerwise pooling is available with KVPP for the Mooncake block-key data plane. Each TP rank owns a contiguous share of the layers, loads only those layers from the pool, and the KVPP broadcast supplies the other ranks when a layer executes. `layerwise_prefetch_layers` sets the prefetch distance: the load of layer `L` is submitted at model layer `max(0, L - layerwise_prefetch_layers)`.

Configure Mooncake as described in the [Mooncake layerwise guide](mooncake_layerwise_adaptation_and_optimization.md), then add:

```bash
--kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"mooncake","use_layerwise":true,"load_async":false}}'
```

Mooncake layerwise pooling requires `load_async=false` (loads are submitted layer by layer) and does not enable cross-layer reuse of persistent owner caches. It was validated on a single node with TP8 and eager execution; PP, PCP, and DCP were not part of that validation.

Each owner waits for the layer's pool load to complete before starting its KVPP broadcast. Attention and the broadcast worker share the completion signal until the next step binds metadata. This ordering also applies when the store is a child of `MultiConnector`.

KVPP Mooncake objects use a separate topology namespace: ordinary replicated cache objects have different layer ranges and cannot be reused by KVPP. Enabling KVPP therefore starts a separate pool cache. Existing Mooncake restrictions on PCP, DCP, and producer/consumer topology mismatch still apply.

### Combining Connectors

Use `MultiConnector` to combine `MooncakeConnectorV2` with `AscendStoreConnector`. KVPP validates the type and role of each child connector. Set the P/D connector to `kv_producer` and enable KVPP only on the prefill node; child roles are configured independently of the wrapper role. The pooling backend still requires its own SDK and service configuration.

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_producer",
        "kv_port": 37000
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "backend": "memcache",
          "lookup_rpc_port": "0",
          "use_layerwise": false
        }
      }
    ]
  }
}
```

## Performance

The following measurements of GLM-5.2 on 950DT Products show the impact on cache capacity, time to first token (TTFT), and prefill throughput. The single-node and dual-node deployments use different quantized weights; compare KVPP on and off within each deployment.

### Test Configuration

| Setting | Single node | Dual-node PP |
| --- | --- | --- |
| Hardware | One node with 8 950DT Products | Two nodes with 8 950DT Products per node |
| Model | GLM-5.2-W4A8C8 | GLM-5.2-W8A8C8-mxfp8 |
| Parallelism | TP8 + EP | TP8 + PP2 + EP, 38/40 layer split |
| Common settings | chunked prefill, prefix caching, asynchronous scheduling, LI-C8, eager mode | Same as single node |
| Scheduling and memory | `max_num_batched_tokens=32768`, `max_num_seqs=12`, `gpu_memory_utilization=0.90` | Same as single node |

MTP and SFA-C8 were not enabled. All measurements below use a 32K scheduling token budget.

### KV Cache Capacity

The service calculated capacity automatically without a block-count override. Single-node values come from the pooling test configuration; dual-node values come from non-pooling service startup logs.

| Deployment | KVPP disabled (tokens) | KVPP enabled (tokens) | Capacity multiplier | Equivalent KV memory savings at the same token capacity |
| --- | ---: | ---: | ---: | ---: |
| Single node | 235,008 | 1,467,904 | 6.25× | 83.99% |
| Dual-node PP | 555,392 | 2,972,416 | 5.35× | 81.32% |

Equivalent KV memory savings are estimated from the capacity ratio and do not represent the same reduction in total HBM usage. Capacity gains depend on the model and parallel configuration.

### Time to First Token

This test measures long-input, zero-prefix-hit, low-concurrency requests. The single-node deployment uses GLM-5.2-W4A8C8 with eight 950DT Products and TP8 + EP. The dual-node deployment uses GLM-5.2-W8A8C8-mxfp8 with eight 950DT Products per node and TP8 + PP2 + EP (38/40 layers). Other service settings follow the test configuration above.

Input lengths are 32K, 64K, and 128K tokens. Prefix cache hit rate was 0%, with 1 output token. Each result is the mean TTFT of 4 requests at client concurrency 1. A positive change indicates increased latency.

| Deployment | Input length | KVPP disabled | KVPP enabled | TTFT change |
| --- | --- | ---: | ---: | ---: |
| Single node | 32K | 2.623 s | 2.614 s | -0.33% |
| Single node | 64K | 5.295 s | 5.375 s | +1.52% |
| Single node | 128K | 10.901 s | 11.268 s | +3.36% |
| Dual-node PP | 32K | 2.900 s | 2.905 s | +0.19% |
| Dual-node PP | 64K | 4.503 s | 4.887 s | +8.53% |
| Dual-node PP | 128K | 7.805 s | 8.665 s | +11.01% |

### Non-Pooling Prefill Throughput

Each run contained 40 requests at client concurrency 12, with 128K input tokens and 1 output token per request. KV pooling was disabled in both configurations.

| Deployment | Prefix cache hit rate | KVPP disabled (input tokens/s) | KVPP enabled (input tokens/s) | Throughput change |
| --- | --- | ---: | ---: | ---: |
| Single node | 0% | 12,492.4 | 12,205.7 | -2.30% |
| Single node | 90% | 116,516.2 | 110,224.8 | -5.40% |
| Dual-node PP | 0% | 21,395.2 | 18,742.3 | -12.40% |
| Dual-node PP | 90% | 177,375.4 | 154,717.3 | -12.77% |

The actual hit rate in all 90% scenarios was 89.9414%. Under these workloads, KVPP increased cache capacity; non-pooling throughput did not improve.

## Usage Recommendations

Evaluate KVPP first on workloads limited by KV cache capacity. Compare capacity, TTFT, and throughput with KVPP on and off using realistic input lengths, concurrency, and prefix distributions, then select a configuration that meets the service latency requirements.

When cache capacity is already sufficient, use measured latency and throughput to decide whether to enable KVPP.
