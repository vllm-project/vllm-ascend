# KVPP

## Overview

KVPP (KV pipeline parallelism) distributes historical KV caches that would otherwise be replicated across TP ranks by layer for non-hybrid MLA/SFA models. This reduces persistent cache storage per rank, allowing the same HBM capacity to accommodate more context tokens or concurrent requests.

When a layer executes, the rank responsible for its cache broadcasts the complete cache to the other ranks in the group. Model computation retains its TP/EP/PP configuration. With PP enabled, each stage assigns caches and broadcasts within its own cache-replica group. The group spans TP ranks, or PCP × TP ranks when PCP is enabled on Model Runner V2, and never crosses DP replicas or PP stages.

The pipeline parallelism in KVPP refers to KV cache storage and communication, while model computation keeps the parallel configuration described above. For the underlying layer-wise KV storage concept, see **LayerSplit** in Z.ai's [Scaling Pain of Coding Agent Serving: Lessons from Debugging GLM-5 at Scale](https://z.ai/blog/scaling-pain).

## Use Cases

KVPP primarily targets long-context and concurrent serving workloads limited by KV cache capacity. By reducing duplicated caches within a TP group, it stores more KV tokens within the same HBM budget. In the tests on 950DT products reported in the PR, cache capacity reached **6.25×** the disabled configuration on a single node and **5.35×** with dual-node PP.

| Workload | Value and selection criteria |
| --- | --- |
| Long-document question answering and long-context inference | Accommodates more KV tokens when historical caches consume substantial memory; actual maximum context length also depends on model limits and other memory overhead. |
| Concurrent requests | Provides more cache space for concurrent requests within the same HBM budget; increased capacity does not imply a proportional throughput increase. |
| Multiple reusable long prefixes | Keeps more prefixes resident in HBM and can reduce CPU cache loading when combined with pooling; see the performance section for measurements and version scope. |
| Sufficient cache capacity with minimum latency as the priority | Compare broadcast overhead before enabling KVPP; non-pooling throughput decreased in the reported tests. |

## Usage

Enable KVPP through `enable_kvpp` in `--additional-config`, using eager mode:

```bash
vllm serve <model-path> \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --additional-config '{"enable_kvpp": true}'
```

Replace `<model-path>` with a supported MLA/SFA model path and select TP according to model size and available devices. The KVPP group size follows TP × PCP and requires no separate setting. With PCP disabled, TP=1 provides no cache-sharing benefit across ranks.

If the launch command already contains `--additional-config`, merge `enable_kvpp` into the existing JSON object.

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
    --enforce-eager \
    --additional-config '{"enable_kvpp": true}'
```

Adjust the token budget and maximum number of sequences to device capacity and workload.

For PCP, set `VLLM_USE_V2_MODEL_RUNNER=1` and add `--prefill-context-parallel-size` to the launch configuration. KVPP shares caches across PCP × TP ranks; Model Runner V1 does not support KVPP with PCP.

For PP, add `enable_kvpp` to the existing PP launch configuration. Each stage allocates its caches independently. KVPP does not change PP layer partitioning.

Fixed-step MTP can be combined with KVPP, but MTP caches remain independently allocated and are excluded from KVPP layer partitioning. Follow the model-specific configuration requirements for MTP launch arguments.

## Configuration Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `additional_config.enable_kvpp` | `false` | Enables KVPP; the group size follows TP × PCP. |
| `--enforce-eager` | Not enabled | Required for KVPP; graph execution is not currently supported. |

KVPP broadcasts each full layer once. No broadcast granularity or separate KVPP parallel size needs to be configured.

## Support and Constraints

| Area | Scope |
| --- | --- |
| Models and runners | Non-hybrid MLA/SFA models; Model Runner V1 and V2 |
| Parallelism and scheduling | TP, EP, PP, chunked prefill, prefix caching, asynchronous scheduling |
| KV cache layouts | Allocated from actual specifications, including LI-C8 and SFA-C8 |
| Speculative decoding | Fixed-step MTP; variable-step MTP and other speculative decoding methods are not supported |
| Execution mode | Eager mode only; graph execution is not supported |
| Context parallelism | PCP requires Model Runner V2; DCP is not supported |
| KV pooling | Memcache with `AscendStoreConnector`, `kv_producer`, asynchronous whole-block loading; PCP disabled |
| PD disaggregation | `MooncakeConnectorV2`; enable KVPP on the prefill node only; PCP disabled |

Feature combinations must also meet the requirements of the model and the individual features.

### PD Disaggregation

Use `MooncakeConnectorV2` with `kv_producer` on the prefill node and `kv_consumer` on the decode node. Enable KVPP only on the prefill node. MTP caches remain replicated and are transferred alongside the persistent target caches.

```bash
# Prefill node: include {"enable_kvpp": true} in --additional-config.
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":37000}'

# Decode node: leave KVPP disabled.
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":37010}'
```

Each worker uses a handshake port derived from `kv_port` and its parallel rank. Choose non-overlapping port ranges when both nodes run on one host.

### Memcache Pooling

Configure the memcache SDK and MetaService as described in [KV Pool](kv_pool.md), then add:

```bash
--kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache","use_layerwise":false,"load_async":true}}'
```

This role both saves and loads pooled prefixes. Keep `discard_partial_chunks=true` (the default). Layerwise pooling, KV events, `kv_consumer`, `kv_both`, and consumer write-back are not supported with KVPP.

Each TP rank saves one complete object per token block containing its persistent target layers and its own MTP caches. Scratch buffers are excluded. Loading restores those same persistent buffers; the existing KVPP broadcast supplies other ranks when a layer executes. Pool lookup requires every nonempty owner shard across all PP stages.

## Performance

The following measurements on 950DT products from [PR #16094](https://github.com/vllm-project/vllm-ascend/pull/16094) show the impact on cache capacity, time to first token (TTFT), and prefill throughput. The single-node and dual-node deployments use different quantized weights; compare KVPP on and off within each deployment.

### Test Configuration

| Setting | Single node | Dual-node PP |
| --- | --- | --- |
| Hardware | One node with 8 950DT products | Two nodes with 8 950DT products per node |
| Model | GLM-5.2-W4A8C8 | GLM-5.2-W8A8C8-mxfp8 |
| Parallelism | TP8 + EP | TP8 + PP2 + EP, 38/40 layer split |
| Common settings | DSA-CP, chunked prefill, prefix caching, asynchronous scheduling, LI-C8, Model Runner V1, eager mode | Same as single node |
| Scheduling and memory | `max_num_batched_tokens=32768`, `max_num_seqs=12`, `gpu_memory_utilization=0.90` | Same as single node |

MTP, SFA-C8, and FlashComm1 were not enabled. This section presents the 32K token-budget results; see the PR for the complete 16K token-budget comparison.

### KV Cache Capacity

The service calculated capacity automatically without a block-count override. Single-node values come from the pooling test configuration; dual-node values come from non-pooling service startup logs.

| Deployment | KVPP disabled (tokens) | KVPP enabled (tokens) | Capacity multiplier | Equivalent KV memory savings at the same token capacity |
| --- | ---: | ---: | ---: | ---: |
| Single node | 235,008 | 1,467,904 | 6.25× | 83.99% |
| Dual-node PP | 555,392 | 2,972,416 | 5.35× | 81.32% |

Equivalent KV memory savings are estimated from the capacity ratio and do not represent the same reduction in total HBM usage. Each rank still needs two scratch buffers and independent MTP caches, so capacity gains depend on layer count, TP size, PP partitioning, and per-layer cache sizes.

### Time to First Token

Prefix cache hit rate was 0%, with 1 output token. Each result is the mean TTFT of 4 requests at client concurrency 1. A positive change indicates increased latency.

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

The actual hit rate in all 90% scenarios was 89.9414%. Under these workloads, KVPP increased cache capacity while adding broadcast overhead; non-pooling throughput did not improve.

### Pooled Prefill Throughput with Multiple Prefixes

These measurements use the pooling test version (vllm-ascend `493ec2b`, vLLM `b2f6858`). KV transfer integration is complete but is not included in this submission. The integration code and usage configuration will be merged in a follow-up submission.

Both KVPP configurations enabled `AscendStoreConnector` with the Memcache backend, `kv_role=kv_both`, `use_layerwise=false`, and `load_async=true`. CPU pool capacity was configured as 32 GB per device.

The scenario uses multiple reusable long prefixes. The total warmed prefix data exceeded HBM KV capacity in both configurations. HBM caches were retained after warmup, and both configurations used identical request data and ordering. Each run contained 40 requests at client concurrency 12, with 128K input tokens, approximately 90% shared prefix, and 1 output token per request.

- Single node: 16 prefix families were warmed; measurement used 8 families with 4 requests each and another 8 families with 1 request each.
- Dual node: 32 prefix families were warmed; measurement used 16 families with 2 requests each and 8 families with 1 request each. The remaining 8 families were used only for warmup.

| Deployment | Configuration | Throughput (input tokens/s) | HBM cache hit rate | CPU pool cache hit rate | Throughput change |
| --- | --- | ---: | ---: | ---: | ---: |
| Single node | KVPP disabled | 66,362.5 | 8.46% | 81.48% | — |
| Single node | KVPP enabled | 97,318.3 | 45.46% | 44.48% | +46.65% |
| Dual-node PP | KVPP disabled | 86,438.8 | 0.59% | 89.35% | — |
| Dual-node PP | KVPP enabled | 125,699.3 | 53.87% | 36.07% | +45.42% |

Cache hit rates use all input tokens as the denominator; the combined HBM and CPU hit rate was 89.94% in all configurations. With KVPP enabled, more prefixes remained in HBM and the share loaded from the CPU pool decreased, improving prefill throughput by approximately 45%–47% in this scenario.

All throughput values are total input tokens divided by measurement duration, including cached tokens. They do not represent the throughput of recomputed tokens or decode performance for long outputs. Every throughput run completed 40/40 requests without preemption. Percentages are calculated from unrounded measurements. Both dual-node configurations used the same temporary NFS forwarding route, with model loading and warmup completed before measurement.

## Usage Recommendations

Evaluate KVPP first on workloads limited by KV cache capacity. Compare capacity, TTFT, and throughput with KVPP on and off using realistic input lengths, concurrency, and prefix distributions, then select a configuration that meets the service latency requirements.

Broadcasts transfer all allocated logical blocks of a layer without filtering for active requests, so increasing cache capacity also increases broadcast volume. Prefetching can overlap part of the communication with computation, but the benefit remains workload-dependent. When cache capacity is already sufficient, use measured latency and throughput to decide whether to enable KVPP.
