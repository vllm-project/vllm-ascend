# Mooncake Layerwise Hybrid Attention

This extension adds multi-KV-group support to the Mooncake layerwise range/session
implementation introduced in [PR #16004](https://github.com/vllm-project/vllm-ascend/pull/16004).
The implementation was based on upstream main at
`be427041bf63a620e4a637b60f2656e87dcdf8f6`.

It targets attention layouts such as DeepSeek-V4, where sliding-window KV,
compressed KV, indexer caches, and compressor state can belong to different KV
groups. It reuses the shared reachability coordinator also used by Memcache;
see the design discussion in [issue #12234](https://github.com/vllm-project/vllm-ascend/issues/12234).
It does not translate Mooncake operations into Memcache GVA operations.

## Scope and prerequisites

- Use `AscendStoreConnector` with `backend="mooncake"` and `use_layerwise=true`.
- Install a Mooncake client with the range/session interfaces listed in the
  [single-group guide](mooncake_layerwise_adaptation_and_optimization.md#51-mooncake-version).
- The model must provide valid `KVCacheConfig.kv_cache_groups` and group block
  tables. Runtime group block sizes already describe raw-token coverage; do not
  multiply them by the compression ratio again.
- This patch wires compute windows into the Ascend DSA, FA, and SFA attention
  paths. Start validation in eager mode. Graph-mode execution and additional
  attention backends require separate integration validation.
- PP, DCP, PCP, and prefill/decode TP mismatch retain the restrictions of the
  upstream-main Mooncake layerwise implementation. This branch does not include
  the independent PP/DCP adaptation.
- Recurrent Mamba state is explicitly rejected. Hybrid attention and hybrid
  recurrent/linear-attention state are not interchangeable.
- Only complete, coordinator-aligned block snapshots are published. Partial
  block offloading and cross-layer cache-buffer reuse are not part of this
  hybrid implementation.

## Group-aware objects and range offsets

The single-group wire format is unchanged. Hybrid layouts use a separate
namespace:

```text
model@mooncake_hybrid_v1:<layout-digest>@group:<id>@family:<family>@block:<size>@<hash>@<head>
```

The digest includes TP size, ordered group membership, cache-spec types, and
cache-spec fields. It prevents incompatible group layouts from reading the same
objects. It is not a model-weight checksum or a tenant isolation mechanism:
use isolated deployments for different weights with identical model names and
cache specifications.

Each key stores one block for one group and storing head/rank. Its bytes contain
only that group's registered cache entries, ordered by physical layer and cache
name. Object size is the sum of their actual per-block byte lengths. A transfer
uses the group-local layer index to calculate remote offsets, while the physical
layer index selects the compute event and completion signal.

For example:

| Group | Raw tokens/block | Physical layers | Commit boundary |
| --- | --- | --- | --- |
| Full attention | 16 | 0, 2 | Layer 2 |
| Compressed KV | 32 | 1, 2, 3 | Layer 3 |
| Window/state | 16 | 0, 3 | Layer 3 |

Several groups can transfer at one physical layer. A group is committed after
its own last layer, not necessarily the model's last layer. Multiple cache
entries at the same physical layer remain separate byte ranges.

## Reachability and session lifetime

The scheduler queries `batch_is_exist` for the keys selected by the coordinator's
per-group lookup masks. A block is usable only if all required storing-head keys
exist. The coordinator then determines a common reachable token boundary across
groups. A full-attention hit alone is insufficient when the corresponding
window or compressor state is missing.

The worker uses the same coordinator for store and load masks. It does not
interpret a sparse state cache as a contiguous prefix. A mask-generation failure
is not silently converted into a request to copy every state block.

The worker creates separate request views for each group. Session ownership is
indexed by `(request_id, group_id, logical_block_index)`, so block zero in one
group cannot replace block zero in another. Across chunked-prefill steps, committed
keys can be restored using the current local block IDs even without a new
scheduler load specification. Each group retains independent active-transfer
state and resets it at its last group layer.

A failed range put revokes the affected keys; successful keys in other groups
can still commit. A preparation exception revokes the objects opened earlier in
that preparation. A failed get stops the forward path before incomplete hybrid
KV/state is used. Get sessions are released after in-flight reads finish.

## Transfer timing: overlap attention, fence communication

The heavy payload operations use a per-layer attention compute window:

```text
previous collectives / cache updates
              |
       cache-ready NPU event
              |
     +--------+---------------------+
     |                              |
 current attention kernel     put current-layer ranges
                              get future-layer ranges
     |                              |
     +---------- transfer drain ----+
                         |
              output projection / MoE communication
```

At attention entry, an NPU event protects cache writes and preceding work on the
compute stream. Prefetched gets wait for that event. The worker records the save
event and submits current-layer puts at the same boundary. The host then launches
the attention kernel and waits for queued payload transfers before returning to
the following output projection or MoE communication.

In DSA this boundary is after the compressor/indexer/cache updates and before
the sparse-attention operator. In SFA it surrounds the sparse-attention operator;
in FA it surrounds the paged/FIA attention path. It does not surround the whole
transformer layer.

Consequences:

- If transfer time exceeds attention compute time, the remaining transfer time
  delays subsequent communication. It is not hidden by launching HCCL concurrently.
- An initial, unprefetched demand load has no preceding attention window. It
  waits for earlier compute-stream work, then completes before attention starts.
- Session allocation, existence queries, and other metadata RPCs are not payload
  DMA and can occur outside the window.
- This is a local execution-order fence, not a cluster-wide bandwidth scheduler.
  Unrelated workers and independent communication streams are not globally
  serialized. Validate multistream and multi-rank behavior using device traces.
- No latency or throughput improvement is claimed without hardware measurements.
  Short attention kernels can expose a substantial transfer tail.

## Configuration

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
vllm serve /path/to/hybrid-model \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "backend": "mooncake",
      "use_layerwise": true,
      "layerwise_prefetch_layers": 1,
      "layerwise_max_transfer_blocks": 64,
      "layerwise_max_transfer_bytes": 16777216
    }
  }'
```

Choose TP size, chunk size, quantization, and model arguments for the actual
hardware/model. Begin with one future-layer prefetch window. This example is
not a verified DeepSeek-V4 deployment recipe or a memory-capacity guarantee.

## Validation

CPU/mock unit tests cover group-aware byte round trips, different block sizes,
unequal cache-entry counts, sparse masks, independent commit boundaries, group
session ownership, continuation with remapped local blocks, negative transfer
results, allocation rollback, device-event gating, and fatal-thread propagation.
They do not execute attention kernels or a real Mooncake service.

For opt-in NPU validation, start Mooncake separately and run from the repository:

```bash
python tests/e2e/nightly/single_node/models/scripts/mooncake_layerwise_hybrid.py \
  --model /path/to/hybrid-model --tp 8 \
  --prompt-tokens 32768 --chunk-tokens 4096 \
  --warm-runs 3 --profile-dir /tmp/mooncake-hybrid-traces
```

The script compares greedy output tokens from a fresh-prefix cold run against
warm runs. It resets the local prefix cache without clearing the remote connector
between runs and requires a positive cached-token count. It reports complete
generation wall time and request TTFT when available; these are different metrics.
Use `--engine-args` for additional model-specific LLM arguments.

Repeat with `--non-layerwise` for the existing Mooncake non-layerwise baseline,
using otherwise identical settings and an isolated test pool. Run performance
measurements again without profiling or range-debug logging enabled.

Before claiming hardware support or a performance improvement, verify:

1. Cold/warm token equality, positive remote hits, and multiple chunk boundaries
   on the real hybrid model.
2. Every group, including sparse state and indexer groups, has valid load ranges.
   Check Mooncake range-debug logs alongside the device trace.
3. Put/get payload ranges begin after cache-ready events and complete before
   subsequent HCCL on the tested device; inspect all relevant streams and ranks.
4. TTFT, throughput, exposed transfer tails, and HCCL duration against the
   non-layerwise baseline under the same model, TP, prompt, and pool conditions.
5. Additional concurrent-request, preemption, model-specific state, and service
   failure tests. The smoke script alone does not establish production readiness.

Real NPU/Mooncake validation is pending; the implementation environment only
supports CPU/mock checks.
