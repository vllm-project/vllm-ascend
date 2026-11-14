# DeepSeek V4.1 Mooncake V2 Push

## Scope

This experimental integration connects the existing DeepSeek V4.1 cache-write
notifications to `MooncakePushConnector`. `MooncakeConnectorV2` selects that
connector when `kv_connector_extra_config.use_layerwise` is `true`.

PR #21690 provides connector selection and full allocation-block fan-out through
`AscendMultiConnector`. The V4.1 integration adds model-specific resource
readiness and SWA replay validation to the existing V2 transport. It reuses
`build_layer_cache_access_plan`; it does not introduce a second model topology
or copy the legacy connector's transfer threads.

## Supported configuration

Use the same model, TP size, cache specifications, kernel block layout and
`swa_bounded_replay` setting on P and D. Both ends must run the updated code.
PP, PCP and DCP must all be 1. Speculative decoding and KVPP are unsupported.
Producer preemption/resumption is rejected; size the producer cache and workload
to avoid preemption. Broader topology support requires separate validation.

On P, configure:

```json
{
  "kv_connector": "MooncakeConnectorV2",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {"use_layerwise": true}
}
```

On D, use `kv_role: "kv_consumer"` with the same extra configuration. Explicit
`MooncakePushConnector` is also supported. Use the existing
`examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py`
proxy. When combining with another connector, configure the option on the
Mooncake child and use `AscendMultiConnector`, which forwards V4.1 notifications
in slot-release-provider order.

## Components

| Component | Responsibility |
| --- | --- |
| `models/deepseek_v41/layerwise_cache_access.py` | Resolve execution layers, cache owners, reads and potential writes. |
| `attention/dsa_v41.py` | Notify after a real cache write, including SWA slots and cache-only replay. |
| `mooncake/connector.py` | Forward V4.1 notifications to the Push worker while connector metadata is bound. |
| `mooncake/dsv41.py` | Track resource ownership and request-local replay coverage. |
| `mooncake/push_scheduler.py` | Publish immutable allocation/range snapshots, truncate stateful prefill and reject resumption. |
| `mooncake/push_worker.py` | Bind the plan, publish explicit readiness and execute existing WRITE/DONE tasks. |
| `mooncake/metadata.py` and `base_scheduler.py` | Preserve the V4.1 plan signature through worker-to-scheduler metadata aggregation. |

Each worker owns its plan and coverage state. The model thread alone mutates
that state. Sender threads use immutable range snapshots and device events;
they do not mutate the plan. Request retirement discards its replay coverage.

## Resource readiness

SWA belongs to each execution layer. Long KV, indexer K, folded indexer K and
compressor state are published only by their owner according to the topology.
A consumer layer reading shared KV does not publish it as a new write.

The flow is:

1. Register all configured cache tensors and build the full backbone access
   plan. Reject a missing or unexpected transfer resource.
2. Exchange a signature containing the access plan, cache specs and replay
   setting. Check matching P/D topology, groups, block sizes, scales, payload
   lengths and shapes before dispatching WRITEs. Base addresses and page strides
   remain local descriptors and may differ between P and D.
3. Before forward, install the scheduled request ranges. A prefix-hit count
   alone does not mark all V4.1 resources ready.
4. On the explicit post-write hook, mark only the plan's written resources ready
   and record a compute-stream event. The existing Push pool can transfer full
   immutable blocks while later layers execute.
5. At `wait_for_save`, verify every required ordinary resource was published for
   the current range. V4.1 does not use the generic end-of-forward fallback.
6. Drain submitted WRITEs, then send terminal DONE with success or failure.
   Preserve the existing source-block lifetime and cancelled-receive protection.

Ordinary chunked caches transfer complete blocks until the final chunk, which
also includes a partial tail block. Circular compressor state transfers only at
the final chunk. V2's descriptor-based address calculation retains separate
payload length and page stride, including packed or strided cache components.

## Bounded SWA replay

Cache-only replay can populate SWA without completing a normal layer forward.
Its callback records slots but does not publish shared KV or compressor state.
The normal callback publishes the source layer's shared resources.

SWA slot mappings are cloned on the compute stream at notification time because
later replay forwards may reuse the original buffers. At the step boundary,
the plan copies these snapshots to CPU and accumulates coverage per request and
SWA resource. For prefill length `L` and SWA window `W`, the required tail is
`[max(0, L - W + 1), L)`; the first D query supplies its own window slot.

Coverage is restricted to the current target slots across chunks. The final
chunk must have resident source blocks and complete coverage in every SWA
resource before publishing SWA readiness. Missing slots or blocks produce a
failed DONE. Token progress alone cannot turn incomplete replay into success.

The CPU inspection happens at the step boundary, not inside each layer's hook.
It still introduces device synchronization and Python set work. Hardware
profiling must measure that cost and the slot snapshot memory before claiming
a performance improvement.

## Validation and acceptance

Run in the documented vLLM Ascend CPU test environment:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
  tests/ut/kv_offload/mooncake_v2/test_dsv41_plan.py \
  tests/ut/kv_offload/mooncake_v2/test_push_sending.py \
  tests/ut/kv_offload/mooncake_v2/test_push_scheduler.py \
  tests/ut/kv_offload/mooncake_v2/test_connector.py \
  tests/ut/kv_offload/mooncake_v2/test_metadata.py \
  tests/ut/kv_offload/mooncake_v2/test_base_scheduler.py \
  tests/ut/kv_offload/test_ascend_multi_connector.py
bash format.sh ci
```

The plan tests cover ownership, cache-only callbacks, chunk accumulation, reused
buffers, missing slots/blocks, request isolation and configuration boundaries.
Push integration tests cover explicit readiness, rejection of the generic
fallback, replay-gated completion and incompatible peer layouts. Existing
Push tests continue to cover asynchronous WRITE failures and cancellation.

Before production use, run the following on Ascend 910B/C with the same model
and sampling parameters as a colocated baseline:

| Scenario | Required evidence |
| --- | --- |
| Cold prefill, chunking on/off | Correct output and transferred content for each resource. |
| Prefix hits on P, D, and both | Matching output, including compressed-cache boundaries. |
| Bounded replay on/off | Final SWA tail coverage and identical decode results. |
| Short prompts, nonaligned lengths, multiple chunks | No missing blocks or premature DONE. |
| Batched requests and MultiConnector | Request isolation, full allocation blocks and ordered callbacks. |
| WRITE failure, cancelled P/D request, missing callback | Failed receive, no successful DONE and safe block lifetime. |
| Mismatched replay/layout/TP and producer resumption | Explicit rejection. |
| Sustained workload | TTFT, throughput, device/host memory, snapshot overhead and synchronization cost. |

Record hardware, CANN, torch/torch_npu, vLLM commits, model revision, complete
launch commands and observed results. CPU mocks and isolated plan tests do not
establish NPU correctness or performance.

## Follow-up work

Producer resumption needs a protocol to retire old transfer progress and reset
coverage before a new allocation is used. PP requires partitioning the topology
and resource owners; PCP/DCP and different P/D TP layouts need explicit sequence
and replica mappings. Speculative execution needs draft namespaces and final
state semantics. These extensions should follow hardware validation of the
restricted path above.
