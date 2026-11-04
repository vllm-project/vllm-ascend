# FastEEP on Ascend (experimental)

This port adapts the retained-state design in
[vLLM FastEEP #44994](https://github.com/vllm-project/vllm/pull/44994)
to Ascend HCCL and CANN memory pools. Start all workers at the maximum
EP size and select an active prefix of ranks. Sleeping ranks release weights,
KV allocations, captured graph storage, and unused native HCCL communication
buffers. Worker processes and CPU rendezvous groups remain available. Returning
ranks recover the same virtual addresses, refill dense weights from a live peer,
and receive individual experts according to the new placement.

The runtime accepts Model Runner V1 or V2, BF16 weights, TP=PP=PCP=DCP=1,
one Python API process, internal DP routing, and AllGather or AllToAll MoE
communication. Nodes use a mutually reachable CPU rendezvous network and HCCL
network. Every rank has equal expert-slot capacity; the active ranks together
must have enough slots to cover all logical experts. Complete shard replicas
are not required. Continuous EPLB and prefix caching are disabled.

Eager and PIECEWISE graph execution are supported by the implementation.
PIECEWISE captures compute while MoE communication stays outside capture, so
native communicators can be recreated without retaining stale graph handles.
Graph storage is copied to CPU before physical memory is released, then restored
to the original addresses before replay. FULL graphs containing communication
handles are rejected. This configuration contract is distinct from the hardware
qualification results described below.

## Launch

Build the native extension with CANN headers providing `HcclCreateSubCommConfig`.
The C++ shim uses the installed `HcclCommConfig` ABI. Select V1 or V2 with
the existing `VLLM_USE_V2_MODEL_RUNNER` setting.

For Qwen3-30B-A3B-Base on four NPUs, create a placement JSON with 48 entries
in `layer_list`. Each entry has a `device_list` of four objects whose
`device_expert` arrays are respectively 0–63, 32–95, 64–127, and 0–31 plus 96–127.
Set `moe_layer_count` to 48. This provides 128 redundant experts and supports
active prefixes of two, three, or four ranks. The full example is in
`tests/e2e/pull_request/four_card/test_fasteep_serving.py`.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-30B-A3B-Base \
  --data-parallel-size 4 --data-parallel-size-local 4 \
  --enable-expert-parallel --api-server-count 1 --dtype bfloat16 \
  --enforce-eager --enable-sleep-mode --no-enable-prefix-caching \
  --worker-extension-cls vllm_ascend.distributed.fasteep.runtime.FastEEPWorkerExtension \
  --middleware vllm_ascend.distributed.fasteep.frontend.FastEEPMiddleware \
  --additional-config '{"eplb_config":{"dynamic_eplb":false,"num_redundant_experts":128,"expert_map_path":"placement.json"}}'
```

`--enable-elastic-ep` is left unset: the retained worker lifecycle is managed
by the FastEEP worker extension. The bundled stateless EEP platform changes
originate in [Ascend #19961](https://github.com/vllm-project/vllm-ascend/pull/19961).

Set `additional_config.fasteep.communication` to `"alltoall"` to use AllToAll;
the default is `"allgather"`. Optional `verify_migration: true` checks destination
weights immediately after each transfer and reports the checked tensor count.
The checks synchronize each layer and are intended for validation.

For captured execution, remove `--enforce-eager` and add:

```bash
--compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
```

## Resize

```bash
curl -X POST http://localhost:8000/fasteep_scale_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":2,"drain_timeout":120}'
```

The endpoint uses the server's configured API-key authentication. New inference
requests wait while existing responses (including SSE streams) finish. The
engine drains and clears its caches before the worker collective changes the
topology. The response is sent after every startup worker reports the committed
state. A second concurrent resize returns 409. Disconnecting the resize client
does not cancel an in-progress mutation. A failure after pausing keeps admission
closed and requires restarting the instance. A timeout while waiting for HTTP
responses reopens admission without changing topology. Close streaming-input
sessions before resizing.

The native `/scale_elastic_ep`, sleep/wake, pause/resume and collective-RPC
control endpoints are disabled in this mode. Worker
processes and CPU rendezvous groups remain resident until server shutdown.
Native migration communicators exist only during transitions. Active serving
communicators are rebuilt for the new group, and replaced startup device
communicators are destroyed. This releases their communication buffers while
retaining the process and device context.

## Validation

- CPU tests exercise placement validation, admission, streaming drain, routing,
  authentication, timeouts, cancellation and worker-state receipts.
- `test_fasteep_hccl.py` exercises actual four-NPU retained communicator handles,
  collective values, excluded ranks and peer copies.
- `test_fasteep_memory.py` exercises pooled weight/KV discard, peer refill,
  rank-local metadata restoration and address stability.
- `test_fasteep_serving.py` requires a compatible current-main build and the
  Qwen3 checkpoint; it checks repeated 4→3→2→4 transitions, deterministic outputs,
  streaming drain, admission during resize, and output after final restoration.
- `test_fasteep_expert_recovery.py` combines heterogeneous expert placement,
  actual native communicator destruction, zero-split AllToAll, captured graph
  release/restore/replay, and stable parameter addresses.

Run the targeted NPU tests on four idle devices. Component results and full
serving qualification must be reported separately for the exact tested source.

### Full-model functional qualification

Qwen3-30B-A3B-Base BF16 was validated on three Ascend 910B3 NPUs across
two nodes, using Model Runner V2, PIECEWISE graphs, AllToAll, and
TP=PP=PCP=DCP=1. Repeated DP=EP 3→2→3 transitions, streaming drain,
arrival admission, and final restoration produced 16/16 exact output-token
and finish-reason pairs, including 128 streaming chunks.

Each rank retained the same 388 captured graph objects. Inactive ranks
offloaded and restored 24 MiB of graph storage at the original addresses.
Observed inactive native communication reclamation was 145,805,312,
307,748,864, and 307,576,832 bytes across shrink transitions. Each rejoin
verified 6,144 expert tensors after migration. Initial MC2 handle cleanup
executed, and owned processes and reservations were released afterward.
These observations establish functionality for that configuration.

Single-node checks also compared V1 AllGather graph execution against eager
execution at each matched topology and an independent EP2 control. BF16
reduction grouping can change output across different EP topologies;
graph/eager parity is evaluated at the same topology. Qualification used
a normalization compatibility shim corresponding to
[Ascend #8384](https://github.com/vllm-project/vllm-ascend/pull/8384).

Performance of the earlier retained-communicator implementation is separate
from the reclaimable graph/communication implementation described here.
