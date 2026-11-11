# Ascend-owned Store recovery protocol

## Scope and version

The partial-hash path of `AscendStoreConnector` owns its coordinator, scheduler,
worker and wire metadata in `kv_pool/ascend_store/recovery/`. It does not inherit
the upstream Mooncake Store classes or access their private fields. The initial
protocol port is based on vLLM v0.33.0, commit
`4ba787ec9959e566d3b587a796dff07673324da5`, and Ascend baseline `d64d5967a`.
Source attribution is retained in each ported module.

This is an ownership change, not a new user-selectable cache mode. Core resolves
`--prefix-match-unit` and the cache geometry. The connector selects the recovery
stack when Core's recoverable hit alignment is finer than the scheduler block.
The existing whole-block and layerwise implementations remain separate and retain
their existing gates. There is no `pool_prefix_mode` switch.

## Responsibilities and Core contracts

| Owner | Core contract and behavior |
| --- | --- |
| `geometry.StoreGeometry` | Calls Core's `resolve_kv_cache_block_sizes`; captures the resolved scheduler/hash sizes, prefix-cacheable groups (including physical specs), hit alignment, retention and EAGLE policy. Scheduler and worker receive this plan, rather than deriving competing sizes. Producer-only EAGLE normalization copies group specs without mutating Core's plan. |
| `coordinator.StoreCoordinator` | Uses Core's cache-spec registry, single-type cache managers, `replay_boundaries` and EAGLE proof rules to intersect group recoverability. A hit requires a common recoverable Attention/Mamba boundary, not merely an existing Attention key. |
| `scheduler.StoreScheduler` | Implements connector lookup/allocation callbacks and consumes `SchedulerOutput`. Core still decides the final local APC/external hit and passes that decision into `update_state_after_alloc`. The scheduler never replaces that decision with its earlier lookup estimate. |
| Scheduler boundary handoff | `_handle_boundary_state_offloads` consumes Core boundary snapshots; `register_finished_partial_tail` captures the final partial tail. `BoundaryPut` carries explicit group, physical block and token boundary into the save job. Attention and Mamba use the same prefix hash for that recoverable boundary. |
| Scheduler block lifetime | `_reference_save_blocks` touches Core `BlockPool` blocks before request cleanup; monotonically numbered save jobs own those references. `update_connector_output` frees them only after all worker ranks acknowledge that IO has stopped. |
| `worker.StoreWorker` | Orchestrates async lookup, receive and send queues. `ReqMeta`, `LoadSpec`, ordinary pool keys and `TailKeyBoundary` retain the upstream data contract. Invalid loads feed the connector's load-error reporting; save errors are reported in worker metadata and retire only after all ranks acknowledge. |
| Ascend device adapter | `AscendHashStoreWorker` in `pool_worker.py` implements NPU registration, real storage/stride descriptors, event synchronization and the existing shared-engine/direct-only rules. It derives from the Ascend-owned worker. Native MooncakeConnectorV2 transport is unchanged. |

`data.py`, `protocol.py`, `events.py` and `metrics.py` are shared support modules,
not additional control layers. Runtime dependencies on vLLM remain its connector
base contracts, Core cache/spec APIs, configuration and native Store setup/RDMA
utilities. The differential tests alone import upstream Store classes as an oracle.

## Save lifetime and failure ordering

1. Core supplies a stable boundary snapshot and scheduler block IDs.
2. Scheduler builds a save job and pins its physical blocks, including CoW tails.
3. Worker waits for the NPU event and queues the immutable job metadata.
4. IO completes, fails, or is cancelled before starting. The worker's `finally`
   path always acknowledges the stopped job and completes the queue task.
5. Worker metadata reports job IDs, rank identities and failures. Rank identities
   are unioned so duplicates and out-of-order delivery cannot prematurely free
   references.
6. Scheduler releases the references after every expected rank has acknowledged.

Cancellation invalidates work but does not authorize freeing a buffer still used
by IO. `has_pending_push_work` keeps Core stepping to collect late completions.
Reset is refused while save references, load specifications or lookup futures
remain. Shutdown requires admission to have stopped and Core to have drained save
completions before scheduler close. The worker drains queues before stopping
threads and closing its native handle. A bounded close that cannot drain raises
an error and keeps the registered buffers/handle alive; it does not pretend that
a timeout makes DMA safe to free. The caller must retry close after draining.

Lookup publication follows worker initialization. Lookup RPC send/receive waits
are bounded; correlated relaxed REQ messages cannot accept a previous timed-out
reply as the next lookup result.

## CPU validation snapshot

Validated on Windows Python 3.12 / torch 2.10 CPU with the above local Core tree.
The narrow bootstrap substitutes `uvloop` with `asyncio`, provides build metadata,
and uses the existing `_mock_deps` NPU/native-dependency shim. Core cache managers,
cache specs and recovery algorithms are real. This is not an NPU acceptance run.

Targeted suite: **255 passed** across:

- `test_hash_recovery.py` (existing connector/device-adapter isolation tests);
- `test_recovery_coordinator.py`;
- `test_recovery_scheduler.py`;
- `test_recovery_protocol.py`.

The protocol suite compares 40 seed/EAGLE combinations against upstream group
recovery, including missing keys. Three fixed seeds (0, 42, 1024) compare ordinary
and boundary payload bytes, lookup boundaries and deterministic CPU state replay
outputs. This uses byte-copy Store IO, not model generation or native RDMA.
Additional tests exercise local-tail arbitration, Core boundary handoff, pin
release, rank ordering/duplication, cancellation, failures, safe reset/close,
metadata aggregation and legacy-path isolation.

Compatible legacy regression: **445 passed, 282 subtests passed, 4 deselected**
across connector, coordinator, metadata, pool scheduler/worker, Mooncake layerwise,
pipeline, metrics and transfer tests. The exclusions are the two
`joint_prefix_group_ownership` cases (missing Core API), the independent-DP IPC
binding case (Windows transport), and the wrapped-Mamba post-compute mapping case
(custom-spec registry initialization). They remain Linux/environment gates.

The full Ascend Store directory is **not certified green** in this checkout:
collection encounters newer GLM/Core APIs absent from v0.33.0; Windows ZMQ lacks
IPC transport; the narrow bootstrap does not register all custom hybrid specs.
The existing layerwise mock-layout failure and two missing-Core-API scheduler
failures were reproduced with the three modified production modules loaded
directly from baseline `HEAD`, without changing the working tree. These findings
are not a substitute for Linux/NPU regression.

## Maintenance and remaining gates

Owning the three layers duplicates upstream protocol implementation deliberately.
Future Core boundary fields, cache-manager semantics, job-completion contracts,
key serialization or native Store API changes need an explicit port and renewed
differential tests. This is more maintenance than subclass reuse. It does not make
Core experimental connector interfaces stable or support arbitrary future Core
versions automatically.

Before merge, run Linux/NPU mixed and PD gates using the same model, seed and
target: verify external sub-page hits, common recurrent-state boundaries, restored
model output, stronger local partial-tail preservation, normal whole-block and
layerwise behavior, async failures and task-owned resource cleanup. PD must retain
the native MooncakeConnectorV2 implementation. Resource allocation requires user
confirmation; none is performed by the CPU validation described here.
