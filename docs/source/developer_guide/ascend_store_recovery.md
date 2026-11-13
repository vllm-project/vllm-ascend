# AscendStore hash-granular recovery

`AscendStoreConnector` uses its existing coordinator, scheduler and worker for
both whole-block and hash-granular recovery. There is no separate recovery
stack, upstream Store subclass, or user-selectable prefix mode. Only Core's
resolved `--prefix-match-unit` and cache plan select the supported hit alignment.
Layerwise transfers retain their existing implementation and physical-page gate.

## Responsibilities and Core contracts

| Existing Ascend owner | Responsibility |
| --- | --- |
| `metadata.PoolCacheGeometry` | Shares Core-resolved hash size and hit alignment, DCP-adjusted group block sizes, pool transfer alignment, retention and EAGLE policy between scheduler and worker. Does not change Core's plan. |
| `coordinator.AscendStoreCoordinator` | Intersects group recoverability through Core's registered cache managers. Saving uses retention; lookup considers all externally available recovery points. |
| `pool_scheduler.KVPoolScheduler` | Accepts Core's final local/external hit arbitration and constructs load/save metadata. A stronger local partial tail is not replaced by a shorter external hit. |
| Scheduler `_prepare_boundary_saves` | Consumes `SchedulerOutput.kv_connector_block_state`: current block tables and exact recurrent `boundary_state_offloads`, never an append-only positional mirror. |
| Scheduler `register_finished_partial_tail` | Pins the explicit final checkpoint before Core frees a finishing request, then publishes it on the next scheduler step. |
| `in_flight_save_blocks.InFlightSaveBlockTracker` | Owns source-block references per save job. Releases only after all ranks acknowledge, deduplicating rank identities and handling out-of-order completion. Reset refuses outstanding saves. |
| `pool_worker.KVPoolWorker` | Performs lookup and physical-page replay through existing backends and threads. Ordinary and boundary values share the group/rank-isolated key namespace and physical payload protocol. |
| `kv_transfer` | Waits for device events, writes boundary values before ordinary pages, and acknowledges successful, failed or cancelled jobs. Failed saves never advance saved offsets. |
| Ascend device adaptation | Owns NPU registration, real stride/segment descriptors and device events. Mooncake setup and engine selection use the existing backend paths; hash recovery does not install a direct-only Store adapter or change native MooncakeConnectorV2. |

## Geometry and configuration

`resolve_pool_cache_geometry` reads `KVCacheConfig.hash_block_size` and
`cache_hit_alignment_tokens`. When either field is unresolved, it calls Core's
`resolve_kv_cache_block_sizes` and `resolve_cache_hit_alignment_tokens` on a
normalized copy of the cache plan. It never treats the requested PMU as a
physical page size or writes a new plan into Core.

The geometry keeps three distinct quantities:

- `block_sizes`: each group's token span, resolved through Core's
  `resolve_dcp_kv_block_size`.
- `transfer_granularity`: the LCM of the prefix-cacheable groups' block sizes.
  This is a common pool alignment, not an allocation size in bytes.
- `core_hit_alignment`: the boundary Core permits for cache hits. When it is
  smaller than the pool transfer alignment, the geometry advertises partial
  hash-hit capability. Layerwise operations retain their existing transfer gates.

The current geometry field named `scheduler_block_size` also stores the pool
transfer alignment. It is not a preserved copy of Core's scheduler size: for
multiple groups, Core takes the LCM across all groups, whereas the pool alignment
only includes prefix-cacheable groups. These values can differ when an excluded
group introduces an additional alignment requirement.

`discard_partial_chunks` is an existing connector extra setting, defaulting to
`true`. It controls incomplete ordinary save chunks; it does not disable
hash-boundary checkpoint saves or sub-page hit lookup. Non-layerwise hash recovery
rejects an explicit value other than `true`. Users do not need to set this option
to enable PMU recovery. There is no `pool_prefix_mode`,
`npu_shared_engine_direct_only`, or `npu_store_setup_lock` implementation, and no
special migration-error check for these old keys.

## Hash hits versus physical replay

Lookup checks hash-unit keys and asks Core's group managers for the common
recoverable token boundary. Attention and recurrent state must both prove that
boundary, including any EAGLE proof margin. Lookup metadata carries each group's
selected tail key to the existing load job.

A fine-grained token boundary does not imply a fine-grained DMA allocation.
Attention restores the complete containing physical page; Mamba restores the
complete checkpoint selected by Core. Actual storage strides and dense payload
segments prevent copying padding or adjacent views. Core resumes computation at
the accepted token boundary.

## Failure and lifetime ordering

1. Core hands off a stable checkpoint and current block IDs.
2. Scheduler pins sources before request teardown or block reuse.
3. Worker waits for device readiness and performs asynchronous IO.
4. Completion, failure or cancellation emits a stopped-job acknowledgement.
5. Scheduler releases references only after all expected ranks acknowledge.

Cancellation does not authorize freeing buffers still used by IO. Pending saves
keep Core stepping; reset is rejected until acknowledgements are collected.
Worker shutdown joins threads before closing the backend. Cancelled queued saves
produce failed acknowledgements and complete fences. A native IO timeout is not
treated as proof that DMA has stopped.

## Validation and maintenance

CPU tests cover byte-copy boundary save/replay, local-tail arbitration, exact
Core handoffs, pin release, failure/cancellation, duplicate and out-of-order
acknowledgements, and existing bulk/layerwise paths. Differential tests use
upstream Store only as an oracle; production code does not import its three
Store classes.

CPU tests do not establish native DMA or model accuracy. NPU acceptance must
separately demonstrate mixed TP4 and native MooncakeConnectorV2 P/D TP2+TP2:
cold reference, seed, local-only APC reset, and recovery with a non-page-aligned
external hash hit, zero local-hit increment and matching cold-reference output.
Each run retains requests, responses, metrics, versions and source hashes.

Core's connector, boundary-state and cache-manager contracts remain dependencies.
Their evolution requires renewed protocol and differential regression even though
Store classes are not inherited.
