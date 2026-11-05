# Qwen4Exp model-aware layerwise cache plan

## Planning contract

`core/layerwise_cache_plan.py` defines immutable policy, owner, backing, view,
transfer-region, alias and dependency records. Plans contain no tensors,
device events or process-local handles. The pure planner solves the exact
cost function, including fixed costs and alignment, and validates overrides.
Duplicate backing IDs count once only if their definitions agree.

`models/layerwise.py` is the adapter selection boundary. The only production
adapter is `models/qwen4_exp/layerwise.py`; unrelated models continue through
their existing paths. A test-only adapter exercises non-linear capacity,
multiple backing sizes, joined dependencies and fanout without registering
another model.

Configuration carries a declared `LayerwiseKVCacheConfig.layerwise_summary`
field containing the normalized policy, adapter version, original budget,
group signature, owners, dependencies and physical-plan signature. It survives
deepcopy, scheduler group simplification and worker MessageQueue transport.
Worker materialization validates the signature, capacity, policy and exported
descriptors. Negative descriptor strides identify non-affine model layouts
and must never reach a generic allocator without this summary.

## Qwen4 physical layout

For 48 layers, independent `[0, 1]` and two shared slots, the state pool has
four slabs: independent 0, independent 1, A and B. QSA main and GDN share these
slabs only across groups with disjoint physical block IDs. Within a group,
successive owners of the same slot have an explicit save dependency.

QSA layers `3, 11, ...` use A and `7, 15, ...` use B. GDN rotates across
group boundaries (`2:A, 4:B, 5:A, 6:B, 8:A, ...`). The two compressed-key slots
follow their main-QSA owners. Twelve raw ring regions and PLE remain private.
PLE retains its own contiguous backing. GDN conv/SSM preserve their original
strides; QSA kernel-page conversion remains model-specific.

The allocator allocates each backing once. Views and transfer regions retain
their logical owner names. Existing pool registration derives segmented
addresses from these views, so transfers select only the reachable blocks
and valid component bytes. Null checkpoint block IDs are never restored.

## Runtime ordering

The current QSA and GDN hooks remain in place. A physical layer completes
only after every transfer component for that layer has saved successfully.
Main/compressed and conv/SSM therefore share the existing layer completion
fence. PLE and layer-1 GDN retain their existing single-cursor hook ordering.

`LayerLoadTask` accepts multiple predecessors and a generation number.
Legacy tasks retain the single-predecessor behavior. Model tasks wait for
every predecessor's CPU notification, failure status, NPU event and external
reader release. An empty restore still executes this dependency fence.
Completion events are not cleared by the first waiter: they remain valid for
all successors in the generation.

At the next metadata bind, both transfer queues drain before events are
recycled. The final-layer path drains all saves and loads, synchronizes the
last compute event, and publishes block-retention completion. Every rank,
including a non-saving TP rank, performs the local compute fence. The existing
TP barrier prevents a reader from querying another rank's unfinished history.

The scheduler retains source block references for all groups until all ranks
report completion. Cancellation of the request does not release these retained
references early. Failed saves and loads propagate as fatal transfer errors;
waiters wake but cannot overwrite the source region.

## Upstream integration and removal

The existing KV-cache planning patch routes Qwen4 admission and configuration
through the same physical cost. Its `get_kv_cache_configs` wrapper rebuilds
offsets after upstream minimum-capacity normalization, using each rank's
original budget. Runner V2 materializes the selected plan before applying
layerwise layout; allocation and reshape consume that fixed plan.
Worker profiling recognizes model adapters before group planning and preserves
the physical budget. It skips both the legacy attention-only reuse planner and
the logical-layer budget multiplier, which would count the saving twice.
Kernel-page selection validates the model summary before treating a negative
descriptor stride as a non-affine view. QSA therefore retains its backend's
128-token pages and compressed 32-state pages inside each 768-token manager
block; Mamba groups retain their manager block size.

Upstream descriptors currently describe affine layer storage and do not carry
model-specific non-affine layout metadata through normalization. These hooks
can be removed when upstream exposes model-aware capacity, configuration
serialization, post-normalization rebuilding and allocation/reshape extension
points with equivalent contracts. No new environment variable is introduced.

## Tests

- `tests/ut/core/test_layerwise_cache_plan.py` and
  `test_layerwise_cache_planner.py`: exact costs, bounds, alias validation,
  dependencies and normalized policies.
- `tests/ut/models/qwen4_exp/test_layerwise.py`: exact 3,648-block fixture,
  allocation, views, mismatch rejection, admission, normalization and real IPC.
- `tests/ut/distributed/ascend_store/test_layerwise_model_plan.py`: generation
  fences, failure propagation, external reads, source references and real CPU
  segmented-copy addresses.
- `tests/e2e/pull_request/one_card/model_runner_v2/test_qwen4_layerwise_layout.py`:
  actual NPU writes/restores with stable view addresses across three steps,
  plus real Memcache segmented-region roundtrips when an existing service is
  configured through `MMC_LOCAL_CONFIG_PATH`.

The NPU layout test verifies physical storage semantics; it is not a substitute
for model-output, chunked-prefill, concurrency or graph-replay validation.
