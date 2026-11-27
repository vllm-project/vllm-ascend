# Ascend Sleep Mode: upstream communicator lifecycle alignment

## Scope and current implementation

The public entry point is `model_config.enable_nccl_comm_suspend`. The NPU worker
uses the upstream `suspend_device_comms()` / `resume_device_comms()` interface,
with `NPUCommunicator.suspend()` / `resume()` implementing the device contract.

The current backend is **HCCL destroy/restore with ACL Graph recapture only for
cycles that rebuild HCCL PGs**. This is
an interface-first transition, not native lightweight suspension or in-place
reuse of graphs across communication-domain destruction. Native
`HcclCommSuspend` probes remain experimental and are outside production sleep.
The reported zero-HBM-reclamation and unreliable multi-rank resume observations
apply to the tested software stacks; symbol availability alone is insufficient
for enabling a native backend.

## Ownership and object lifetime

- `CaMemAllocator` continues to own tagged weights/KV allocations.
- `GroupCoordinatorPatch` owns PG topology and the binding to its communicator.
- Existing `HcclPgRegistry` owns shared PG refcounts and final native release.
- `HcclSleepAdapter` delegates to those existing PG operations. It survives
  native teardown; a failed native operation prevents automatic retry.
- `NPUCommunicator` remains the same Python object across sleep/wake and binds
  to the restored PG. Final shutdown is distinct from temporary suspension.
- `SleepWakeupManager` only orders graph, communication and MoE lifecycle work.
  It does not maintain a second business-group destroy/restore implementation.

### Temporary traversal bridge

The fixed upstream main SHA is
`544f182d9f0c9bfe8bd7645525951d5e3692a542`; the matched release is `v0.32.0`.
Both expose communicator sleep traversal, but skip groups whose
`device_communicator` is `None`. Ascend still needs to reclaim PG-only world and
single-rank group allocations on multi-rank workers.

`patch_distributed.py` therefore adapts **only the two public sleep traversal
functions** to include the same per-owner adapter for those groups, in registry
order. It excludes the lifecycle anchor. It does not modify the general
`_apply_to_device_comms` helper used for checkpointing, enable a data
communicator on control groups, or add another physical PG registry.

Remove this bridge once upstream can discover PG-only lifecycle participants.
It is a compatibility extension to upstream traversal, not a claim that the
unmodified upstream walker already covers all Ascend PG owners.

## Configuration

Enable `enable_sleep_mode=True`, `enable_nccl_comm_suspend=True` and
`additional_config={"weight_nz_mode": 0}`. This does not require RL mode.

The released `rl_config.enabled && rl_config.sleep_mode_extra_cleanup` opt-in
maps to the same upstream flag with a deprecation warning. Enabling both paths
executes one lifecycle. Other RL behavior is unchanged. The three unpublished
policy keys from the earlier PR implementation have been removed; there is no
independent graph/HCCL policy matrix.

## Ordering and synchronization

Sleep:

1. Drain pipeline send work and synchronize the NPU.
2. Check owner capabilities and agree on group topology/order and live PG
   availability over the CPU control group. The preflight reports whether
   any non-anchor HCCL PG will be rebuilt. Initialize the small HCCL lifecycle
   anchor physically only when there is HCCL work.
3. Invalidate ACL graph entries, attention/draft workspaces and runner graph
   manager state only when graphs are enabled and the cycle will rebuild HCCL.
4. Prepare MoE HCCL state and agree on readiness before the first PG release.
5. Invoke communicator suspension using the destroy/restore adapter.
6. Keep existing transfer-buffer unregistration and allocator sleep behavior.

Wake:

1. Validate staged tags, restore allocator memory, parameters and buffers.
2. Restore all required PGs through communicator resume.
3. Refresh MoE dispatcher HCCL metadata, then runtime contexts. Runtime refresh
   can issue collectives, so it must follow restoration of all required groups.
4. Recapture invalidated graphs after weights and KV cache are ready.
5. Synchronize on the live anchor and release it after all ranks finish recovery.

Never issue a barrier on suspended, destroyed or not-yet-restored business PGs.
Never attempt native Suspend followed by destroy as a fallback. CPU/Gloo
preflight collectives retain their existing distributed timeout behavior.

For a single-rank worker, uninitialized distributed backend, or a multi-rank
plan with no non-anchor HCCL owners, communication cleanup is a no-op. Leave
graph entries, workspaces, events and runner graph-manager state intact, and do
not call capture_model on wake. This decision is recomputed every cycle.
Invalidation must still happen before a planned teardown, rather than waiting
until PG destruction has already occurred; a failed teardown fails the cycle.
A registered HCCL owner unexpectedly missing its device PG is an inconsistent
state: preflight rejects it rather than treating it as a healthy no-op.

## Staged wake, idempotence and failure

The supported staged order is `weights` wake, external level-2 weight reload,
then `kv_cache` wake. The final stage recaptures graphs only if HCCL teardown
invalidated them, and releases the anchor when present.
A full wake restores both. KV-first wake is rejected before allocator restore.
Only the external weight loader can establish that newly supplied weight data
is complete; callers must finish loading before requesting KV wake.

Communicator suspend/resume is idempotent; repeated manager wake does not
recapture again. Sleep preserves CPU groups and communicator shells. Final
shutdown closes the adapter and releases outstanding resources without waking
sleeping PGs or decrementing released references a second time.

Native release/restore, MoE or capture errors propagate to the worker caller.
A partially failed cycle requires worker restart. There is no promised generic
rollback/cold rebuild transaction, and failures are never converted into a
successful wake. Distributed peer failures can still require the existing
process-group timeout / engine failure handling.

## Verification and future work

CPU tests cover graph invalidation only for HCCL rebuild cycles, graph/workspace
preservation on no-op cycles, ordering, staged wake, repeated cycles,
error propagation, topology preflight, PG sharing, communicator identity,
PG-only owners, shutdown while suspended and unchanged checkpoint traversal.
The raw probe tests separately retain handle-conversion and exact-handle
fallback checks.

NPU validation of this refactor remains pending: TP1/TP2, dense/MoE, graph/eager,
L1/L2, repeated cycles, HBM use and wake latency. Historical runs of the earlier
implementation are not validation of this refactor.

A future native backend needs demonstrated HBM reclamation, reliable multi-rank
collectives and generation, repeated-cycle stability, and graph dependency
rebinding before enabling in-place replay. No such capability is inferred from
CANN version numbers alone.

Delete the legacy configuration alias after its deprecation window. Removing
all Ascend ordering hooks additionally requires upstream graph/MoE lifecycle
extension points; a communicator walker alone is not sufficient.

## Integration baseline

This local working tree is based on review-sleep-pr20481-local at c5e62799f.
It has not been rebased onto PR #22421 or a freshly fetched upstream main.
The push workflow must first verify the intended PR tip, port the topic changes
onto that baseline, preserve newer communicator/quantized-all-reduce behavior,
and rerun the applicable checks. Do not replace a newer branch with this older
history or use an orphan force-push.
