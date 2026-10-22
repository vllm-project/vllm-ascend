# Window-capped KV pools: full-attention capacity and group-local admission

Window-capped pools are an Ascend per-layer-layout optimization for hybrid
full-attention and sliding-window models. Each KV group has a private backing
buffer and an independent block-ID space. A block free in one group cannot be
allocated to another group.

## Capacity contract

The first resolved full-attention group is the primary pool. It is selected
from the KV spec, including uniform-spec wrappers, not from group order,
maximum block count, hardware name or model name.

- `KVCacheConfig.num_blocks`, `PerGroupBlockPools.num_gpu_blocks`, scalar free
  capacity and scalar remaining-request demand use primary Full-pool units.
- `group_num_blocks` remains the authoritative capacity for each physical
  buffer and connector registration. Equal values still describe independent
  pools when that metadata is present.
- The KV usage gauge reports the highest usage among full-attention groups,
  using each real pool's null-block-adjusted usage. It is a capacity-pressure
  metric, not total HBM byte occupancy. Per-block residency collection is
  attached only to the primary pool because its block IDs are group-local.

## Admission contract

The coordinator obtains a demand vector from the existing single-type managers.
It does not estimate demand using only token counts, and it does not turn a
shortfall into a synthetic demand value. Prediction remains independent of the
current free capacity, including for future in-flight reservations.

Both existing `KVCacheManager.allocate_slots` admission gates check each pool:

```text
full-sequence gate:
    demand[group] + primary_watermark[group] <= free[group]

step gate, after upstream skipped-window reclamation:
    demand[group] + inflight_reservation[group]
        + primary_watermark[group] <= free[group]
```

The configured watermark applies only to the primary pool, with upstream's
existing waiting/preempted and already-scheduled-request conditions. It is not
subtracted as the same absolute block count from every smaller sliding pool.

The full-sequence check retains `apply_admission_cap=True`. The step check
retains uncapped allocation prediction and the original processed-token basis
for sliding-window reclamation. Prefix-hit touches, external KV allocations,
partial-hit copy-on-write and new slot allocation happen only after all groups
pass. A shortfall returns `None`, allowing existing scheduler waiting/preemption
logic to run; it is not caught after partial allocation.

Async KV-load reservations are carried as an immutable `GroupBlockReservation`,
computed by summing each in-flight request's remaining demand elementwise. The
shared-pool path retains integer reservations. Independent pools accept scalar
zero for callers without reservations; a nonzero scalar is ambiguous and is
rejected explicitly. Custom schedulers must pass the typed group reservation.

## Patch composition and compatibility

The Ascend-local patch replaces only the two upstream demand acquisitions and
two admission predicates. It validates source anchors before installing the
method, instead of copying the scheduler or silently accepting a changed core
interface. An upstream source change that invalidates the anchors fails with
an explicit installation error and requires review.

Source-derived patches retain their generated source in `_ascend_source`, so
other source-based rewrites installed before or after the admission patch
compose from the same body instead of unwrapping it. Upstream removed the
legacy SWA bounded-replay backport (its replay guard now ships in the pinned
vLLM core), and the admission rewrite still detects that guard via its source
anchors; the regression suite models a foreign source rewrite to pin the
composition contract in either installation order. Generated code is never
written into the upstream checkout.

The independent coordinator is used even when prefix caching is disabled.
Without `group_num_blocks`, existing shared-pool accounting remains unchanged.
Only attention-only layouts with both full and sliding groups are selected by
the window-capped planner.

Set `VLLM_ASCEND_DISABLE_WINDOW_CAPPED_POOLS=1` to select stock planning. This
must be set before engine startup; it is not an online capacity migration.
AscendStoreConnector additionally needs the per-group tensor-registration
support in [companion PR #19249](https://github.com/vllm-project/vllm-ascend/pull/19249).
That connector fix does not replace scheduler admission checks.

## Validation status

This revision's tests were written under a user-requested baseline freeze.
No tests, lint, compilation, server restarts or benchmarks may run until the
`issule-fix` session completes its baseline and explicitly hands off the
validation environment. Passing historical Plan B benchmarks are not evidence
that these new admission changes have been verified.

After handoff, the required checks include full/sliding exhaustion below 75%
aggregate usage, exact fits, reservations, watermark conditions, reclaimed-window
capacity, prefix touch, external KV, CoW, shared-pool compatibility, prefix-cache
disabled mode and replay-patch installation order. Run the regression against
the old implementation before asserting that it detects the original defect.
