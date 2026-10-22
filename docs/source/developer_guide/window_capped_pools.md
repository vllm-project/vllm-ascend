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

Validated on the current `main` base (`a9958712`, pinned vLLM `a98247ab`)
in the Ascend container:

- RED: the old implementation with the new backpressure cases fails 3/3 —
  both group shortfalls raise
  `ValueError: Cannot get 2 free blocks from the pool` mid-allocation, and
  the customer-incident case shows aggregate usage 0.667 against full-group
  usage 0.933.
- GREEN: pools + admission + backpressure suites 48/48.
- Regressions: `tests/ut/patch/platform` 438 passed; `tests/ut/core` +
  `kv_transfer` + `patch/worker` 510 passed; the full `tests/ut` failure and
  error set is identical to the baseline tree (zero new failures).
- Lint: ruff 0.14.0 (pre-commit pinned) check + format clean.
- NPU (4-card TP, gemma-4-31B, per-group pools `[1460×5, 2560]`, primary =
  full group): a 32×24K long-dwell wave drove the full pool to 97.5% usage
  with `waiting_by_reason{capacity}` draining 27 → 0; 32/32 requests
  completed with zero allocation errors, zero 5xx and zero preemptions.
