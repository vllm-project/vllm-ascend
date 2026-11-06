# Qwen4Exp layerwise cache placement

See the [user guide](../../user_guide/feature_guide/qwen4_layerwise_kv_offload.md)
for startup configuration, supported deployment modes and capacity examples.

## Reuse intent and grouping

KVCacheSpec.reuse requests participation in layerwise reuse. Its default is
False; disabling layerwise leaves the original descriptors unchanged.
The keyword-only field does not participate in spec equality or hashing,
so it does not change logical cache groups. KVCacheGroupSpec.reuse_layer_names
preserves per-name intent when homogeneous specs are merged. This requires
the companion vLLM interface/planning change; a dynamic attribute on a frozen
spec is not a replacement for the declared field.

Qwen4 declares main/compressed QSA and GDN layers from physical layer 2 as
reuse participants when constructing its specs. GDN layers 0 and 1, raw rings,
PLE and hidden state remain resident. Independent/excluded configuration that
also names a reuse=True cache is an error. The planner never changes intent
to resolve a conflict. A small bucket may require no actual aliasing.

## One slot inference rule

build_layerwise_reuse_layout receives the original KVCacheConfig.
Ordinary contiguous regions are bucketed inside each original descriptor;
independent tensors are never merged based on spec equality. Compatibility
includes the complete spec and its concrete type. Simultaneously used
components are assigned separate lanes.

The public planner has no model imports, model-name dispatch or model planning
callbacks. Reuse intent comes exclusively from the expanded specs. The existing
Qwen4KVCacheLayout describes the original storage through ordinary affine
descriptors, with one explicit KVCacheTensor.backing_id for its shared pool.
This optional keyword-only companion vLLM field identifies physical allocation;
equal sizes do not. Unmarked descriptors retain independent ownership for reuse
planning. No owner-description hierarchy or second model-specific plan is added.

Within each backing, the public planner buckets compatible specs and physical
block strides, then derives slots, dependencies and new descriptors itself.
It preserves existing exact cross-group pool aliases when packing rotating
slots; partially overlapping regions and simultaneous same-group aliases are
rejected. Resident banks remain separate from all rotating banks.
Each compatible rotating slot emits one descriptor, including when its layers
span several groups. Group membership and transfer routing stay in
kv_cache_groups and are resolved by layer name. The 48-layer fixture therefore
uses nine descriptors: three for GDN (one resident, two rotating), two each for
main and compressed QSA, one for raw rings and one for PLE.
QSA and GDN keep independent rotation sequences. Their shared state slabs are
permitted by the existing disjoint group block-ID lifetimes, not by coinciding
slot numbers. Unknown backing identity, ambiguous geometry and unsupported
save/restore requests fail before placement is published.

The result contains policy, slot members, resident names and overwrite
predecessors. It does not copy specs, group structures, component views or
transfer regions. LayerwiseKVCacheConfig declares that result for dataclass
IPC. Scheduler spec simplification consumes this result without trying to
recover the original layout from a representative group spec.

## Allocation and capacity

apply_layerwise_kv_cache_plan infers placement, exports descriptors and
validates complete name coverage before updating configuration. No model plan
argument is accepted. Reapplying the same placement is idempotent; changed
capacity, policy, intent or descriptors is rejected.

Qwen4's existing allocator creates its shared backing once and its private PLE/hidden
banks separately. Consecutive resident banks use a nonzero layer_stride, so all
12 raw rings fit in one descriptor while retaining 12 distinct layer views.
Reshape uses offset + layer_index * layer_stride while
preserving K/V kernel-page conversion and GDN conv/SSM strides.

The existing pool-cost and admission hooks probe the same placement at one
block. The current Qwen4 layout has aligned linear block cost, so ordinary
integer division determines capacity. No independent generic capacity solver
or budget-padding reservation is needed. Worker profiling uses the same slot
inference and does not multiply a budget already charged by physical cost.
The public worker does not import model layout functions or classify these
layouts by model name. It queries the existing planning bridge for physical
pool accounting and uses its placement result for the probe. The probe clears
the block-count override on a shallow configuration copy, leaving the engine's
requested capacity intact. Model dispatch stays at the planning boundary.

Cross-rank normalization re-enters the existing config builder at the common
block count. Offsets are regenerated at that count. An explicit override must
fit the physical budget.

## Transfers and lifecycle

Both pool scheduler and pool worker consume the serialized slot result or
invoke the same inference function on complete original configuration.
Transfer addresses come from the actual allocated views. Resident/rotating
classification applies to group/layer transfer regions, not the entire
physical decoder layer.

The current transfer task cannot split mixed resident/rotating components
inside one group/layer. Such requests are rejected explicitly. In particular,
main and compressed QSA must declare compatible intent; neither flag is
silently changed.

Loads wait for every overwrite predecessor, including empty restores.
Completion events remain readable by all successors until both transfer
queues drain at a generation boundary. Partial save/restore failures are
fatal to reuse. Null checkpoint blocks are not restored. Scheduler references
stay pinned until all ranks complete saves, including cancellation paths.
Non-saving TP ranks still fence their local computation.

## Integration and validation

The existing planning patch retains per-name intent and publishes placement
before scheduler simplification. Remove this bridge when upstream supplies
equivalent plugin placement and serialized connector metadata hooks. No new
environment variable is introduced.

Tests cover independent backings, mixed residency, full component coverage,
configuration conflicts, same-byte-size incompatible geometry, multiple
predecessors, real strided CPU copies, serialization, runner allocation and
kernel-page reshape. NPU layout and Memcache roundtrip tests use the same
descriptors. Model output, cancellation, chunked prefill and automatic
capacity require fresh end-to-end validation after changing placement.
