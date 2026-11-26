# MRv2 EPLB Deployment Export and Static Initialization

MRv2 can reuse a deployment learned by dynamic EPLB. The deployment file
contains installed physical-to-logical expert mappings, not a planner's
uncommitted target and not expert weights.

The implementation has two reviewable parts:

1. Versioned deployment files and a collective, offline export operation.
2. One-time installation during MRv2 model initialization, followed by fixed
   routing without runtime collection, planning, or migration.

A [Chinese code walkthrough](mrv2_eplb_placement_walkthrough_zh.md) explains
the implementation and provides a suggested learning order.
The [local validation record](mrv2_eplb_placement_validation.md) distinguishes
executed checks from the supported-environment and NPU checks still required.

## Deployment representation

`vllm_ascend/distributed/eplb/placement.py` has no torch or NPU dependency.
Its `EplbPlacement` contains:

- `schema_version`: version `2` for a target-only deployment and version `3`
  when a registered speculative draft placement is included; unknown versions
  fail.
- `metadata`: model identity/revision, architecture, HF configuration digest,
  quantization, dtype, EP/TP/DP/PP sizes, expert counts, and ordered module paths.
- `physical_to_logical`: one flattened global physical-slot map per target MoE
  layer. Version 3 files also contain `draft.physical_to_logical` and its own
  draft metadata.
- `provenance`: the calibration workload, policy, window, interval, and phase.

For EP size `N` and `S` slots per rank, physical slot `r * S + s` belongs to
rank `r`, local slot `s`. The value at this position identifies its logical
expert. For example, `[0, 1, 2, 0, 1, 3]` with EP=2 has expert replicas 0
and 1 on both ranks, and one replica each of experts 2 and 3.

Validation requires equal rank capacity, complete logical-expert coverage,
valid integer IDs, no rank-local duplicate experts, and exact runtime metadata
matching. Derived logical maps and replica-routing tables are rebuilt instead
of serialized. Metadata deliberately excludes HF's local `_name_or_path` and
`transformers_version` when hashing structure, but the configured model
identifier and revision must still match. The format does not authenticate
checkpoint contents; callers must use the same checkpoint when a model name
or local directory can be overwritten.

## Export transaction

The supported entry point is an offline all-worker RPC, after generation has
returned and before another generation is submitted:

```python
results = llm.collective_rpc(
    "export_eplb_placement",
    args=("/shared/eplb/deployment.json", "dataset, lengths, concurrency, rounds"),
)
```

Do not invoke this RPC independently on one EP rank. For externally managed
DP engines, every participating engine must issue it concurrently; the
provided example uses one offline LLM with TP/EP and DP=1.

Worker execution serializes the RPC between forwards. The export operation
does not advance `step()` and therefore cannot publish a new rearrangement.
`finish_pending_commits()` instead polls the existing async result-readiness
protocol and invokes the patched upstream `_move_to_workspace` for each
published result. That existing boundary installs compute weights, commits
maps, refreshes routing in place, and only then acknowledges the receive
workspace. It also handles the no-change result of a planner cycle.

`drain_async()` is intentionally not reused: its shutdown/elastic cancellation
contract consumes pending results without installing them.

After commits and foreground device work finish, each rank copies the
installed physical map to CPU. All ranks validate and compare canonical
SHA-256 digests, including calibration metadata. EP rank 0 writes a complete
temporary file in the destination directory and atomically replaces the
destination. Validation and write failures are shared across ranks before
returning. Export is not a hot-path operation; host synchronization is
intentional here.

Export guarantees an installed, consistent deployment. It does not guarantee
that calibration has converged or that a rearrangement changed any experts.
Use representative traffic and inspect dynamic-cycle/performance evidence.

## Static initialization

Keep `--enable-eplb` enabled: it controls the router and expert allocation,
which static execution still needs. The Ascend `mode` separately controls
the lifetime of dynamic activity.

The controller reuses the upstream model-registration and load hooks. In its
static `maybe_start_async_loop` branch it installs the file instead of
starting a worker. This happens before memory profiling and graph capture.

The first implementation intentionally uses existing checkpoint loading and
then performs one exact initialization-time deployment:

1. Load the checkpoint in the existing initial physical layout and expose the
   quantization method's actual compute-weight views.
2. Read and validate each rank's deployment file; require identical digests
   and runtime metadata before backend registration or movement.
3. Snapshot and verify the old layout, then build an exact source coordinate
   for each target slot. Prefer unchanged slots, then rank-local copies.
4. Initialize the existing HIXL session if needed, without a background thread.
5. Stage one changed layer into the existing expert buffer through
   `stage_explicit_layer_transfer`.
6. Synchronize and reach an EP CPU barrier before overwriting live sources.
7. Install buffered weights using upstream `move_from_buffer`, rebuild and
   copy the logical maps/replica counts, and refresh routing in place.
8. Synchronize before reusing buffers for the next layer, then close temporary
   HIXL registrations after the complete deployment is installed.

This sequence also supports within-rank swaps: both old sources are staged
before either live destination is overwritten. Directly calling
`update_mapping` or `from_mapping` cannot replace these weight transfers.

Quantization metadata moves with the exact weight views already supplied by
the quantization method. No controller-side weight-layout knowledge is added.
Initialization retains upstream compatibility load/workspace buffers in this
version; static mode is a runtime-overhead reduction, not a claim of minimum
memory consumption. Direct checkpoint-to-target loading is future work.

During serving:

- controller `prepare_forward` and `step` return without dynamic work;
- state `step`, profile rearrangement and async startup are guarded;
- ordinary forwards keep each layer's `load_recording_enabled=False`, omitting
  both token recording and AllGather eligibility/counting from graph capture;
  a scheduled observation forward enables one sampled recording step and then
  disables it again;
- logical-to-physical mapping remains active.

## Initial support boundary

Both export and static consumption require PP=1. Static mode supports MTP
when the draft configuration disables EPLB and redundant draft experts. When
a registered MoE draft enables EPLB, dynamic export includes its committed
placement and static startup installs both the target and draft placements
before graph capture.
One registered target MoE model is required. EP/TP/DP sizes, model identity,
revision, structure, quantization, dtype, expert counts and layer order must
match. Every static rank must be able to read the file at startup.

Existing MRv2 quantization/hardware constraints, including A2 redundancy
restrictions, still apply. Static installation uses the same communicator and
weight movement implementation as dynamic EPLB, including its NZ/Gloo limits.
There is no runtime fallback after a partially failed migration. Auto EPLB,
elastic EP and cross-topology deployment conversion are outside this change.

## Validation and review split

CPU tests cover the schema, mismatch rejection, source planning, atomic-file
failure, collective errors/consensus, local swaps, remote-source installation,
Tensor List views, commit completion, routing-table identity and static guards.

The four-card NPU test `test_mrv2_eplb_placement.py` runs dynamic calibration,
exports collectively, restarts statically, compares the deployment and selected
compute/scale samples, verifies answers, and checks that deployment/weights
remain fixed with no recording or migration worker.

Before merge, run it on real Ascend hardware for both supported transports,
exercise a deployment with changed layers, and compare disabled/static/dynamic
throughput, TTFT, TPOT, tail latency, startup time and memory. CPU transport
stubs do not establish HIXL/HCCL/NZ correctness. Follow the repository's
[PR testing requirements](../contribution/testing.md#pr-testing-requirements).

For two PRs, review schema/export/RPC/example tests first; then review
configuration/static installation/guards/restart tests. Release-note draft
templates are in `release_notes/eplb-placement-export.yaml.in` and
`release_notes/eplb-static.yaml.in`. Replace the placeholder with the actual
PR number and place each final fragment in `release_notes/fragments/` when
the corresponding PR is opened.
