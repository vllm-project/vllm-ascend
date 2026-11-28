# EPLB Deployment Import and Static Initialization

Ascend EPLB can reuse a deployment learned by an earlier calibration run. The
deployment document contains installed physical-to-logical expert mappings, not
a planner's uncommitted target and not expert weights. This change implements
the consumer side: static initialization from a versioned document.

## Deployment representation

`vllm_ascend/distributed/eplb/placement.py` has no torch or NPU dependency.
Its `EplbPlacement` contains:

- `schema_version`: currently `3`; versions `2` and `3` are readable. The
  optional `draft` entry carries a registered speculative draft placement.
- `metadata`: model identity/revision, architecture, HF configuration digest,
  quantization, dtype, EP/TP/DP/PP sizes, expert counts, and ordered module
  paths.
- `physical_to_logical`: one flattened global physical-slot map per target MoE
  layer. A deployment that also covers a registered draft contains
  `draft.physical_to_logical` and its own draft metadata.
- `provenance`: an informational object describing how the document was
  produced. It is validated as canonical JSON but is not used for routing.

For EP size `N` and `S` slots per rank, physical slot `r * S + s` belongs to
rank `r`, local slot `s`. The value at this position identifies its logical
expert. For example, `[0, 1, 2, 0, 1, 3]` with EP=2 has expert replicas 0 and
1 on both ranks, and one replica each of experts 2 and 3.

Validation requires equal rank capacity, complete logical-expert coverage,
valid integer IDs, and exact runtime metadata matching. Derived logical maps
and replica-routing tables are rebuilt instead of serialized. Metadata
deliberately excludes HF's local `_name_or_path` and `transformers_version`
when hashing structure, but the configured model identifier and revision must
still match. The format does not authenticate checkpoint contents; callers
must use the same checkpoint when a model name or local directory can be
overwritten.

## Import transaction

`mode=static` keeps `--enable-eplb` enabled because it controls the router and
expert allocation. The controller installs the document instead of starting
the dynamic EPLB worker. Installation happens before memory profiling and
graph capture:

1. Load the checkpoint in the existing initial physical layout and expose the
   quantization method's actual compute-weight views.
2. Read the document on EP rank 0, broadcast it to every rank, and validate it
   against each rank's runtime metadata before backend registration or
   movement.
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
memory consumption.

## Static serving

During serving:

- controller `prepare_forward` and `step` return without dynamic work;
- state `step`, profile rearrangement and async startup are guarded;
- ordinary forwards keep each layer's `load_recording_enabled=False`, omitting
  both token recording and AllGather eligibility/counting from graph capture;
  a scheduled observation forward enables one sampled recording step and then
  disables it again;
- logical-to-physical mapping remains active.

`eplb_config.log_balancedness=true` opts into a rank-local load sample so that
data-parallel prefill engines do not enter a device collective from the hot
path. It never triggers planning or migration.

## Support boundary

Static import requires PP=1. A registered MoE draft that enables EPLB is
installed from the same document. One registered target MoE model is required.
EP/TP/DP sizes, model identity, revision, structure, quantization, dtype,
expert counts and layer order must match. EP rank 0 must be able to read the
file at startup; the document is broadcast to every rank.

Existing Ascend EPLB quantization/hardware constraints, including A2 redundancy
restrictions, still apply. Static installation uses the same communicator and
weight movement implementation as dynamic EPLB, including its NZ/Gloo limits.
There is no runtime fallback after a partially failed migration.

The deployment document must be produced by a compatible external
calibration tool; this change only consumes it. Export, auto EPLB, elastic EP
and cross-topology deployment conversion are outside this change.

## Validation

CPU tests cover the schema, mismatch rejection, source planning, collective
errors/consensus, local swaps, remote-source installation, Tensor List views,
routing-table identity and static guards.

Before merge, exercise a document with changed layers on real Ascend hardware
for both supported transports, and compare disabled/static/dynamic throughput,
TTFT, TPOT, tail latency, startup time and memory. CPU transport stubs do not
establish HIXL/HCCL/NZ correctness. Follow the repository's
[PR testing requirements](../contribution/testing.md#pr-testing-requirements).
