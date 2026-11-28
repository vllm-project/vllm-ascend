# EPLB Deployment Export

Ascend EPLB can export the deployment its dynamic mode has installed, so the
calibration result can be inspected, diffed, or shipped to another
deployment. The deployment file contains installed physical-to-logical
expert mappings, not a planner's uncommitted target and not expert weights.

The implementation is one part: versioned deployment files and a
collective, between-forwards export operation that one configuration knob
triggers.

## Deployment representation

`vllm_ascend/distributed/eplb/placement.py` has no torch or NPU dependency.
Its `EplbPlacement` contains:

- `schema_version`: currently `3`; unknown versions fail. The optional `draft`
  entry carries a registered speculative draft placement.
- `metadata`: model identity/revision, architecture, HF configuration digest,
  quantization, dtype, EP/TP/DP/PP sizes, expert counts, and ordered module paths.
- `physical_to_logical`: one flattened global physical-slot map per target MoE
  layer. A deployment that also covers a registered draft contains
  `draft.physical_to_logical` and its own draft metadata.
- `provenance`: the derived deployment description, policy, window, interval,
  and phase. The description names the mode, placement scope and collection
  phase, so it stays truthful without a user-supplied workload string.

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

Export is configuration-driven, not an RPC. Setting `expert_placement_path`
together with `export_after_rearranges` makes the engine export once, in the
first forward gap after that many completed rearrange cycles, and keep serving:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-expert-parallel --enable-eplb \
  --additional-config '{"eplb_config":{"expert_placement_path":"/shared/eplb/deployment.json","export_after_rearranges":5}}'
```

`examples/eplb/export_placement.py` drives the same mechanism from an offline
calibration run: it generates representative prompts and waits for the engine
to write the file. A cycle that publishes no statistics because no rank
recorded fresh load does not count towards the limit, so a configured number of
cycles always corresponds to that many real rearrangement attempts.

The export runs between forwards, inside the controller's forward preparation,
and does not advance `step()`; it therefore cannot publish a new
rearrangement.
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

## Initial support boundary

Export requires PP=1. One registered target MoE model is required. EP/TP/DP
sizes, model identity, revision, structure, quantization, dtype, expert counts
and layer order must match between the exporting engine and the file. A
registered MoE draft that enables EPLB is exported together with the target,
so one file describes both models.

Existing Ascend EPLB quantization/hardware constraints, including A2 redundancy
restrictions, still apply. Auto EPLB, elastic EP and cross-topology deployment
conversion are outside this change.

## Validation and review split

CPU tests cover the schema, mismatch rejection, atomic-file failure, collective
errors/consensus, commit completion and the one-shot export trigger.

Before merge, run an export on real Ascend hardware for both supported
transports and exercise a deployment with changed layers. CPU transport stubs
do not establish HIXL/HCCL/NZ correctness. Follow the repository's
[PR testing requirements](../contribution/testing.md#pr-testing-requirements).
