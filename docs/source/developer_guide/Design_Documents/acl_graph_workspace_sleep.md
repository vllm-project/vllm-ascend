# ACL Graph Workspace Lifecycle in Sleep Mode

## Problem

Sleep Mode returns NPU memory to a colocated training job without exiting the
inference process. When ACL Graph is enabled, the captured graphs own device
memory that is not covered by the weight/KV-cache sleep pools:

- attention workspaces used by FIA / MLA / SFA / DSA graph replay;
- captured graph device memory and the pool segments backing it;
- breakable piecewise graphs and multimodal encoder budget graphs;
- speculative-decoding graph managers (MTP / EAGLE / DFlash / DSparK).

The default sleep path releases only the CaMem-managed weights/KV cache. The
opt-in `rl_config.sleep_mode_extra_cleanup` path releases ACL graph state too,
but it must cover every graph owner and survive repeated sleep/wake cycles.
Releasing only the Python workspace handles is not enough: the workspace
tensors are weak references into the graph pool, so the pool segment is freed
only when the graph object itself is dropped.

## Ownership and release boundary

| Resource | Owner | Rebuildable | Action on sleep |
|---|---|---|---|
| `manager.graphs[desc]` | target / speculator / encoder manager | yes | drop |
| `ACLGraphWrapper.concrete_aclgraph_entries` | piecewise wrapper registry | yes | drop |
| `BreakableCUDAGraphWrapper` instances | class-level `_all_instances` | yes | `clear_all_graphs()` |
| `GraphParams.events/handles/attn_params` | module-level `GraphParams` | yes | clear list values |
| `GraphParams.workspaces` | module-level `GraphParams` (weak handles) | yes | set to `None` after graphs are dropped |
| `GraphParams` capture-size keys | module-level `GraphParams` | **no** | keep |
| `_capture_descs` | graph manager | **no** | keep |
| `update_stream`, input/output buffers | runner / manager | **no** | keep |
| graph pool handle | platform singleton | yes | keep (avoid churn) |
| HCCL groups / lifecycle anchor | `HcclSleepWakeupManager` | yes | handled by the HCCL policy |

The ordering is important:

1. drop all graph objects (target, speculator, encoder, breakable, piecewise);
2. clear replay metadata (`events` / `handles` / `attn_params`);
3. clear weak workspace handles.

Dropping the graphs first guarantees that a replay cannot observe a live graph
whose workspace was already set to `None`. The weak handles are not
dereferenced while clearing, so dropping the graph first is safe.

## Multiple capture sizes

`GraphParams` stores every field in a dict keyed by the actual capture token
count. Under `FULL_DECODE_ONLY` with speculative decoding, the key is the
rounded token count from the capture descriptors, not the raw
`cudagraph_capture_sizes` value.

- Sleep clears values per key and keeps the key set unchanged.
- Wake re-captures with the same `_capture_descs`; attention backends see a
  `None` workspace and allocate a new one for that key.
- Updating a workspace for an unknown key is a programming error and raises an
  actionable `RuntimeError` instead of a bare `KeyError`.

## Sleep / wake state

`AclGraphSleepWakeupManager` tracks `_needs_recapture`:

- `sleep()` is a no-op when already released; otherwise it drops graphs,
  metadata and weak handles, then sets the flag.
- `wakeup()` is a no-op when the flag is unset, which prevents an accidental
  `capture_model()` when `wake_up()` is called without a preceding `sleep()`.
- Staged level-2 wakeup (`tags=["weights"]`) does not recapture; the KV-cache
  wake (`tags=["kv_cache"]` or `None`) performs the recapture.
- If recapture fails, partially captured graphs are dropped and the flag stays
  set, so a later wakeup can retry instead of hitting
  `assert desc not in self.graphs`.

`SleepWakeupManager` tracks `_asleep` at the HCCL + ACL graph level so a
duplicate `sleep()` does not redo HCCL teardown and a `wake_up()` without a
preceding `sleep()` does not touch HCCL state.

## Graph mode coverage

| Mode | Owner | Released by |
|---|---|---|
| `FULL` | `ModelAclGraphManager` / `ACLGraphWrapper` | manager `graphs.clear()` / wrapper entries |
| `FULL_DECODE_ONLY` | same | same |
| `PIECEWISE` (compiled) | `ACLGraphWrapper` | wrapper entries |
| `PIECEWISE` (breakable) | `BreakableCUDAGraphWrapper` registry | `clear_all_graphs()` |
| MM encoder budget graphs | `EncoderAclGraphManager` | `clear()` |
| Speculative decoding | speculator `cudagraph_manager` / `prefill_` / `decode_` / `query_` | manager `graphs.clear()` |

## Tests

Unit tests cover:

- all target/speculator/encoder managers are released;
- breakable registry is cleared;
- capture-size keys survive and accept new workspaces after wake;
- duplicate `sleep()` and `wakeup()` without sleep are no-ops;
- recapture failure keeps the retry flag and cleans partial graphs;
- unknown capture size raises an actionable error;
- draft prefill workspaces are weak-referenced.

NPU validation covers identical outputs before/after sleep, first and subsequent
replay after wake, no address errors, no memory growth over repeated loops, and
sleep-time memory / wake latency / first-request latency measurements.

## Relationship to other work

- #20481 proposes an in-place ACL graph lifecycle. It shares this file but
  assumes that clearing a workspace handle releases HBM while the graph is
  retained; that assumption does not hold for the NPU graph pool.
- #20427 proposes a CaMem pool that backs up graph memory and restores it at the
  same virtual address on A5. This design is the "always re-capture" baseline;
  it does not implement that pool.
- HCCL suspend/resume policy is intentionally out of scope here and stays with
  the HCCL cleanup path.
