# Ascend sleep-mode backends

Ascend registers `ascend_legacy` and `ascend` with the upstream
`SleepModeBackendFactory` through the `ascend_sleep_mode_backend` general plugin.
The NPU platform translates the upstream default `cumem` to `ascend_legacy`.
The built-in GPU `cumem` registration is unchanged. Unknown names fail during
configuration; they do not silently choose another backend.

`ascend` is opt-in:

```bash
vllm serve /path/to/model --enable-sleep-mode --sleep-mode-backend ascend \
  --additional-config '{"rl_config":{"enabled":true,"sleep_mode_extra_cleanup":true}}'
```

To disable the new lifecycle, restart with `--sleep-mode-backend ascend_legacy`
or omit the backend option. There is no new HTTP endpoint or environment variable.
The existing `/sleep` and `/wake_up` dispatch paths call the selected backend.
General plugin discovery must include `ascend_sleep_mode_backend` when users
restrict plugins with upstream `VLLM_PLUGINS`.

## Ownership and extension points

The upstream factory constructs a backend without arguments. `NPUWorker` binds
an `AscendSleepRuntime` to that instance. The runtime holds a weak worker
reference and borrows its model, allocator, transfer engine and cleanup manager.
There is no global current-worker reference. Plugins can be discovered without
initializing an NPU or constructing an allocator.

| Resource | Existing owner | Suspend / resume behavior |
| --- | --- | --- |
| Weights | CaMem `weights` pool | Level 1 CPU offload; level 2 discard and caller reload |
| Retained parameters / buffers | Worker CPU backups | Save before level 2; consume only during weights wake |
| KV cache / other tagged allocations | CaMem | Unmap; restore requested tags at the original reserved addresses |
| Transfer registration | Global transfer engine | Unregister before unmap; reregister after KV remap |
| ACL graphs and attention workspaces | Existing sleep manager / runner | Extra cleanup clears them; KV wake recaptures |
| HCCL / MoE communication resources | Existing HCCL sleep manager | Use existing teardown, anchor, restore and refresh algorithms |
| Process SIGTERM / SIGINT handlers | EngineCore / WorkerProc caller | Preserve around SDK initialization and graph rebuild; keep SDK atexit finalizers |
| CPU world process group | Distributed runtime | Retained for stage-result agreement; backend does not destroy it |
| Compiled artifacts / external resources | Existing runtime | No additional preservation guarantee |

This architecture task does not implement graph reuse, a new allocator,
HCCL suspend/resume, durable checkpoints or selective discard. All upstream
preservation capability methods remain conservatively false. The legacy adapter
retains the previous lifecycle ordering. A backend `close()` never wakes
discarded weights or destroys borrowed allocator / process-group resources.

## New lifecycle and failure behavior

```text
RUNNING
  -> drain -> backup -> allocation inventory
  -> cleanup -> synchronize -> unregister -> allocator sleep
  -> SUSPENDED
  -> validate wake -> remap requested tags
  -> register KV / restore saved weights and buffers as selected
  -> HCCL and graph rebuild -> synchronize
  -> RESUMING while tags remain; RUNNING only when all are restored
```

Each stage agrees on success over the surviving CPU world group before the
next stage. This prevents a healthy rank from advancing after a peer reports
a local error. It cannot recover a lost process, repair a corrupted device,
or prevent a timeout after failure *inside* an HCCL collective. Distributed
timeouts and worker restart remain required for those failures.

Repeated sleep at the same level, already-restored tag wake, full wake while
running and close are idempotent. Changing the sleep level mid-cycle fails.
Invalid levels / tags and KV-before-weights restoration fail before mapping.
A backup failure before destructive cleanup clears CPU backups and rolls back
to RUNNING only after rank agreement. Failure during cleanup or later leaves
the backend failed and rejects execution and further lifecycle operations.
Recovery is worker restart, not a blind retry after a partial copy or unmap.

Level 2 callers must restore weights, reload discarded model weights, then
restore KV. `wake_up(tags=["weights"])` keeps the backend RESUMING. The existing
manager initializes HCCL for external weight loading at this point, but delays
graph capture until KV wake. Saved parameter patterns retain their existing
meaning; updating the contents of valid mapped weight storage is supported.
The backend cannot infer whether an external trainer finished reloading weights.

Execution and lifecycle operations share a worker-local lock. Existing
engine/scheduler pause and request-drain behavior remains the request boundary;
the lock drains an already-entered worker call and device synchronization drains
queued device work. New worker execution while suspended, partially resumed,
failed or closed raises an error. This is not a replacement for request admission
control at the server boundary.

## Configuration behavior and validation scope

| Configuration | Behavior |
| --- | --- |
| Sleep disabled | No backend instance; normal execution unchanged |
| `ascend_legacy` | Previous allocator/cleanup order and diagnostics |
| `ascend`, eager | Same allocation policy; explicit lifecycle and rank agreement |
| `ascend`, ACL graph with extra cleanup | Existing recapture on KV wake for every configured capture size |
| Extra cleanup disabled | Existing graph / communicator retention path; no new preservation guarantee |
| Attention backend / model / TP / EP / runner choice | Existing manager and model support applies; this change adds no kernel or model support |
| FRACTAL_NZ weights | Existing wake validation rejects before mapping |
| Selective discard | Explicit unsupported error |

Configured capture sizes and attention backends are not rewritten. They are
passed to the existing runner's capture algorithm, including all its current
compatibility restrictions. Supporting a configuration in code does not mean
it has hardware acceptance evidence. See the PR evidence index for the exact
tested source SHA, environment, models, captures, repeats and open limitations.

Useful logs include `Ascend sleep backend stage=<name> status=complete`,
CaMem restore counts, HCCL restore records and the existing graph capture logs.
These logs prove stage execution, not graph reuse or output correctness.

Graph SDK lazy imports may replace process-wide SIGTERM/SIGINT handlers. Worker
initialization, model loading, profiling, initial capture and wake recapture
preserve the handlers present on entry. This applies to both backend choices
and normal initialization, so a graph SDK import cannot swallow the caller's
shutdown request. Restoration uses public Python signal APIs on the main thread;
it does not replace SDK functions or remove their atexit finalizers.
