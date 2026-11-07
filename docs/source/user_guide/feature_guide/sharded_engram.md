# Static multi-owner Engram on A3 (experimental)

This opt-in extension distributes each immutable Engram table's rows across two
or four A3 owners. Independent P/D instances keep their own TP/DP configuration
and share the backing. It reuses native VMM and the existing lease protocol.
The single-owner path remains available with its original launcher arguments.

## Membership and layout

All pods receive the same local JSON file before model readiness; no shared
filesystem is required. Owner order determines row ownership. For example:

```json
{
  "schema": 1,
  "deployment_id": "engram-deployment-1",
  "owners": [
    {"host": "OWNER_0_IP", "port": 29699},
    {"host": "OWNER_1_IP", "port": 29699},
    {"host": "OWNER_2_IP", "port": 29699},
    {"host": "OWNER_3_IP", "port": 29699}
  ]
}
```

Use actual reachable IP addresses. The existing production discovery of a
healthy decode cannot bootstrap this layout: decode may itself need the
prefill owners. The current implementation accepts pre-supplied membership;
the early owner registry described below provides automatic membership discovery.

Each layer is divided into balanced groups of `2^22` rows, with the final
partial group retained. Therefore ownership is approximately one quarter,
not necessarily exactly one quarter. Codes and scales share row ownership.
Owners allocate and load only their local rows. Allocation rounding to GiB
must be included in physical-memory accounting.

Consumers build a device pointer table in global chunk order. Each pointer
addresses an imported owner's device mapping. Head-range validity remains
unchanged, while addressing uses the global row. Kernel output and existing
TP exchange retain their original layout by default. Optional SP token-owned
lookup is described below.

## Launch

On owner node `i`, with the membership file at `/tmp/owners.json`:

```bash
python -m vllm_ascend.shared_engram_launch \
  --role owner --model /path/to/model \
  --pool-dir /tmp/engram/new-generation \
  --listen-host "$POD_IP" --owner-set /tmp/owners.json --owner-rank "$i" \
  --device 0 -- <existing-model-service-command>
```

An independent consumer that stores no backing uses:

```bash
python -m vllm_ascend.shared_engram_launch \
  --role consumer --model /path/to/model \
  --pool-dir /tmp/engram/new-generation --owner-set /tmp/owners.json \
  -- <existing-model-service-command>
```

The model command retains `cpu_offload=true`, `dp_shared_memory=true`, and
`additional_config.engram_shared_host_dir=/tmp/engram/new-generation`.
Each pod needs a fresh private directory. The launcher starts the local
owner, bootstraps all owners concurrently and starts the model only after
the bundle is ready. The optional loading-overlap mode is described below.

Cross-layer table leases reuse one owner connection per worker. The last
layer release closes the pool. Stop device work and graphs before releasing
views. Preserve the established trusted-network restriction of the native
bootstrap; no control handle should be exposed publicly.

## Validation and limitations

CPU tests cover two/four-owner TCP bootstrap and leases, nonaligned head
ranges, global code/scale pointers, mismatched layouts/generations and actual
safetensors shard loading. An isolated A3 two-device probe passed 72 eager
and changed-ID graph checks across two tables and TP2/TP4 head layouts, using
synthetic weights. It compiled the current native source. This was one pod;
it does not establish multi-node/model correctness or performance.

The original prototype was exercised in a four-node A3 3P1D setup; the public
port still requires NPU revalidation. A2 HIXL and A5 cross-node support are
outside this A3 implementation. Dynamic membership, transparent failover and
backing migration are not implemented.

## Early discovery through the existing conductor/kvmaster pod

For production-style startup, run the registry in the existing kvmaster
container before model health is available. Invoke the script directly in a
CPU-only control container so package initialization cannot probe NPU drivers.
It uses Python's standard library
and does not initialize NPU operators. No extra shared filesystem, Service or
injected key is needed; restrict the endpoint to the trusted deployment network.

```bash
python /path/to/vllm_ascend/shared_engram_registry.py --owners 4 --port 29698 -- \
  /path/to/mooncake_master <existing-master-flags>
```

On each of the four P/D nodes, use the owner's launcher with
`--owner-registry "http://${KVMASTER_ADDR}:29698"` instead of `--owner-set`
and `--owner-rank`. Supply the local reachable address through `--listen-host`.
The registry orders the four registered node identities and publishes the same
membership to every participant; the launcher persists it privately, then
starts the local shard owner. Independent read-only consumers can discover
that membership without registering another owner.

The registry freezes after four owners. A changed endpoint, model identity or
replacement owner is rejected. Restarting the registry requires a new deployment;
it does not provide transparent owner failover. Stop all consumers before owners
or the control pod during planned teardown.

A review YAML is retained with the experiment artifacts. It preserves the
existing P50GiB/D0 pool settings and model engine flags. It requires a newly
built image containing these modules; the supplied October image requires the
validated source overlay for the isolated experiment.

## Experimental SP direct lookup

`additional_config.engram_sp_direct_lookup=true` enables token-owned lookup when
the model actually uses sequence parallelism and has global sharded pointers.
Non-SP execution retains the original head-sharded path. The option defaults
to false and requires the shared sharded backend.

The hash/history update still processes the full step. Each TP rank then looks
up all heads only for its own padded token window, directly into its persistent
SP buffer. The fused lookup retains the old head-owner vocabulary guards.
The group submits T×H row slots, rather than duplicating T×H on every rank.
The Engram TP head gather and the later result SP copy are bypassed. Other
model collectives remain unchanged.

For two layers at P capacity8192/TP4, persistent lookup tensor capacity drops
from192MiB to48MiB per rank; global token-mask capacity is unchanged. This is
workspace accounting, not a claim that total NPU HBM falls by144MiB, since the
KV planner may use the freed memory. Runtime logs report actual workspace bytes.
Profiler ranges `engram_tp_head_all_gather` and `engram_sp_direct_lookup` allow
checking the changed producer path.

CPU and two-device checks cover token-window mapping, full history updates,
query count, cross-head invalid IDs, padding and changed-ID graph replay.
Service performance and full-model SP validation must be reported separately;
operator correctness alone does not establish an end-to-end gain.

## Experimental loading overlap

Add `--overlap-load` to the sharded `shared_engram_launch` command to start the
model command while the owners load their shards. This works with either static
membership or the early owner registry. The default serial startup is unchanged.
All participating images need the deferred-attachment worker implementation;
changing only the launcher in an older image is insufficient.

The launcher creates a fresh private `loading` manifest, then bootstraps owner
handles in the background and atomically publishes `ready`. During construction,
shared tables expose only meta parameters and remember their view specifications.
The normal weight loader continues to skip those tables. After target and draft
weight loading, the worker waits for the bundle, validates deployment identity,
creates leases/mappings and fills pointer tables before memory profiling, warmup
or graph capture. These late device allocations count toward model memory usage.
An unready lookup raises instead of submitting a kernel with missing pointers.
Bootstrap failure, deadline expiry or a changed layout prevents startup.

Launcher stage records (`OWNER_STARTED`, `MODEL_STARTED`, `SHARDS_READY`) go to
stderr. Compare serial and overlap startup using the same checkpoint, resources,
owner settings and file-cache policy. Report table loading, dense/draft loading,
attachment, warmup/graph, ready and first-request times separately. Reattaching
consumers to retained owners must not reload their tables. The implementation
alone does not establish a cold-start improvement; concurrent reads may compete
for storage or CPU resources.

## Shared-table shutdown and owner supervision

Workers stop device work before unmapping shared tables. At shutdown the worker
synchronizes, closes target/draft table views and acknowledges owner leases before
unrelated KV transport cleanup. A failed device synchronization leaves mappings
and leases intact; an individual table cleanup error is reported after the other
tables have also been attempted. Retry or controlled recovery is required.

The sharded service launcher checks every owner's protected ping against the
identity and generation captured during bootstrap. An unreachable, draining or
replaced owner causes the launcher to stop the service; it never silently adopts
a new generation. These checks run in the host supervisor, outside the lookup
kernel. Detection is bounded by polling and socket timeouts, not instantaneous
physical fault tolerance. Owners must run the matching supervision protocol.

For consumer-only restarts, retain independently supervised owners and use the
consumer role with a fresh private bundle. The owner role ties its spawned owner
to the local service lifetime and drains it on service exit. EOF is not evidence
that device accesses ended: abandoned leases conservatively retain backing.
Before force-retiring such an owner, stop the recorded consumer process trees
and prove all nodes have no remaining consumer device processes. Recovery uses
fresh owner generations/bundles and new model processes, rebuilding mappings,
events and captured graphs. A live mapping is never redirected to a replacement.

Immediate request abort and process cleanup use separate time budgets. On NPU,
a zero request timeout still aborts requests immediately. A zero or unspecified
process cleanup budget gives the EngineCore
process manager15 seconds to release mappings and acknowledge leases. This
uses the existing upstream process-cleanup hook; positive request-drain budgets
and their exhausted remaining deadlines are unchanged. The paired vLLM version
must expose `get_engine_process_shutdown_timeout`. Its current diagnostic log
may still call this the ROCm cleanup grace; the Ascend override runs on NPU.
