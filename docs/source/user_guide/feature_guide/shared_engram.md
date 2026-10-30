# Shared host-backed Engram

This opt-in path lets independent DeepSeek V4.1 instances map the same
read-only INT8 Engram tables on supported A3 systems. A separate owner loads
the tables into host DRAM. Consumers import the physical allocations and use
the existing NPU gather/dequantization kernel; only selected BF16 results are
materialized in HBM. This does not remove model collectives or KV transfer.

The default remains unchanged. Without `engram_shared_host_dir`, existing
HBM and host-offload paths retain their normal behavior.

## Requirements

- A3 devices with CANN Fabric-handle support and working cross-node host-memory
  imports. Same-supernode placement alone is not sufficient proof of support.
- The optional `libengram_host_vmm.so` build target, enabled when CANN exposes
  the required APIs on an `ascend910_93` build.
- An immutable, identical checkpoint on every instance. Supported tables use
  INT8 codes of width256 and group32 scales; the owner converts scales to FP32.
- Either a private shared directory or a private local directory per pod for
  network bootstrap, plus a trusted network for the owner control socket.
- Enough owner-node DRAM for the complete table. Codes and scales are separate
  mappings, each rounded to 1GiB. No full-table HBM allocation is required.

The identity check hashes checkpoint configuration and index metadata, not all
weight payloads. Pin an immutable model revision independently. Generation
files contain capabilities: do not publish them, include them in bug reports,
or expose the directory to other users. Control traffic is not encrypted and
carries no table rows; network bootstrap itself is not authenticated.

## Start and attach

### Network bootstrap without a shared filesystem

When the deployment provides a stable TCP address for the owner, the launcher
can transfer the small VMM handle descriptors over the control connection. The
Engram tables stay in owner DRAM. Each pod uses its own private local directory
at the same path; no filesystem is shared between pods.

On the designated owner node:

```bash
python -m vllm_ascend.shared_engram_launch --role owner \
  --model /models/immutable-checkpoint --pool-dir /tmp/engram/pool \
  --listen-host OWNER_POD_IP --owner-port 29699 \
  -- /path/to/service-command
```

On each consumer node:

```bash
python -m vllm_ascend.shared_engram_launch --role consumer \
  --model /models/immutable-checkpoint --pool-dir /tmp/engram/pool \
  --owner-host OWNER_SERVICE_ADDRESS --owner-port 29699 \
  -- /path/to/service-command
```

The parent of `/tmp/engram/pool` must exist locally and the pool directory must
not exist. The same local path is passed as `engram_shared_host_dir` to each
model server. The owner address and node placement are deployment
responsibilities. Initial bootstrap is neither authenticated nor encrypted;
restrict the owner port to trusted peers. A random generation identifier
rejects descriptors cached before an owner restart; consumers must bootstrap
again into a new local pool directory. The shutdown constraints below apply.

### Shared filesystem bootstrap

The installed package also provides a launcher for an existing service
command. On one designated owner node, run:

```bash
python -m vllm_ascend.shared_engram_launch --role owner \
  --model /models/immutable-checkpoint --pool-dir /shared/private/engram-generation-001 \
  --listen-host OWNER_ADDRESS -- /path/to/service-command
```

On each other node, use the same model and pool directory, with `--role
consumer` and its own service command. The launcher checks checkpoint identity
and waits for the live owner before starting the command. It does not modify
the command's Engram options: configure `engram_shared_host_dir` as shown below.
The owner runs in the designated node's launcher, so stop all other consumers
before stopping that node. Allow enough termination time for the owner to drain
leases; unexpected owner failure and forced container termination are not
transparent failover.

Choose a new, nonexistent generation directory on the private shared mount.
Start the owner in a supervised process, using an address reachable from every
consumer:

```bash
python -m vllm_ascend.shared_engram \
  --model /models/immutable-checkpoint \
  --pool-dir /shared/private/engram-generation-001 \
  --device 0 --listen-host OWNER_ADDRESS
```

Wait for `POOL_READY`. The owner must remain alive independently of all model
servers. Add these options to each server's otherwise unchanged launch:

```bash
--engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
--additional-config '{"engram_shared_host_dir":"/shared/private/engram-generation-001"}'
```

Merge this key into any existing additional configuration; do not replace
other model settings. Consumers validate identity and layout, import their TP
row views, and skip reloading shared table parameters. Different TP layouts
can use the same physical backing. Request scheduling, token IDs, and KV state
remain independent. Dummy loading and sleep mode are not supported.

## Shutdown, recovery, and fallback

Stop request admission and shut down consumers before stopping the owner.
Mappings must be synchronized and released before their control lease is
acknowledged. Graceful owner shutdown enters draining, refuses new consumers,
and waits for existing leases rather than freeing live backing memory.

Unexpected consumer disconnection conservatively retains an orphaned lease.
Do not force-stop the owner while another consumer may still access its table.
For orphan recovery, stop and verify termination of **all** consumers of that
generation, then terminate the owner and create a fresh generation. Never reuse
stale capability files. Automatic orphan reclamation and transparent owner
failover are not implemented. Active-graph owner-failure behavior still needs
hardware validation before production use.

To fall back, restart consumers without `engram_shared_host_dir`. Budget for
the original per-instance table replicas before doing so. Preserve the owner
until all shared consumers are gone.

## Validation scope

The network bootstrap was also tested with two independent A3 pods on separate
nodes in one physical supernode, using a small synthetic checkpoint. The owner
and consumer launchers exchanged VMM descriptors over TCP; remote INT8
lookup, dequantization, graph replay, consumer reattachment, and orderly owner
shutdown passed. This is not a full 3P1D or production-image validation.

This release-branch backport preserves the native CANN VMM implementation from
[PR #19528 at f1b80d6db](https://github.com/vllm-project/vllm-ascend/commit/f1b80d6db4ed73c059f7dcda4f487321786e9add).
It does not include the later MemFabric/GVA replacement. The measurements below
are historical evidence from that VMM implementation, not a new benchmark of
this release-branch backport. NPU compilation and full-model validation of the
backport remain required before enabling it in a deployment.

CPU protocol tests are in `tests/ut/models/test_engram_shared_pool.py`. They
exercise real control sockets with a fake allocator; they do not establish
hardware correctness. A3 validation has exercised remote native lookup,
independent detach/reattach, actual checkpoint rows, and embedding graph replay.
Full-model P/D was tested on four A3 nodes in one physical supernode, with
CANN9.2 and torch-npu2.10. The pinned deployment used three DP4/TP4 prefills
and one DP8/TP2 decoder, native INT8 lookup and the same KV connector for both
placements. Both runs completed1440/1440 requests without errors. Physical
table backing fell from approximately824GiB to208GiB. This is table storage,
not total system memory or HBM usage.

| Prompt bucket | Concurrency | Native QPS | Shared QPS | Shared TTFT P50 (s) | Shared TPOT P50/P90 (ms) |
| --- | ---: | ---: | ---: | ---: | --- |
| 0–10K | 96 | 3.819 | 4.381 | 1.357 | 28.527 / 34.950 |
| 10–20K | 108 | 3.756 | 4.144 | 2.966 | 28.260 / 38.143 |
| 20–40K | 120 | 3.581 | 3.847 | 8.238 | 26.042 / 35.879 |
| 40–80K | 64 | 1.969 | 2.027 | 9.272 | 24.616 / 32.580 |
| Cache-hit dataset | 92 | 3.981 | 4.154 | 1.398 | 29.538 / 36.130 |

Each case used three closed-loop waves, a700-token cap,24 warmup requests and
backend prefix-cache reset. Paired request hashes and prompt-token counts
matched. The cache-hit label identifies the dataset, not a measured hit rate.
This single sequential A/B shows no observed QPS regression; it does not prove
a stable speedup. The20–40K TTFT target of7s was missed by both baseline7.640s
and shared8.238s. Full SLO acceptance and owner-failure validation remain open.
The private request corpus is not distributed with this change; reproduce
performance using a representative corpus and report workload differences.
