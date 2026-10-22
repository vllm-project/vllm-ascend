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
- A private shared-filesystem directory accessible under the same UID by the
  owner and all consumers, plus a trusted network for the owner control socket.
- Enough owner-node DRAM for the complete table. Codes and scales are separate
  mappings, each rounded to 1GiB. No full-table HBM allocation is required.

The identity check hashes checkpoint configuration and index metadata, not all
weight payloads. Pin an immutable model revision independently. Generation
files contain capabilities: do not publish them, include them in bug reports,
or expose the directory to other users. Control traffic is authenticated but
not encrypted; it carries no table rows.

## Start and attach

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

CPU protocol tests are in `tests/ut/models/test_engram_shared_pool.py`. They
exercise real control sockets with a fake allocator; they do not establish
hardware correctness. A3 validation has exercised remote native lookup,
independent detach/reattach, actual checkpoint rows, and embedding graph replay.
Full-model matched P/D performance and lifecycle validation are still in
progress. No end-to-end throughput or SLO improvement is claimed here.
