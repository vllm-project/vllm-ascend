# Shared host-backed Engram

This opt-in path lets independent DeepSeek V4.1 instances map the same
read-only INT8 Engram tables on supported A3 systems. A separate owner loads
the tables into MemFabric BM host DRAM. Consumers join the BM group and use
the existing NPU gather/dequantization kernel; only selected BF16 results are
materialized in HBM. This does not remove model collectives or KV transfer.

The default remains unchanged. Without `engram_shared_host_dir`, existing
HBM and host-offload paths retain their normal behavior.

## Requirements

- A3 devices with CANN Fabric-handle support and working cross-node host-memory
  imports. Same-supernode placement alone is not sufficient proof of support.
- MemFabric BM with `create2`, dynamic membership and `gva_to_va` support.
  No custom VMM library or additional CMake target is required.
- A correctly packaged MemFabric wheel containing the upstream duplicate-library
  loading fix (memfabric_hybrid commit `cafd1251`). Older wheels can load two
  copies of native libraries and abort during process teardown. Check clean
  startup/shutdown before using a new image; the version string alone does not
  establish that the packaging fix is included.
- An immutable, identical checkpoint on every instance. Supported tables use
  INT8 codes of width256 and group32 scales; the owner converts scales to FP32.
- A private shared-filesystem directory accessible under the same UID by the
  owner and all consumers, plus a trusted network for the owner control socket.
- Enough owner-node DRAM for the complete table. Codes and scales are separate
  mappings, each rounded to 1GiB. No full-table HBM allocation is required.

The identity check hashes checkpoint configuration and index metadata, not all
weight payloads. Pin an immutable model revision independently. Generation
files contain control credentials: do not publish them, include them in bug reports,
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
servers. The BM store uses an available port on `--listen-host` by default;
use `--store-url tcp://OWNER_ADDRESS:PORT` when an explicit port is needed.
`--max-consumers 64` reserves room for the owner plus 64 worker processes,
enough for three DP4/TP4 prefills and one DP8/TP2 decoder. Two Engram layers in
one worker share a process-wide BM session and use separate BM objects.
Another subsystem in that process must not initialize the BM singleton with a
different store; MemFabric TransferEngine has a separate API.

Consumers initialize the model's AICPU metadata scheduler before creating BM
mappings. On the tested A3 stack, starting that scheduler after remote mappings
failed `halMemBindSibling` and stalled FULL decode graph capture. The deterministic
startup warmup uses the existing metadata operator once per worker; it does not
modify model weights, sampling state or the inference graph. Both Engram layer
clients keep the same BM session alive throughout the model lifetime.

Add these options to each server's otherwise unchanged launch:

```bash
--engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
--additional-config '{"engram_shared_host_dir":"/shared/private/engram-generation-001"}'
```

Merge this key into any existing additional configuration; do not replace
other model settings. Consumers validate identity and layout, import their TP
row views, and skip reloading shared table parameters. Consumers contribute zero
local DRAM/HBM table capacity. MemFabric handles allocation, membership and
mapping; a one-time `gva_to_va(..., LOCAL_DEVICE)` resolution supplies the NPU
pointer table. Runtime lookup is still a direct load plus dequantization into
selected BF16 HBM rows, with no BM copy/staging call or per-request GVA query.
Different TP layouts
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

## Current MemFabric validation

CPU protocol and storage tests use fake BM allocation and real control sockets.
They cover publication bounds, zero-consumer allocation requests, metadata-only
views, live-view release rejection and retry after a failed leave. Real A3
component tests exercised a persistent owner, two layers/four BM objects,
two per-layer clients sharing one BM session, native gather/dequantization,
graph replay and continued reads after one layer's client detached.
The tested deployment used a repaired 1.2.0 wheel with a single canonical
native library set; native ELF code was unchanged. This packaging repair is
not shipped by this PR, and deployment needs a correctly packaged MemFabric.

A full-model GVA comparison ran on four A3 servers in one physical supernode,
with CANN 9.2, torch-npu 2.10 and 64 NPUs: three DP4/TP4 prefills and one
DP8/TP2 decoder. Both native host offload and GVA completed 1440/1440 requests
without errors. One owner held two checkpoint tables/four BM HOST objects;
64 workers attached 128 layer views with zero local table capacity.
Table backing fell from approximately 824 GiB to 208 GiB; consumers had no
large POSIX table replicas. This is table storage, not total memory or HBM.

| Prompt bucket | Concurrency | Native QPS | GVA QPS | GVA TTFT P50 (s) | GVA TPOT P50/P90 (ms) |
| --- | ---: | ---: | ---: | ---: | --- |
| 0–10K | 96 | 3.979 | 4.181 | 1.709 | 28.951 / 35.532 |
| 10–20K | 108 | 4.133 | 4.172 | 3.014 | 28.998 / 39.317 |
| 20–40K | 120 | 3.773 | 3.832 | 8.182 | 26.217 / 36.375 |
| 40–80K | 64 | 2.038 | 2.034 | 9.571 | 24.356 / 32.142 |
| Cache-hit dataset | 92 | 4.024 | 4.089 | 1.481 | 29.158 / 36.863 |

Each case used three closed-loop waves, a 700-token cap, 24 warmup requests,
and backend prefix-cache reset. All paired request hashes and prompt-token
counts matched. The cache-hit label names a dataset, not a measured hit rate.
This single sequential A/B supports broadly unchanged throughput; it does not
prove a stable speedup. Both baseline (8.007s) and GVA (8.182s) missed the
20–40K TTFT target of 7s. Full SLO acceptance and model-quality validation
remain open. The private request corpus is not distributed with this change.

The hardware run used Engram leaf-module overlays on a pinned model runtime,
not a newly built complete framework wheel. Initializing the metadata AICPU
operator before remote BM imports resolved a reproducible decode graph-capture
stall; FULL decode graphs were retained. The tested prototype initialized
before embedding construction; the implementation narrows this to the shared
allocation branch before the first client attaches. Active-graph owner failure
and arbitrary BM teardown/reinitialization remain unvalidated.

The component benchmark can be reproduced with the following two commands in
separate terminals on a supported, correctly packaged A3 environment. Choose a
fresh private shared root, wait for `POOL_READY`, then start consumers:

```bash
python benchmarks/benchmark_shared_engram.py owner \
    --root /shared/private/engram-bm-smoke --host OWNER_ADDRESS --device 0
python benchmarks/benchmark_shared_engram.py consumer \
    --root /shared/private/engram-bm-smoke --device 1 --tag consumer-1
```

Repeat the consumer command with distinct tags to exercise fresh processes.
Stop all consumers before creating `/shared/private/engram-bm-smoke/stop` to
shut down this benchmark owner. This uses small warm tables; its HBM/GVA lookup
timings are not full-model throughput or DRAM bandwidth-ceiling measurements.
The real model owner remains `python -m vllm_ascend.shared_engram` above.

## Historical VMM validation

CPU protocol tests are in `tests/ut/models/test_engram_shared_pool.py`. They
exercise real control sockets with a fake allocator; they do not establish
hardware correctness. A3 validation has exercised remote native lookup,
independent detach/reattach, actual checkpoint rows, and embedding graph replay.
The previous raw-VMM backend's full-model P/D was tested on four A3 nodes in one physical supernode, with
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
