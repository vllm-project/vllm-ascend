# Ascend Cross Encoder Store ownership

The Ascend connector reuses upstream content keys, tensor metadata, cache
integration, admission accounting and publication ordering. Ascend owns the NPU
buffers and their transport registrations.

## Fixed staging pool

The producer allocates one 2 MiB-aligned slab before exposing worker services.
Its usable capacity is the configured `ec_buffer_size`. There is one producer
allocator and no additional arena, capacity calculation or compatibility option.

Responsibilities follow the upstream split:

- `store_client.py` subclasses the upstream client, reusing existence checks,
  read-byte grouping, metadata helpers and poison ownership. It overrides NPU
  tensor I/O and fixed-pool binding/validation/lifetime; it never uses the
  upstream CPU buffers or transient registration path.
- `store_backend.py` selects the engine topology, initializes native Store and
  extends the upstream publisher with deferred submission, admission/storage
  accounting and asynchronous ownership.
- `memory.py` owns pool allocation; `transfer.py` owns the EC engine adapter.

There is no separate registration manager for fixed-pool suballocations.

All producer P2P sources, Store PUT sources and Store GET destinations are
suballocations of this pool. Original encoder tensors and final cache tensors
are never dynamically registered. There is no runtime registration expansion,
per-request deregistration or reconnect workaround. Slab initialization failure is fatal; there is no alternate allocation retry.

| Topology | EC registration | Store access to the producer slab |
| --- | --- | --- |
| Shared RoCE | Register once through `global_te` | Borrow that registration |
| Independent RoCE | Register once through `global_te` | Register the same slab once through Store |
| FabricMem | Register once through `global_te` | No additional explicit Store registration |

Store setup and slab binding finish before consumer/control services start.
Existing transport selectors are preserved: explicit RoCE-only EC configuration
keeps its HCCL selectors through deferred channel creation. Mixed HCCS/RoCE
operation still needs hardware validation. Registering one slab in two engines
is an ownership policy, not proof of transport correctness or an established
root cause of earlier failures.

Fixed EC slabs and their native owners remain retained until process exit,
including producer-only P2P. Python shutdown drains work but is not treated as
proof that native channel construction has retired. This policy does not
support repeatedly reclaiming and rebuilding these resources in one process.

## Encoding, P2P and PUT

The transfer thread waits for source readiness, then atomically allocates the
whole batch. An admitted Store object's metadata header and whole payload
occupy one allocation. NPU copies finish before pointers are submitted to
Mooncake. EC fanout and Store PUT share the staged payload, with no payload
prefix/suffix splitting or chunked payload copy.

The producer's original tensor remains owned through staging; existing push
and publisher bookkeeping may hold it longer. The scheduler continues to own
`encoder_cache` eviction. Store admission and pending-storage budgets are
unchanged.

EC completion notification follows completion of all its shard writes. It
does not wait for Store PUT. After EC writes settle, callbacks retain the
staging lease until every submitted PUT settles, then return the regions to
the pool. PUT completing first cannot release bytes while EC is reading them.
A rejected/admission-skipped Store publication does not invalidate successful
EC delivery.

Temporary exhaustion waits for completed leases. A whole batch, including headers and allocator alignment,
that exceeds pool capacity fails explicitly; it never falls back to dynamic
registration. There is no second payload arena. GET sends use a separate executor with
the existing transfer-worker limit so they can release GET leases even when
normal transfer threads are blocked waiting for staging capacity.

## GET and subsequent P2P

Store reads complete objects, including the existing 24-byte header, directly
into leased pool addresses through `batch_get_into`. Metadata and exact result
sizes are validated before publishing a hit. Read grouping respects both the
existing read-byte limit and aligned pool capacity; individual payloads are not
split into chunks.

GET executes on the model thread and does not wait for pool space: that thread
may still need to bind or dispatch work which would release space. If no whole
chunk lease is available, those hits retain the existing encoder-computation
fallback. Oversized objects are skipped before I/O.

For a valid hit, hand the registered payload directly to a P2P task before
submitting the copy into an independently owned NPU cache tensor. P2P can wait
for remote reservations on its own executor while the model thread copies and
synchronizes the cache tensor. The upstream resolver then inserts that independent
tensor into `encoder_cache`, so the runner skips encoding this hit.

The GET chunk lease remains owned until both the cache copy and every direct
P2P task finish. P2P completion notification does not wait for the cache copy.
No weak references, optional slab-copy cache, or lookup by cache tensor identity
are used. Later P2P requests for an already cached tensor use normal staging.

The direct task claims the same upstream producer records under their lifecycle
lock. Those records cannot also be submitted by normal cache binding/dispatch;
request cancellation and shutdown retain the upstream in-flight ownership rules.
GET sends do not acquire staging space or submit redundant Store PUTs. A hit
with no waiting push releases its GET lease as soon as the cache copy completes.

## Failures and verification

Safe native rejections and malformed metadata release completed resources.
Unknown GET/PUT/EC or asynchronous copy completion poisons the affected pool,
wakes blocked acquirers and retains relevant tensors, leases and native owners.
Potentially live bytes are not returned for reuse.

CPU tests cover aligned slab initialization, atomic header/payload allocation,
space waiting, direct GET-to-P2P handoff, both reader completion orders, poison handling,
EC/PUT completion ordering and startup across all three Store topologies.
Native Mooncake calls and NPU streams are mocked; these tests do not establish
hardware transfer correctness. Acceptance still requires the actual installed
Mooncake/CANN/driver combination with repeated reuse, EC fanout, PUT and GET
hits.

## Observations

`EC_OBSERVATION` logs are cumulative per process/component and emitted at most
once per five seconds during activity, with a final snapshot on orderly close.
GET counters distinguish candidate objects/payload bytes, pool-unavailable
objects/payload bytes, and successful loads. Pool-unavailable means an entire
GET group could not be reserved (including fragmentation), not proof of OOM.
Lease durations run from allocation through final release; `get` includes cache
copy and direct P2P, `p2p_put` includes shared EC/PUT ownership, and `p2p` is
ordinary staging. Durations overlap across concurrent leases and are not wall
time or isolated network/PUT latency. Unreleased/poisoned leases are not counted
as completed samples. Allocation-wait duration measures condition waits.

Consume the latest cumulative snapshot rather than summing snapshots. These
are process-lifetime totals, not phase-specific or per-object latency percentiles.

RoCE selectors are reapplied after EC engine initialization and fixed-slab
registration on both roles. `EC_TRANSPORT` initialization logs compare Python
and libc environment values before/after these boundaries; they do not prove
HCCL adopted a configuration or that channels are ready. Hardware acceptance
must check that both endpoints actually select the same NIC/link; Python env
tests alone do not establish that.

### RoCE E2E configuration and Store hit evidence

For a RoCE-only EC/Store test, set `comm_resource_config.protocol_desc` to
`["roce:device"]` at both the top level and under `store`. A per-service
`ASCEND_GLOBAL_RESOURCE_CONFIG` replaces the launcher value; a Store-only
configuration does not select RoCE for EC consumers. Producers can subsequently
set the HCCL selectors during Store setup, leaving the two EC peers with different
transport selections. The four-card test keeps both configurations explicit.

A successful GET emits `Loaded N encoder output(s) from Mooncake Store` after
metadata validation and cache-copy completion. Empty, invalid or failed reads do
not emit this marker. It establishes successful Store loading; completion of the
subsequent P2P readers remains governed by their futures and the slab lease.
