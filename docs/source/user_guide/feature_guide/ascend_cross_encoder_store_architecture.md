# Ascend Cross Encoder Store ownership

The Ascend connector reuses upstream content keys, tensor metadata, cache
integration, admission accounting and publication ordering. Ascend owns the NPU
buffers and their transport registrations.

## Fixed staging pool

The producer allocates one 2 MiB-aligned slab before exposing worker services.
Its usable capacity is the configured `ec_buffer_size` plus the capacity
resolved from the existing `ascend_mooncake_bounce_arena_size` option. The
legacy option keeps its existing default and validation; its bytes now belong
to staging, with no separate bounce partition or prefix/suffix path.

All producer P2P sources, Store PUT sources and Store GET destinations are
suballocations of this pool. Original encoder tensors and final cache tensors
are never dynamically registered. There is no runtime registration expansion,
per-request deregistration or reconnect workaround. Slab initialization failure
is fatal; there is no smaller bounce-only allocation retry.

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

Temporary exhaustion waits for completed leases. Completed optional GET copies
are reclaimed first. A whole batch, including headers and allocator alignment,
that exceeds pool capacity fails explicitly; it never falls back to dynamic
registration. There is no second payload arena or new executor.

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

For a valid hit, copy the payload into an independently owned NPU cache tensor
and synchronize the copy. This prevents cache residency from pinning the pool
until scheduler eviction. Keep the completed pool copy as an optional lease,
indexed by a weak reference to the cache tensor:

- The next P2P staging operation can take that lease and use the received
  payload directly, without another allocation or copy.
- A different allocation can reclaim this optional copy under pressure. The
  independent cache tensor remains valid; later P2P stages it normally.
- Dropping the cache tensor returns an unused optional lease.
- A lease selected by a waiting P2P operation is removed from the reclaimable
  set before waiting, so another allocator cannot overwrite it.

Thus GET does not occupy a pool region indefinitely, and GET-to-P2P never
requires holding one mandatory lease while waiting for a duplicate copy.

## Failures and verification

Safe native rejections and malformed metadata release completed resources.
Unknown GET/PUT/EC or asynchronous copy completion poisons the affected pool,
wakes blocked acquirers and retains relevant tensors, leases and native owners.
Potentially live bytes are not returned for reuse.

CPU tests cover aligned slab initialization, atomic header/payload allocation,
space waiting, cache-copy reclamation, GET-to-P2P lease transfer, poison handling,
EC/PUT completion ordering and startup across all three Store topologies.
Native Mooncake calls and NPU streams are mocked; these tests do not establish
hardware transfer correctness. Acceptance still requires the actual installed
Mooncake/CANN/driver combination with repeated reuse, EC fanout, PUT and GET
hits. Benchmark inputs, concurrency and capacities were not changed or rerun
as part of this implementation.
