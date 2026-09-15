# RFork lease lifetime assessment

## Scope and evidence

This change sets the example planner's default lease TTL to 60 seconds. The existing
`RFORK_MOCK_LEASE_TTL_SEC` environment variable and `--lease-ttl-sec` option remain
available. CLI values override valid environment values; durations must be positive
integer seconds. Seed heartbeat expiry remains independent. Release decoupling and
late acquisition are implemented in separate follow-up commits. Renewal remains a
proposal because the customer planner cannot currently be changed.

The reported run completed weight transfers and then received HTTP 400 on release.
The captured log does not include the response body of those real release requests.
A separate probe with a fabricated USER_ID returned `userID ... not found`; that
establishes the planner's response for a nonexistent ID, not the actual failure cause.

The supplied Go entry point calls `ReclaimTimeoutSeeds(60s)` and actively health-checks
seeds. Without that function, `HealthCheck`, `DeleteSeed`, and the get/put handlers,
we cannot classify 60 seconds as a lease TTL rather than a seed heartbeat timeout.
Likewise, `NewSeedMap()` alone does not establish whether storage is shared across
planner replicas. Replica routing and state persistence require deployment evidence.

The current Python example has an independent lease TTL (previously 300 seconds),
seed heartbeat expiry, and no corresponding outbound seed health-check watchdog.
It returns 404 for an expired or unknown lease, which the client accepts as released.
This can hide early expiry: startup succeeds even though the planner already returned
the source's capacity to the pool while a transfer was still active.

## Options

| Option | Benefit | Cost and limitation |
| --- | --- | --- |
| Configurable fixed TTL | Simple, compatible with existing clients | A short TTL releases capacity during a slow transfer; a long TTL retains abandoned capacity |
| Acquire after initialization and layout preparation | Removes local preparation from the lease lifetime | A seed miss now discards a prepared model and runs the default loader again |
| Acquire after memory registration | Lease covers metadata fetch and transfer only | Requires splitting session preparation from transfer and changing state transitions |
| Renewable lease | Covers unpredictable transfer duration while reclaiming abandoned leases | Requires negotiated planner support, independent renewal execution, and loss handling |
| Late acquisition plus renewal | Reduces both occupied capacity and expiry risk | Combines implementation and validation requirements of both changes |

## Late acquisition

Before late acquisition, `RForkModelLoader.load_model` acquired a lease before
initialization and optional processed-layout conversion/NPU synchronization.
It now prepares the model first, then acquires the lease immediately before
registration and transfer. `RForkSession.transfer_from_seed` schedules asynchronous
release after a successful read. Initialization, layout, acquisition, registration,
metadata, read and lease-holding durations are observable separately.

Acquisition now follows layout synchronization and precedes `transfer_from_seed`.
Initialization and layout preparation do not consume the lease;
the session can stay INITIALIZED until acquisition succeeds. Keep model eval and seed
advertisement after successful loading. Handle main and draft sessions consistently.

This reduces the lifetime to registration, metadata fetch, transfer, and release.
The observed roughly 23-second registration/transfer interval is one measurement,
not an upper bound. A 60-second TTL can still expire under load or network stalls.

The seed-miss path deletes the prepared model and reloads it through the default loader.
It now snapshots the compilation registries and rotary cache before construction and
restores that baseline on fallback, including partially failed construction. This
assumes serialized model construction in each worker. CPU tests exercise late seed
miss and preservation of preexisting shared state. Main/draft shared storage,
partial initialization failures, stale layer registrations, and NPU memory retention
must be checked. Reusing an already transformed empty model for checkpoint loading is
not automatically safe for quantized layouts.

Moving acquisition after registration would further shorten leases but is a separate
refactor: acquisition currently requires INITIALIZED while transfer changes LEASED
to REGISTERED. Preparation, acquired lease ownership, and source advertisement need
separate state handling. Preserve memory owners until all in-flight reads have ended.

Required tests include processed and unprocessed call order, initialization/layout
failure before acquisition, late seed miss, metadata/transfer failure after acquisition,
release retry, main/draft isolation, and cleanup of only discarded model state. Measure
seed-hit and seed-miss startup cost and memory on NPU before adopting the change.

## Release retries and startup progress

The release-decoupling implementation now uses one asynchronous worker per lease,
with three single-request attempts and 30 seconds between transient failures by default.
Configure these through `rfork_lease_release_max_attempts` and
`rfork_lease_release_retry_interval_sec` in `--model-loader-extra-config`.
The synchronous lease-release helper uses the same settings; seed removal retains
its independent internal retry policy. Seed heartbeat waits are configurable through
`rfork_heartbeat_interval_sec` (default 30 seconds), without adding lease renewal.
Release HTTP calls execute outside the session lock. Permanent rejection stops the
worker immediately. Retry exhaustion retains an unresolved lease and suppresses seed
promotion; it does not claim release success or fail a successfully transferred model.
Shutdown signals cancellation without waiting for release I/O, and retains resources
until release has been acknowledged. Per-request timeouts remain inactivity limits.
An event-controlled CPU test verifies startup progress and shutdown while release
I/O is blocked. NPU and customer-planner integration remain to be validated.

Before this change, release failure was also a startup-latency problem. After a
successful transfer, the startup thread synchronously called `release_seed`, which
could make three HTTP attempts. The subsequent background retry had no total attempt
or lifetime budget and performed network I/O while holding the session RLock.
`start_seed_service` needs the same lock, so a background network stall can delay the
startup thread before it even returns DEFERRED. Requests connect/read timeouts are
not a strict end-to-end deadline. Fast 400 responses alone do not prove a permanent
startup deadlock: the supplied log already shows several workers entering draft
loading after release failure. Diagnose the live worker stacks to distinguish lock
contention from draft loading, collective waits, or other initialization stalls.

Prioritize separating release network I/O from the session lock and giving retries a
bounded recovery policy. Use immutable lease identity plus an in-flight release guard
to prevent duplicate submissions; reconcile a result only against that same lease.
Keep startup progress independent of source lease bookkeeping and optional new-seed
advertisement once reads have completed. Persistent rejection should remain visible
as an unresolved release, not become either endless work or fabricated success.
Classify transient failures separately from permanent protocol errors; do not accept
all 400 responses as successful release. Preserve the existing memory-ownership and
shutdown safeguards. Add event-controlled tests that hold a fake HTTP release open
while checking startup progress, concurrent release/shutdown, and retry exhaustion.

## Proposed renewal protocol

Renewal is a proposed extension, not an endpoint that the existing planner supports.
Negotiate an explicit protocol capability and return the granted lease duration with
`GET /get_seed`. Only start renewal for an advertised capability. Legacy planners
continue to use fixed TTL; an unknown HTTP 404 is not enough to distinguish an absent
renewal route from a missing lease.

Use a proposed `POST /renew_seed` with USER_ID and the complete source identity
(SEED_KEY, SEED_IP, SEED_PORT, SEED_RANK). In one atomic planner operation, validate
identity and expiry, then extend that existing lease from server time. Renewal must
never allocate another capacity point, revive an expired lease, or recreate a released
lease. Return a granted duration on success and structured, distinct errors for missing
lease, identity mismatch, invalid input, and unsupported protocol. A delayed duplicate
release must not decrement capacity belonging to a later lease.

Use a monotonic clock for durations within one process; persisted/shared leases need
a consistent expiry authority. Record acquisition time separately from last renewal
and expiry. Optionally enforce a configured maximum lifetime to avoid indefinitely
renewing a stuck client. Seed heartbeats describe the source's health; destination
lease renewal describes an active reader. Neither should refresh the other implicitly.

For a granted 60-second TTL, an interval around 15-20 seconds leaves retry headroom.
Use the advertised TTL to derive interval and HTTP timeout with a safety margin;
client deadlines should conservatively account for request latency. Retry transient
failures within the known remaining lifetime, not indefinitely beyond expiry.

The current session holds its RLock throughout registration and transfer. A renewal
thread using that same lock cannot run when needed. It needs an immutable lease
snapshot, its own cancellation/event state, and synchronization independent of the
long-running transfer. Blocking native calls may also retain the Python GIL; verify
that renewal actually runs during NPU registration and reads before trusting a thread.

Serialize renewal with final release. Stop and drain renewal before returning the
lease; avoid joining a thread that needs a lock held by the joiner. Apply the same
ordering to fallback, shutdown, retries, and deferred seed promotion. After successful
transfer, release retries need not renew solely to hold capacity, but seed promotion
must use a verified release or authoritative expiry outcome.

When renewal fails beyond expiry, prohibit starting new reads under that lease. An
in-flight TransferEngine read may not be cancellable: do not free/unregister its
buffers or destroy the source merely because a scheduling lease expired. Wait for
completion or a supported cancellation handshake. Planner capacity accounting alone
does not fence active readers; source-side read admission, fencing tokens or explicit
draining are needed for strict concurrency and memory-safety guarantees.

Validate renewal/GC races, lost responses, renewal/release races, stale USER_ID replay,
planner restart, multi-replica routing, unavailable network, and shutdown during a
blocked read. A planner using per-process state must remain single-process/single-replica
or gain shared atomic state; renewal does not solve cross-replica state loss.

## Rollout and measurements

1. Run the example at 60 seconds with deterministic expiry tests. Record acquisition,
   preparation, registration, transfer and release times separately; record bounded,
   sanitized release error bodies and correlate lease IDs with planner logs.
2. Inspect the deployed Go reclaim/health-check/get/put implementations and actual
   failed responses. Do not treat every 400 as released, or infer a lease TTL from a
   seed timeout name. Even 404 requires an agreed protocol meaning.
3. Benchmark the implemented late acquisition after initialization/layout preparation,
   including the seed-miss cleanup regression cases. Compare actual lease duration
   with 60 seconds under representative load. Retain a configurable larger TTL when
   the observed tail duration requires it.
4. Defer renewal while the customer planner cannot be changed. When server support
   becomes possible, add negotiated renewal to example and client together, then
   integrate with the Go planner. Test NPU progress and reader lifetime before rollout.

The current direction is bounded asynchronous release plus late acquisition, without
client-only renewal. Negotiated renewal remains a future option. A configurable fixed
TTL remains necessary for compatibility and recovery; it is not proof that the
reported deployment's release failures were caused by expiry.
