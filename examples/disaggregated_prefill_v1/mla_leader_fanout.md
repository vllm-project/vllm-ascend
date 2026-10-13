# MLA leader fanout for MooncakeHybridConnector

Use `MooncakeHybridConnector` on both P and D; the feature is not wired into
`MooncakeConnectorV1` or `MooncakeConnectorV2`.

The hybrid Mooncake receive path can pull replicated MLA KV from prefill once
per request, then let the other decode TP ranks read the leader's registered
buffers through Mooncake. This reduces redundant cross-node reads for hybrid
DeepSeek MLA caches. The default receive path is unchanged unless
`leader_fanout` is present in `kv_connector_extra_config` on decode.

## Configuration

Add the following object alongside the existing `prefill` and `decode`
configuration entries on the decode engine:

```json
{
  "leader_fanout": {
    "shared_dir": "/dev/shm/vllm-mla-fanout",
    "session_id": "unique-decode-startup-id",
    "leader_rank": 0,
    "ready_timeout": 30,
    "ack_timeout": 60
  }
}
```

All ranks in one decode TP group must have the same configuration. Use a fresh
`session_id` for every engine startup and a different one for each decode DP
replica. The shared directory must refer to the same **local filesystem** on
all decode ranks and support POSIX `flock`. Mount it into each worker container
when workers have separate filesystem namespaces. Network filesystems are not
supported. Session IDs are hashed before being used as directory names.

All ranks must run on the same host with compatible monotonic clock namespaces.
`pcp_size` checks PCP; `nnodes_within_dp` separately checks the single-node
constraint. A shared mount alone does not make a multi-node deployment supported.

This implementation requires hybrid MLA with replicated KV, decode TP > 1,
single-node decode TP, and PP=1 on both sides. Mamba and non-MLA caches are
rejected when the feature is configured. Existing prefill PCP replica selection
is preserved. Transfer layout lengths, strides, group memberships and address
counts are checked before a local read; incompatible layouts fall back to P.

`leader_rank` defaults to 0. `ready_timeout` and `ack_timeout` default to 30 and
60 seconds respectively and must be finite positive numbers. Remove
`leader_fanout` to restore per-rank P reads. No new environment variables or
additional TCP metadata ports are required.

## Data and lifetime protocol

The leader completes its synchronous P read, then atomically publishes its TE
host/port, registered transfer bases, layout and the actual local block table.
Readers map the required P logical block IDs to the leader's received block IDs
before constructing transfer descriptors. A reader with a shorter prefix hit
can use the corresponding suffix; a reader needing blocks absent from the
leader falls back to P. Empty requests participate in ACK without reading data.
Block remapping never mutates the original request's block lists or P completion
destinations.

The D scheduler assigns a fresh `transfer_attempt` UUID to each newly built
request transfer entry. Connector metadata broadcasts this UUID to all TP workers;
workers forward it to their receive queues. Ready/ACK paths include this attempt,
and readers validate the attempt in the publication. A retry must build a new
entry rather than replay old metadata. Missing attempt metadata falls back to P.

Each reader holds a shared filesystem lock from inspecting ready metadata until
its synchronous TE read returns. It ACKs success or an exception before falling
back. Readers that time out before publication send an early ACK and read from
P. The leader waits for ACKs, then takes the exclusive lock and removes the
publication before allowing the request to finish. After its ACK deadline,
missing readers are no longer accepted. Already active readers continue to pin
the leader's blocks until TE returns, even past that deadline. A permanently
hung TE call therefore requires worker recovery; silently releasing its source
blocks would be unsafe.

Contended lock acquisition logs `MLA fanout pin blocked` every five seconds,
including the key, rank and elapsed wait. This also covers unrelated requests
sharing a lock stripe. Polling starts at 1 ms and backs off to 10 ms, bounded by
the remaining ready/ACK deadline; it is still polling, not event notification.

`MLAFanout.stats()` returns per-worker cumulative `leader_pull`, `local_read`,
`fallback`, reason-specific `fallback_*`, `ack_timeout` and `pin_wait_warnings`
counters. These are process-local diagnostics, not exported Prometheus metrics.
Successful transfer events appear at DEBUG level; fallback logs include reason
and a counter snapshot. The fallback reasons are missing attempt, ready timeout,
expired publication and local read error (including layout/coverage/TE failures).

Normal completion removes per-request metadata and ACKs. At most 4096 lock
files remain per session; they must not be unlinked while workers are alive.
A crashed leader or an early ACK without a leader may leave request artifacts.
Remove an old session directory only after all its workers have stopped.
The fresh startup session prevents old ready/ACK state from being reused.
Attempt IDs must identify unique transfers within a session and must never be
reused across retries. Lock pinning protects active reads; attempt isolation
protects publication/ACK state, not scheduler ownership of reused KV blocks.

The pin delays connector receive completion; it does not introduce a separate
scheduler cancellation protocol. Worker crashes and scheduler abort/preemption
still need end-to-end validation against the deployment's existing block
lifecycle. Configure P-side abort retention to cover transfer and rendezvous
latency, as for the normal Mooncake path.

## Correctness diagnostics

The test-only `registered_page` helper in
`tests/ut/kv_offload/mla_fanout_test_utils.py` exposes an unsigned-byte tensor
view using `registered_base + block_id * block_stride`. Callers supply an owned
backing tensor, stride and transfer length. It checks storage bounds and does
not read NPU addresses with `ctypes`. Production workers retain no extra
registration state for this diagnostic helper.

Use this view for hashes while the worker still owns the request's blocks.
Layer tensor indexing can start at an inner page offset and does not necessarily
represent the bytes that Mooncake transfers. The method returns physical page
bytes, which can include multiple shared layer views and padding; compare the
same registered page between peers. Hashing/copying these views is an explicit
diagnostic operation and is absent from the transfer hot path.

## Validation

Run the focused tests in an environment with the repository's UT dependencies:

```shell
pytest -sv tests/ut/kv_offload/test_mla_fanout.py \
  tests/ut/kv_offload/test_mooncake_hybrid_connector.py
```

The tests cover one P read for eight ranks, block remapping and divergent prefix
coverage, TE/metadata errors, startup isolation, ready/ACK timeouts, late readers,
actual cross-process locking and an active reader outliving the ACK deadline.
Retry tests cover stale ready/ACK files, overlapping attempts, scheduler UUID
creation/broadcast and forwarding through both worker queue branches.
The P-read count assertion provides a regression check for redundant transfers.

On an Ascend PD deployment, compare the default path and fanout with identical
model, prompt seeds and output lengths. Run 32K, 128K and 200K inputs at
concurrency 1, 2, 4 and 8, at least three measured rounds after warmup, alternating
modes. Record TTFT, TPOT, fallback logs and cross-node traffic. Check greedy
outputs and long-context retrieval, prefix hits, cancellation, reader crashes
and worker restarts. The supplied FragPack prototype reported 128K/concurrency-4
TTFT improving from about 24 seconds to about 7 seconds; this is historical
prototype evidence, **not a measurement of this native implementation**.
