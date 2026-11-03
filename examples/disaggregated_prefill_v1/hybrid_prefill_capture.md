# Reference: capture unexpected prefill on Hybrid decode nodes

This opt-in diagnostic captures requests when a dedicated `kv_consumer` node
schedules prompt computation with more than one prompt token still outstanding.
It excludes normal last-prompt-token replay and decode. It also catches
one-token chunks of a longer prefill and recomputation after a valid KV handoff.
The event proves scheduling, not completed device execution.

## Enable on D

Merge these settings into the Mooncake Hybrid connector's
`kv_connector_extra_config`. With MultiConnector, use the Mooncake child
configuration, preserving the other connectors and options:

```json
{
  "log_missing_kv_params": true,
  "prefill_capture_dir": "/path/to/private/captures"
}
```

Both options are disabled by default and apply only to `kv_consumer`.
The logging option diagnoses missing KV parameters at admission. The capture
option independently watches all requests, including requests with KV metadata.
Intentional direct requests to D without a P/D handoff also trigger diagnostics.

To retain the original HTTP body, add this option to the D server:

```bash
--middleware vllm_ascend.distributed.kv_transfer.kv_p2p.hybrid_prefill_capture.CaptureRequestBodyMiddleware
```

The middleware uses the existing `vllm_xargs` extension mechanism for
`/v1/messages`, `/v1/chat/completions`, and `/v1/completions`. It carries the
original body in memory without changing sampling controls. Install it only
when capture is enabled; it increases request serialization and memory use.

## Inspect a capture

Each successfully saved request has a unique `capture-<uuid>` directory:

- `request.json`: the original UTF-8 HTTP body received by D, byte for byte.
  Proxy-side modifications are already present in this body.
- `engine.json`: token IDs, sampling parameters, original admission flags,
  current KV parameters, preemption and scheduling counters, and endpoint.

Each request is submitted for capture at most once. The normal final-token
replay is excluded even though it computes one prompt token. A later preemption
can still trigger capture. Without the middleware, only `engine.json` is saved
and `original_request_available` is false.

Logs contain request IDs, counts, availability status, and file paths, not full
request bodies or token lists. The connector clears `do_remote_prefill` after
registering a receive; a later false value does not prove missing admission
metadata. Local or pooled cache hits can also avoid full recomputation.

The scheduler copies CPU request state. A background thread writes files using
a queue bounded to 16 snapshots and renames each completed directory into
place. Queue saturation, serialization errors, and write failures are reported
without failing generation. Pending captures can be lost if the process exits.
Full prompts can contain sensitive data: use a private directory, restrict
access, and remove captures when the investigation is complete.

This is a diagnostic reference, not a KV-transfer recovery mechanism. Stored
remote KV addresses can expire; a P/D replay must obtain a fresh handoff.
Performance overhead and real-service behavior require deployment validation.
