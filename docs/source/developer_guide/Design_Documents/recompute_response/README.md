# PD recomputation errors

Ascend's recomputation scheduler can end a Decode attempt with `STOP/recomputed`
when KV offload is unavailable. The API reports this interruption as
`RecomputeRequiredError`. A non-streaming response uses HTTP 500; an already-started
stream carries an error event followed by one `[DONE]`.

The error includes `partial_response`, the original terminal response with its
partial output, usage and choice fields. The standard and layerwise Python PD
proxy examples consume this payload and continue through their existing retry
path. They also accept the original `recomputed` terminal response. Upgrade
these proxies before upgrading a backend that emits the structured error.

The patch is installed by the existing Ascend platform-patch loader. Chat uses
the typed raw response generator where available. Completion inspects complete
native serving events before HTTP fragmentation; normal event strings are
forwarded unchanged. No network SSE parser or new model setting is introduced.

The internal scheduler signal and the proxy's prompt reconstruction algorithm
remain unchanged. The backend API conversion preserves explicit user stop strings named
`recomputed`. The proxies retain their legacy continuation interpretation of
that same string.

For synchronous cleanup of multi-prompt Completion requests, the paired vLLM
must include the iterator-cancellation fix in vLLM PR #50830 (or equivalent).
That prerequisite waits for cancelled reads before closing their generators.

Native Router HTTP/SSE error forwarding supports this envelope without the Router compatibility patch (#397). Deployments with the old backend terminal protocol still need a proxy
that understands it or the Router compatibility patch.
