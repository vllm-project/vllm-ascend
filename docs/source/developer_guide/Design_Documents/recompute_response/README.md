# PD recomputation errors

Ascend's recomputation scheduler can end a Decode attempt with `STOP` with the internal `RECOMPUTE_STOP_REASON` marker
when KV offload is unavailable. The API reports this interruption as
`RecomputeRequiredError`. A non-streaming response uses HTTP 500; an already-started
stream carries an error event followed by one `[DONE]`.

The error includes `partial_response`, the original terminal response with its
partial output, usage and choice fields. The standard and layerwise Python PD
proxy examples consume this payload and continue through their existing retry
path. They recognize the structured error type; ordinary stop strings remain
normal completions. Deploy the scheduler, serving patch and proxies as a matched
version. The previous string-based continuation protocol is not accepted.

The patch is installed by the existing Ascend platform-patch loader. Chat uses
the typed raw response generator where available. Completion inspects complete
native serving events before HTTP fragmentation; normal event strings are
forwarded unchanged. No network SSE parser or new model setting is introduced.

The marker uses a negative integer, distinct from user stop strings and valid
nonnegative stop token IDs. The scheduler sends it without new tokens, so the
output processor does not rerun stop-string matching. API conversion consumes
this marker before a normal terminal response can be returned. The proxy's
prompt reconstruction algorithm remains unchanged.

For synchronous cleanup of multi-prompt Completion requests, the paired vLLM
must include the iterator-cancellation fix in vLLM PR #50830 (or equivalent).
That prerequisite waits for cancelled reads before closing their generators.

Native Router HTTP/SSE error forwarding supports this envelope without the Router compatibility patch (#397). Deployments with the old backend terminal protocol still need a proxy
that understands it or the Router compatibility patch.
