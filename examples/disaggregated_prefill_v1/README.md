# Load balance proxy session affinity

The `load_balance_proxy_server_example.py` proxy can keep a multi-turn
conversation on the same prefill endpoint to reuse its prefix cache. Enable
this optional policy with `--enable-prefill-affinity`:

```bash
python load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 9000 --workers 2 \
  --prefiller-hosts 127.0.0.1 127.0.0.1 \
  --prefiller-ports 8100 8101 \
  --decoder-hosts 127.0.0.1 127.0.0.1 \
  --decoder-ports 8200 8201 \
  --enable-prefill-affinity
```

Without the flag, the existing load-based scheduling policy remains unchanged.
Decode scheduling remains load based with either policy.

## Conversation identity

When affinity is enabled, the proxy identifies the conversation in this order:

1. The `X-Session-Id` HTTP header, if nonempty. It overrides the forwarded
   request body's `user` field.
2. A nonempty string in the OpenAI-compatible `user` field.
3. The first `user` message and any preceding `system` or `developer` messages.

The model name is included in every identity. The fallback uses canonical JSON,
including structured message content; later messages and sampling parameters
do not affect it. Requests without an identity use load-based scheduling.
For `/v1/completions`, provide an explicit session ID to enable affinity.

Agent clients should send a stable ID for each task, agent branch and sample:

```bash
curl http://localhost:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-Id: run-123:task-456:agent-0:sample-0' \
  -d '{"model":"your-model","messages":[{"role":"user","content":"Fix this bug"}]}'
```

Keep the ID unchanged across tool calls, subsequent turns and request retries.
Use a different ID for independent tasks, branches or samples. Prefer explicit
IDs if the agent compacts history or rewrites its initial instructions. Do not
use one shared `user` value for every conversation: it would pin them all to
one endpoint. Without explicit IDs, identical initial messages and instructions
share an affinity key.

## Routing behavior and limits

The proxy uses SHA256 rendezvous hashing over the conversation key and each
registered prefill `host:port`, selecting the highest score. Register individual
DP rank endpoints to obtain rank-level affinity; a load balancer hiding multiple
ranks behind one endpoint cannot provide this guarantee. Selection is stable
across proxy workers and restarts while the endpoint set is unchanged. Removing
an endpoint remaps only its conversations; adding one may attract conversations
from existing endpoints. Endpoints marked for removal are excluded immediately.
If no eligible prefill endpoints remain, assignment fails.

Affinity does not move a conversation away from a busy prefill endpoint, so it
can trade load balance for cache reuse. It does not reserve cache space or
guarantee hits after eviction, service restart or prompt-prefix changes. Decode
requests still use the existing KV-transfer flow and load-based scheduler.
