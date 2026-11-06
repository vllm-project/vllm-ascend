# Load balance proxy session affinity

The `load_balance_proxy_server_example.py` proxy can keep a conversation or
prompt-prefix group on the same prefill endpoint to reuse its prefix cache.
Enable this optional behavior with `--enable-prefill-affinity`:

```bash
python load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 9000 --workers 2 \
  --prefiller-hosts 127.0.0.1 127.0.0.1 \
  --prefiller-ports 8100 8101 \
  --decoder-hosts 127.0.0.1 127.0.0.1 \
  --decoder-ports 8200 8201 \
  --enable-prefill-affinity \
  --prefill-affinity-policy group-balanced \
  --prefill-affinity-prefix-chars 256 \
  --prefill-affinity-max-groups 100000
```

Without `--enable-prefill-affinity`, the existing load-based scheduling policy
remains unchanged. The default affinity policy is `rendezvous`, which preserves
the original affinity behavior. Decode scheduling remains load based with
either affinity policy.

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
one endpoint.

### Prefix-based fallback identity

`--prefill-affinity-prefix-chars N` limits the first user message's text to its
leading `N` characters when constructing the fallback identity. Requests whose
prefixes match then share an affinity group even if their suffixes differ. The
limit is measured in characters, not tokens. The default value `0` retains the
full first user message.

This option does not replace an explicit session ID or request `user` value,
and it does not modify the request forwarded to the model. The model and
preceding instructions remain part of the fallback identity. For structured
content, only the first text part is truncated; other content parts are
unchanged.

## Routing behavior and limits

### Rendezvous policy

The default `rendezvous` policy uses SHA256 rendezvous hashing over the affinity
key and each healthy prefill `host:port`, selecting the highest score. Selection
is stable across proxy workers and restarts while the endpoint set is unchanged.
Removing an endpoint remaps only its groups; adding one may attract groups from
existing endpoints. This policy does not retain group-assignment state.

### Group-balanced sticky policy

The opt-in `group-balanced` policy stores assignments in the scheduler shared
by all proxy workers:

- An existing group stays on its current prefiller while that endpoint remains
  healthy and its assignment is retained.
- A new group is assigned to a healthy prefiller with the fewest assigned
  groups. Rendezvous hashing breaks ties deterministically.
- Adding a prefiller does not migrate existing groups. New groups prefer the
  less-occupied endpoint until the group counts become balanced.
- Tainting or removing a prefiller clears only its assignments. Those groups
  are reassigned on their next request; other groups remain sticky.

`--prefill-affinity-max-groups` bounds these assignments with
least-recently-used eviction and defaults to `100000`. Using or evicting a group
only affects future prefix-cache locality; it does not change request
correctness. Group assignments are rebuilt after a proxy restart and may differ
from the previous process.

This is group-count balancing, not dynamic load rebalancing. It does not use
instantaneous QPS, token load or KV-cache pressure, split hotspot groups, or
migrate groups away from a healthy busy prefiller. Register individual DP rank
endpoints to obtain rank-level affinity; a load balancer hiding multiple ranks
behind one endpoint cannot provide this guarantee.

Endpoints marked for removal are excluded immediately. If no eligible prefill
endpoints remain, assignment fails. Affinity does not reserve cache space or
guarantee hits after eviction, backend restart or prompt-prefix changes. Decode
requests still use the existing KV-transfer flow and load-based scheduler.
