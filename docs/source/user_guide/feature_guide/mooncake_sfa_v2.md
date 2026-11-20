# Mooncake V2 with SFA KV offload

On a supported Atlas A3 sparse-attention deployment, the decode node can use
`MooncakeConnectorV2` with ModelRunner V2 while sharing the existing
`SparseKVOffloadManager` and LIM + Copy-SFA attention backend.

The implementation lives in `kv_p2p/mooncake_sfa/` and inherits the common
Mooncake Pull connector, scheduler, worker and receiving-thread lifecycle.
Host cache projection, row reservations, staging and first-decode handoff
belong to the specialization. `MooncakeConnectorV2` resolves to the compatibility
entry in `mooncake_sfa/connector.py`, which lazily selects the SFA scheduler and
worker when sparse KV offload is enabled, and the ordinary Mooncake implementations
otherwise. The common `kv_p2p/mooncake/` package keeps the shared address planning,
registration and thread lifecycle without importing the SFA specialization.
SFA overrides cache registration, bucket execution and receive completion handling;
its receiver drains pending Host copies before returning to the common completion
loop, including on failure. `MooncakeConnectorV2` remains the only Mooncake
configuration entry for sparse KV offload. `MooncakePullConnector` continues to
select ordinary transfers; enabling sparse KV offload with that name is rejected
with a message to use `MooncakeConnectorV2`.
Selection happens before components are constructed or background threads start.

When sparse KV offload is disabled, the compatibility entry delegates to the
upstream V2 factory: `kv_connector_extra_config.use_layerwise=true` selects
Mooncake Push, while the default selects Mooncake Pull. SFA KV offload supports
Pull only. Leave `use_layerwise` unset or set it to `false` on both nodes;
combining enabled sparse KV offload with `use_layerwise=true` fails at startup.

Use ModelRunner V2 on both nodes:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

The prefill node uses `MooncakeConnectorV2` with `kv_role="kv_producer"` and
device-resident KV. On the decode node, use the following configuration in
addition to the deployment's model, tensor parallel and serving options:

```bash
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":27300}' \
--additional-config '{"sparse_kv_offload_config":{"enabled":true,"topk_buffer_size":2048,"dram_size_per_dp_GB":16,"use_fused_overlap":false}}'
```

Choose the Host capacity and hot-row budget for the deployment. The hot-row
budget must satisfy the existing LIM + Copy-SFA configuration requirements.
The independent DSA Host allocator and TP-shared pool have been removed.
Remove the retired `dsa_sparse_attention_config` section; nonempty values
fail at startup with a migration message, including configurations that do
not enable Manager offload. The Manager configuration above is the only
serving Host KV owner for this path. Ordinary device-resident Mooncake and
the existing SFA connector keep their configuration.

The V2 connector publishes persistent Host main KV and HBM indexer/scale
components. SFA device placeholders are not PD payload. Main blocks are
partitioned across decode TP ranks into one shared Host pool; the indexer
remains device-resident and replicated as configured.

After manager cache registration, the adapter directly borrows the manager's
existing Host and hot/tail tensors once during connector registration, including
the restored Host aliases on non-owner TP ranks. It validates LIM + Copy-SFA,
complete registration, the main KV group and layer-name/index correspondence.
This allocates no KV storage and performs no device copy or synchronization.
The manager owns the borrowed tensors for the worker lifetime; callers must
not resize them.

The D worker also registers its existing rank-private HBM hot/tail buffers.
For each request, V2 appends direct producer-to-HBM READs: a short prefix is
placed densely in the reserved hot row; a long prefix's incomplete last block
is placed at `hot_tokens + (logical_block % 2) * block_size` in the existing
two-block circular tail. An aligned long prefix requires no tail READ. These
HBM reads are replicated on every D TP rank, independently of shared Host block
ownership. Like the existing SFA scheduler, placement uses the entire received
KV extent (`computed + external`, bounded by the producer's valid KV count),
without subtracting a replayed prompt token.

P and D must receive the same original prompt. Discard P's first output token;
D replays the final prompt token and generates the complete output, as with the
existing SFA connector. Appending P's first output token to D's prompt is rejected
before receiving KV. Ordinary Mooncake transfers retain their existing
first-token behavior.

The existing V2 batch READ and bounded staging pipeline still land Host main
KV through the SFA manager's local-copy API. Receive completion includes both
direct HBM reads and the final Host staging drain. After all TP workers complete
PD, the first real decode validates the reservation and expected execution
prefix. The scheduler replays the last prompt token; this does not change
PD placement. If a long received extent is block-aligned,
PD has no incomplete tail but that forward needs the preceding
partial block. The V2 adapter restores that history once through the existing
manager tail-copy API before eager execution or graph replay. Other remote
receives do not repeat Host-to-hot initialization. Complete local prefix hits
with no remote PD and prefix rollback retain the existing local recovery path.

The adapter binds transport metadata before attention preparation and ignores
the ordered load hook's repeated bind of that same metadata object. Successful
initialization retires its one-time PD state only after required copies and tail
metadata checks succeed. Failed receives cannot commit initialization. Existing
SFA request-state tracking continues to own decode prefix history and LIM reset;
the connector owns transport completion, ACK and in-flight cancellation.
The model-state preparation calls a connector hook and its completion callback;
the adapter keeps PD state, main-group selection and merged dense-row fills
internal. The original SFA connector and shared request-state rules are unchanged.

The existing SFA connector retains its protocol, scheduling, allocation and
attention operators. It continues to use MemFabric; this Mooncake V2 integration
does not add a Mooncake backend to that connector.

Request admission and stable hot-row reservations reuse the SFA slot allocator.
Finished requests release their reservations. A cancelled in-flight PD retains
its row until all D workers finish their reads, preventing late HBM writes from
corrupting a reused row. A failed receive cannot
publish a ready hot row. Subsequent decode uses the existing SFA metadata,
LIM, Copy-SFA, hot-cache management and current-token Host writeback.

This combination retains existing SFA offload restrictions, including BF16 main
KV, supported device-resident indexer formats, and no context/pipeline parallel
offload. It selects LIM + Copy-SFA; `use_fused_overlap=true` is rejected for this
V2 combination. SFA offload uses native preprocessing; this path retains
the existing hot-buffer memory and direct HBM hot/tail PD reads.

Configure this combination as `MooncakeConnectorV2` plus sparse offload. Wrapping it in
`MultiConnector` is rejected until sparse slot and first-decode hooks are
forwarded. DSpark's remote prompt/draft-KV protocol is also rejected for this
combination. Decode consumers require `kv_load_failure_policy="fail"` (the
default) for both Host-only and `keep_device_kv_cache=true` colocate debugging
modes. Recovery from a failed remote PD transfer to local prefill has not been
adapted. These checks do not change the existing SFA connector's configuration
or flow.

## Staging settings

Set `mooncake_staging_config` through `--additional-config` on the decode node.
The two typed integer settings are parsed once at startup; they do not use
environment variables. Existing sparse-offload options remain in the same JSON:

```bash
--additional-config '{"sparse_kv_offload_config":{"enabled":true,"topk_buffer_size":2048,"dram_size_per_dp_GB":16,"use_fused_overlap":false},"mooncake_staging_config":{"window_chunks":16,"bank_count":4}}'
```

| Setting | Default | Meaning and bounds |
| --- | --- | --- |
| `window_chunks` | `16` | Total reusable 2 MiB-aligned staging chunks. Integer values below 1 become 1. Chunk byte size depends on the KV layout. |
| `bank_count` | `4` | Number of banks sharing the window. Integer values below 1 become 1; the effective count is capped by `window_chunks`. One effective bank uses serial READ/landing; multiple banks enable pipelining. |

Defaults use 16 chunks split into four banks of four chunks. Each bank uses one
copy stream and, when there are multiple banks, one Host worker. Set
`bank_count=1` to use the entire window on the caller thread with synchronous
landing. No separate pipeline switch, per-bank copy-stream setting or READ probe
is exposed. Previously used staging environment variables are ignored; replace
them with the two additional-config fields when updating deployment scripts.

Banks share the existing allocation without increasing total staging capacity.
Each bank has `floor(window_chunks / effective_banks)` contiguous slots; any
remainder is unused. Consecutive READ batches rotate between banks, reducing
maximum chunks per batch to that bank's slot count. A completion event must finish
before a bank's slots can be reused. Workers join and all pending landing copies
drain before receive completion and release of the shared staging lock.

Overlapping Host destination ranges are rejected during planning before any READ
in that receive, including direct HBM READs in a mixed receive. Failed waits retain
the pending buffer owners until draining succeeds. Removing the READ diagnostic
probe does not remove these completion and failure-lifetime guarantees.

## Common extension boundaries

The common Pull receiver retains its ordinary address planner and submits
ranges directly to the transfer engine. The SFA receiver inherits the request
queue, remote metadata handling, block mapping, executor task aggregation,
failure reporting and completion loop. It specializes address planning and
bucket execution to partition shared Host destinations, retain target identities,
append hot/tail reads and execute through the staging reader.

The remaining shared interfaces have the following responsibilities:

| Interface | Shared responsibility | SFA responsibility |
| --- | --- | --- |
| `_create_scheduler()` / `_create_worker()` | Construct the selected component before it starts background work. | Select the SFA scheduler and worker. |
| `_init_transfer_engine()` | Initialize the default Ascend engine; allow a specialization to select another transport. | Use the default engine. |
| `_register_transfer_buffers()` | Register physical storage independently of published cache addresses. | Exclude Host destinations and include existing device hot/tail storage. |
| `_create_recving_thread()` / `_build_recving_thread()` | Construct the selected receiver with common arguments; retain startup and readiness checks in the public worker. | Create the staging reader and pass it to the SFA receiver. |

The SFA scheduler initializes its slot policy before entering the parent
constructor. It overrides the existing `build_connector_meta()` method, validates
slot reservations before the parent consumes its pending queue, and converts
the public result into SFA metadata. Conversion preserves public fields and
adds `copy_sfa_slot`, including slot zero. No metadata factory or policy
initialization hook is needed in the public scheduler.

Target identities and hot/tail request context are local SFA planning data,
not additions to the P/D wire handshake or public bucket signature. The receiver
keeps the current batch available to executor tasks until they have joined and
all landing work has drained, then clears the context before the next batch.
This depends on the inherited receiver processing one batch at a time; overlapping
batches would require explicit per-task context instead.

The common receiver has no SFA block-group, target-view, additional-read or
landing hooks. The SFA address planner and bucket therefore contain some
ordinary planning and execution code that must be reviewed when the public
implementation changes. Physical registration rules, shared Host ownership,
device destinations and landing-before-completion ordering are retained.
