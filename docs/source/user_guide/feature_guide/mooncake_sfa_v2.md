# Mooncake V2 with SFA KV offload

On a supported Atlas A3 sparse-attention deployment, the decode node can use
`MooncakeSFAConnector` with ModelRunner V2 while sharing the existing
`SparseKVOffloadManager` and LIM + Copy-SFA attention backend.

The implementation lives in `kv_p2p/mooncake_sfa/` and inherits the common
Mooncake Pull connector, scheduler, worker and receiving-thread lifecycle.
Host cache projection, row reservations, staging and first-decode handoff
belong to the specialization. The common `kv_p2p/mooncake/` package exposes
protected extension points and does not import the SFA specialization.

Existing `MooncakeConnectorV2` configurations remain supported. An outer
compatibility entry selects the SFA scheduler and worker when sparse KV
offload is enabled, and the ordinary Mooncake implementations otherwise.
`MooncakePullConnector` remains the explicit ordinary Pull entry. Selection
happens before components are constructed or background threads start.

Use ModelRunner V2 on both nodes:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

The prefill node uses `MooncakeConnectorV2` with `kv_role="kv_producer"` and
device-resident KV. On the decode node, use the following configuration in
addition to the deployment's model, tensor parallel and serving options:

```bash
--kv-transfer-config '{"kv_connector":"MooncakeSFAConnector","kv_role":"kv_consumer","kv_port":27300}' \
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

After manager cache registration, `get_copy_sfa_cache_views()` exports immutable
layout metadata referencing the same Host and hot/tail tensors, including the
restored Host aliases on non-owner TP ranks. The adapter obtains this snapshot
once during connector registration. This interface allocates no KV storage,
performs no device copy or synchronization, and leaves the existing PD addresses,
transfer ranges, partitioning and staging execution unchanged. The manager owns
the borrowed tensors for the worker lifetime; callers must not resize them.

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

The existing SFA connector retains its protocol, scheduling, allocation and
attention operators. Its optional Mooncake backend keeps its own bounded
staging implementation; MemFabric remains its default backend.

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

Configure this combination directly as `MooncakeSFAConnector`, or retain the
compatible `MooncakeConnectorV2` plus sparse offload configuration. Wrapping it in
`MultiConnector` is rejected until sparse slot and first-decode hooks are
forwarded. DSpark's remote prompt/draft-KV protocol is also rejected for this
combination. Decode consumers require `kv_load_failure_policy="fail"` (the
default) for both Host-only and `keep_device_kv_cache=true` colocate debugging
modes. Recovery from a failed remote PD transfer to local prefill has not been
adapted. These checks do not change the existing SFA connector's configuration
or flow.
