# Disaggregated-prefill

## Why disaggregated-prefill?

This feature addresses the need to optimize the **Time Per Output Token (TPOT)** and **Time To First Token (TTFT)** in large-scale inference tasks. The motivation is two-fold:

1. **Adjusting Parallel Strategy and Instance Count for P and D Nodes**  
   Using the disaggregated-prefill strategy, this feature allows the system to flexibly adjust the parallelization strategy (e.g., data parallelism (dp), tensor parallelism (tp), and expert parallelism (ep)) and the instance count for both P (Prefiller) and D (Decoder) nodes. This leads to better system performance tuning, particularly for **TTFT** and **TPOT**.

2. **Optimizing TPOT**
   Without the disaggregated-prefill strategy, prefill tasks are inserted during decoding, which results in inefficiencies and delays. Disaggregated-prefill solves this by allowing for better control over the system's **TPOT**. By managing chunked prefill tasks effectively, the system avoids the challenge of determining the optimal chunk size and provides more reliable control over the time taken for generating output tokens.

---

## Usage

vLLM Ascend supports the following KV cache transfer modes:

- **MooncakeConnector**: D nodes pull KV cache from P nodes.
- **MooncakeLayerwiseConnector**: P nodes push KV cache to D nodes in a layered manner.  
- **MooncakeD2RHConnectorV1**: D nodes pull KV cache into local host memory before loading it into device memory.

For step-by-step deployment and configuration, refer to the following guide:  
[PD disaggregation multi-node deployment guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)

---

## How It Works

### 1. Design Approach

Under the disaggregated-prefill, a global proxy receives external requests, forwarding prefill to P nodes and decode to D nodes; the KV cache (key-value cache) is exchanged between P and D nodes via peer-to-peer (P2P) communication.

### 2. Implementation Design

Our design diagram is shown below, illustrating the pull and push schemes respectively.
![Disaggregated prefill pull](../../assets/disaggregated_prefill_pull.png)
![Disaggregated prefill push](../../assets/disaggregated_prefill_push.png)

#### Mooncake Connector

1. The request is sent to the Proxy's `_handle_completions` endpoint.
2. The Proxy calls `select_prefiller` to choose a P node and forwards the request, configuring `kv_transfer_params` with `do_remote_decode=True`, `max_completion_tokens=1`, and `min_tokens=1`.
3. After the P node's scheduler finishes prefill, `update_from_output` invokes the schedule connector's `request_finished` to defer KV cache release, constructs `kv_transfer_params` with `do_remote_prefill=True`, and returns to the Proxy.
4. The Proxy calls `select_decoder` to choose a D node and forwards the request.
5. On the D node, the scheduler marks the request as `RequestStatus.WAITING_FOR_REMOTE_KVS`, pre-allocates KV cache, calls `kv_connector_no_forward` to pull the remote KV cache, then notifies the P node to release KV cache and proceeds with decoding to return the result.

#### Mooncake Layerwise Connector

1. The request is sent to the Proxy's `_handle_completions` endpoint.
2. The Proxy calls `select_decoder` to choose a D node and forwards the request, configuring `kv_transfer_params` with `do_remote_prefill=True` and setting the `metaserver` endpoint.
3. On the D node, the scheduler uses `kv_transfer_params` to mark the request as `RequestStatus.WAITING_FOR_REMOTE_KVS`, pre-allocates KV cache, then calls `kv_connector_no_forward` to send a request to the metaserver and waits for the KV cache transfer to complete.
4. The Proxy's `metaserver` endpoint receives the request, calls `select_prefiller` to choose a P node, and forwards it with `kv_transfer_params` set to `do_remote_decode=True`, `max_completion_tokens=1`, and `min_tokens=1`.
5. During processing, the P node's scheduler pushes KV cache layer-wise; once all layers pushing is complete, it releases the request and notifies the D node to begin decoding.
6. The D node performs decoding and returns the result.

##### Sparse SFA C8 cache transfer

For SFA models, `MooncakeLayerwiseConnector` can transfer the contiguous packed C8 main cache directly. Attention writes the quantized NoPE data, BF16 RoPE data and dynamic scales before recording the cache-write event. The connector waits for that event and copies the bytes without applying the legacy K/V INT8 quantization again. Indexer K and its scale tensor are transferred as separate components of the same physical layer.

For FP8 main and indexer caches, configure both P and D with `--kv-cache-dtype fp8 --attention_config.indexer_kv_dtype fp8`. Both ends must use matching cache layouts. This path requires main and indexer components to share one KV cache group with the same block-size scale. Noncontiguous packed main caches, TurboQuant, and DCP-replicated indexer layouts with different component block-size scales remain unsupported.

This transfer path still requires end-to-end validation on Ascend hardware, including configurations that combine it with `AscendStoreConnector` through `MultiConnector`.

#### Mooncake D2RH Connector

D2RH (device-to-remote-host) splits KV cache transfer into two stages:
**P HBM → D DRAM → D HBM**. Once the first stage completes, P can release the
transferred KV cache without waiting for D's host-to-device (H2D) copy or decoding.

![D2RH KV cache transfer workflow](../../assets/disaggregated_prefill_d2rh.png)

The pull arrows in the diagram represent requests, not the direction of KV data.

1. **Route the request (1–3):** The proxy selects P, waits for prefill to finish,
   and forwards the request and KV transfer metadata to D.
2. **Stage KV cache (4–6):** D's scheduler asks its workers to pull the required
   KV cache from P HBM into D DRAM. The workers notify P when the first-hop reads
   complete, allowing P to release the corresponding cache.
3. **Load and decode (7–10):** The workers notify D's scheduler that staging is
   ready. D allocates the required device blocks and loads the staged KV cache
   into HBM. Decoding starts only after the required H2D transfers complete.

Valid host-cache hits avoid repeated P-to-D transfers. Valid D-local HBM prefix
hits can also reduce H2D copies; a host-cache hit alone is not an HBM hit. Host
blocks remain pinned until all H2D reads using them finish.

To enable this mode, set `"kv_connector": "MooncakeD2RHConnectorV1"` in
`--kv-transfer-config` on both nodes, with `"kv_role": "kv_producer"` on P and
`"kv_role": "kv_consumer"` on D. Keep the `prefill` and `decode` parallelism
settings consistent with the deployment and reserve enough host memory on D
for staging.

For startup scripts and a proxy example, see
[D2RH deployment on Atlas 850E](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md#d2rh-deployment-on-atlas-850e-uboe).

### 3. Interface Design

Taking MooncakeConnector as an example, the system is organized into three primary classes:

- **MooncakeConnector**: Base class that provides core interfaces.
- **MooncakeConnectorScheduler**: Interface for scheduling the connectors within the engine core, responsible for managing KV cache transfer requirements and completion.
- **MooncakeConnectorWorker**: Interface for managing KV cache registration and transfer in worker processes.

#### V1 Mooncake internal layering

The V1 Mooncake connectors keep their registration names and user-facing configuration, but their internals are split into a shared `mooncake_v1` package so that the wire protocol, transport helpers and pure-computation helpers no longer live inside the two engine-facing modules and can be unit-tested on their own.

| Module | Responsibility |
|---|---|
| `kv_p2p/mooncake_v1/common/protocol.py` | Wire message constants (`GET_META_MSG`, `DONE_RECVING_MSG`, `DONE_SENDING_MSG`, `FAILED_SENDING_MSG`) and the shared metadata structures `PullAgentMetadata`, `LayerwiseAgentMetadata`, `LayerMetadata`, `RemotePortInfo`. Field names, field order and defaults are unchanged. |
| `kv_p2p/mooncake_v1/common/tracker.py` | `KVCacheTaskTracker`: finished-request set consumed by the scheduler plus the delayed-free queue that bounds how long the prefill node keeps KV blocks alive after an aborted request. |
| `kv_p2p/mooncake_v1/common/transport.py` | `SizedDict`, ZMQ helpers (`zmq_ctx`, `ensure_zmq_send`, `ensure_zmq_recv`, `group_concurrent_contiguous`, `split_if_not_byte_contiguous`, `string_to_int64_hash`) and the `RemoteSocketPool` with its idempotent `close()`. |
| `kv_p2p/mooncake_v1/common/endpoints.py` | Handshake base/handshake port formulas for both connector families. The two historical pull and layerwise formula sets are kept independent; they are not merged. |
| `kv_p2p/mooncake_v1/common/block_selection.py` | `GroupTransferInfo` plus 11 block-selection/truncation pure helpers (6 used by the pull scheduler, 5 by the layerwise scheduler). |
| `kv_p2p/mooncake_v1/pull/metadata.py` | Pull request metadata types `ReqMeta`, `GroupPull`, `MooncakeConnectorMetadata` and the frozen `TransferRequest` wrapper, including the legacy-dict adapters (`from_legacy_kwargs`, `from_legacy_dict`, `as_legacy_dict`). |
| `kv_p2p/mooncake_v1/pull/transfer_plan.py` | Pure transfer-plan helpers of the pull family: KV-head grouping, CP parameter validation/group metadata, port mapping and HMA shared-port balancing, remote host/port resolution, CP block-size parameters and kernel block parameters. |
| `kv_p2p/mooncake_v1/layerwise/metaserver.py` | `MetaserverClient`: the TLS/plain `httpx.Client` construction, retry loop and idempotent `close()` used by the layerwise scheduler. |

**Dependency direction.** The registered V1 entry modules (`kv_p2p/mooncake_connector.py` and `kv_p2p/mooncake_layerwise_connector.py`, selected by `KVConnectorFactory` with the existing connector names) compose these helpers and re-export the moved symbols, so existing imports and subclasses keep working. `mooncake_v1/common` never imports the V1 entry modules, which keeps the shared layer acyclic. The V2 `mooncake/` connector and the Hybrid connector are not modified by this layering.

**Lifecycle.** Both V1 connectors implement an idempotent `shutdown()` hook. The pull worker signals and joins its transfer threads, shuts down the receive `ThreadPoolExecutor` and closes the remote socket pool; the layerwise scheduler first shuts down its executor and then closes the metaserver client, while the layerwise worker signals and joins its layer threads. The transfer-thread startup path uses the same idempotent shutdown for rollback when a thread exits early or does not become ready within the startup timeout. The sending threads record an unrecoverable error in `fatal_error`, which the worker exposes through `get_thread_fatal_error()` and logs once during shutdown; request-level aggregation of that error is not implemented yet.

### 4. Specifications Design

This feature is flexible and supports various configurations, including setups with MLA and GQA models. It is compatible with A2 and A3 hardware configurations and facilitates scenarios involving equal TP setups and certain unequal TP setups across multiple P and D nodes.

| Feature                       |      Status    |
|-------------------------------|----------------|
| A2                            | 🟢 Functional  |
| A3                            | 🟢 Functional  |
| equal TP configuration        | 🟢 Functional  |
| unequal TP configuration      | 🟢 Functional  |
| MLA                           | 🟢 Functional  |
| GQA                           | 🟢 Functional  |

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🔵 Experimental: Experimental support, interfaces and functions may change.
- 🚧 WIP: Under active development, will be supported soon.
- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
- 🔴 NO plan/Deprecated: No plan or deprecated by vLLM.

### 5. Compatibility, rollback and known limitations

#### Compatibility

- **Configuration**: `--kv-transfer-config` keeps the same fields, semantics and defaults. No new required configuration key or environment variable is introduced.
- **Registration**: the connector names registered through `KVConnectorFactory` are unchanged (`MooncakeConnectorV1` / `MooncakeConnector`, `MooncakeLayerwiseConnector`, `MooncakeD2RHConnectorV1` and the Hybrid connector). Existing P/D deployment scripts keep working as-is.
- **Imports**: the pre-refactor Python import paths keep working. The V1 entry modules re-export the symbols that moved to `mooncake_v1` (for example the wire message constants and the metadata types; the pull module additionally keeps `KVCacheTaskTracker` and `RemotePortInfo` importable), so downstream code that imports them from `mooncake_connector` / `mooncake_layerwise_connector` does not need to change.
- **Wire format**: message constants and metadata field names, field order and default values are unchanged, so peers running the old and the new code remain compatible.
- **D2RH inheritance seam**: the D2RH connector still receives a legacy dictionary in `_transfer_kv_cache_all_groups(req_meta: dict)`; the pull thread queue keeps carrying a dictionary produced by `TransferRequest.as_legacy_dict()` with the same keys and order as before.
- **Out of scope**: the Hybrid connector and the V2 `mooncake/` connector are untouched by this layering.

#### Migration

No user action is required: there are no new required configuration keys, environment variables, connector names or wire changes. As an internal refactor it needs no release note.

#### Rollback

There is no feature flag. The change is split into self-contained, single-topic commits (shared layer, scheduler helpers, worker transfer-plan helpers, lifecycle/error handling, documentation), so it can be reverted per commit or per pull request without touching the remaining layers.

#### Known limitations

1. **Layerwise startup wait**: the layerwise worker used to wait for its layer threads without a timeout. It now gives up after a 5-minute timeout, polling every 3 seconds. A slow but healthy startup can therefore be delayed by up to roughly 3 extra seconds once, and a failed startup now raises instead of hanging indefinitely.
2. **Fatal thread errors are only recorded**: an unrecoverable error in a sending thread is stored in `fatal_error`, exposed by `get_thread_fatal_error()` and logged once during shutdown. It is not yet aggregated into request-level `failed_recving` results; that is a follow-up.
3. **D2RH cleanup is partial**: the D2RH connector inherits the shared idempotent `shutdown()` for the base threads, executor and socket pool, but its overridden `register_kv_caches` does not use the shared startup rollback yet, and its CPU-staging thread and staging resources are not stopped by the inherited shutdown.
4. **Follow-up structural work**: the reformat subsystem extraction, the remaining transfer-plan pure helpers (DCP shards, group pulls, hybrid remote-rank group pulls, SFA replicate-K block ids, remote-rank selection) and a layerwise step-controller / read-only metadata split are intentionally deferred to later changes.
5. **V1/V2 helper unification**: V1 `mooncake_v1/common/transport.py` and V2 `mooncake/utils.py` are not unified. V2 `mooncake/__init__.py` eagerly imports the V2 package and keeps its own `zmq_ctx`/`SizedDict`, so merging the two helper layers is left for a follow-up pull request.

---

## DFX Analysis

### 1. Config Parameter Validation

Validate KV transfer config by checking whether the kv_connector type is supported. On transfer failures, emit clear error logs for diagnostics.

### 2. Port Conflict Detection

Before startup, perform a port-usage check on configured ports (e.g., rpc_port, metrics_port, http_port/metaserver) by attempting to bind. If a port is already in use, fail fast and log an error.

### 3. PD Ratio Validation

Under non-symmetric PD scenarios, validate the P-to-D tp ratio against expected and scheduling constraints to ensure correct and reliable operation.

---

## Limitations

- Heterogeneous P and D nodes are not supported, for example, running P nodes on A2 and D nodes on A3.

- In non-symmetric TP configurations, only cases where the P nodes have a higher TP degree than the D nodes and the P TP count is an integer multiple of the D TP count are supported (i.e., P_tp > D_tp and P_tp % D_tp = 0).
