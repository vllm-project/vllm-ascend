# SFA PD RD2H connector (`sfa_pd_rd2h`)

PD-disaggregated KV-transfer connector for **SFA (Sparse Flash Attention)** models
(MLA latent KV + indexer), dedicated to **decode offload**. It runs in **pull mode over
memfabric**: the Decode node pulls KV out of the Prefill node's HBM — the Prefill node
never pushes.

* Connector name: `SFAPDRD2HConnector`
* Module: `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`
* Backend: `memfabric` only (HMA / `batch_transfer_sync_read`)

## Roles and destination layout

| | Prefill (P) | Decode (D) |
|---|---|---|
| `kv_role` | `kv_producer` | `kv_consumer` |
| KV residency during compute | HBM (regular paged KV) | main MLA → CPU offload pool, indexer → rank-local HBM |
| `sparse_kv_offload_config.enabled` | **must be `false`** | **must be `true`** |
| Transfer direction | exposes KV, notifies READ_READY | pulls KV via memfabric, replies READ_DONE |

Decode offload is asymmetric by design:

* **Main MLA latent KV** is replicated across P's TP ranks (MLA property). On D it lands in
  the **TP-shared CPU pool** owned by `SparseKVOffloadManager`, split `1/d_tp` per D rank.
* **Indexer KV** is needed **in full on every D rank** and lives in **rank-local HBM**.

## Two completion states (do not conflate)

1. **Layer-complete (P-side buffer-reuse gate).** P reuses each layer's KV buffer across
   steps. Before P overwrites a layer's storage slot, D must have finished reading it. D
   acknowledges **every** READ_READY_BATCH with READ_DONE / READ_FAILED — *including a
   batch in which this D rank pulled zero blocks* — so P's per-layer
   `storage_send_done_events` gate always releases. See
   `connector.py:wait_for_layer_send` and `send_thread.py`.
2. **Request-complete (D-side decode start).** D may start decoding a request only after
   the *whole* request's KV has been pulled. Tracked per request in
   `read_thread.py:_done_requests` and surfaced via `worker.get_finished`.

## Unequal P/D TP (e.g. P TP16 → D TP4·DP4)

P and D TP sizes may differ. Constraint (enforced by
`worker.py:_map_prefill_rank_to_decode_rank`):

```text
p_tp_size >= d_tp_size  AND  p_tp_size % d_tp_size == 0
ratio = p_tp_size // d_tp_size          # 16/4 = 4
```

The `ratio` P ranks mapping to one D rank form a **contributor group**. For P rank `p`:
`d_rank = p // ratio`, `group_member_idx = p % ratio`. P tags every READ_READY_BATCH with
`(group_member_idx, tp_ratio)` (appended fields `msg[5]`/`msg[6]`, backward-compatible).

To transfer the group's data **exactly once** (no duplicate writes, no `ratio ×`
bandwidth) the D-side pull is split:

* **Main MLA** — *not* fine-split. Only contributor `group_member_idx == 0` pulls this D
  rank's `1/d_tp` share into the CPU pool; other contributors skip main.
* **Indexer** — *fine-split*. Each contributor pulls `1/ratio` of the full indexer via
  `_tp_block_range`, so the group assembles exactly one full copy per D rank. The indexer
  scale tensor (LIC8) follows the same sub-range.
* **Request completion** — D waits for **all `ratio` distinct contributors'** last-layer
  signal before marking a request done (`read_thread.py:_record_chunk_done`). A
  contributor that owns zero blocks in a small chunk still counts, because D ACKs every
  batch.
* **Layer-complete gate is unchanged** — every contributor ACKs every batch, so P's
  buffer reuse still works regardless of the split.

`ratio == 1` (equal TP) degenerates to a single contributor pulling everything — no
special-casing. When a chunk has fewer blocks than `ratio`, `_tp_block_range` hands some
contributors an empty slice and they just ACK.

## Control-plane ports

P never binds `kv_port`; it learns D's endpoint from the metaserver rendezvous. **Only D
binds.** Each D TP rank runs a ZMQ ROUTER on:

```text
D listen port = kv_port + data_parallel_rank(GLOBAL) * tp_size + tp_rank
```

* `data_parallel_rank` is the **global** DP rank (unique across every D engine that may
  share a host), never the per-host local rank — otherwise single-host multi-DP engines
  all collide on `kv_port + 0`.
* Ports consumed = `dp_size * tp_size`. P computes its target as the advertised D base
  port + `remote_tp_rank` (`worker.py:start_load_kv`).
* Upper bound is validated (`kv_port + (dp-1)*tp + (tp-1) <= 65535`).

For TP4·DP4 with `kv_port = 20000`, D occupies `20000..20015`:

| DP rank | ports |
|---|---|
| 0 | 20000–20003 |
| 1 | 20004–20007 |
| 2 | 20008–20011 |
| 3 | 20012–20015 |

At startup D logs one line you can verify against the formula:

```text
SFAPDRD2H D DP topology: host=..., kv_port=..., data_parallel_rank=... (local=...), ... TP control-plane ports=[a..b]
```

## Wire protocol (ZMQ, msgspec positional tuples)

* `MF_META` — P → D, once per connection: P session id + per-layer base
  addresses/block_len/scale. D replies `ACK`.
* `READ_READY_BATCH` — P → D, per layer per endpoint:
  `(type, layer_idx, layer_name, read_reqs, done_ext_ids, group_member_idx, tp_ratio)`.
  `read_reqs` entries are `(ext_req_id, p_main_block_ids, p_indexer_block_ids,
  main_start_block, indexer_start_block)`.
* `READ_DONE` / `READ_FAILED` — D → P, per layer.

Fields are **append-only**; receivers parse with `len(msg) > N` guards so mixed-version
peers keep working (a missing tail defaults to legacy single-contributor behavior).

## Configuration reference

`--kv-transfer-config` (both sides):

| key | P | D | notes |
|---|---|---|---|
| `kv_connector` | `SFAPDRD2HConnector` | `SFAPDRD2HConnector` | |
| `kv_role` | `kv_producer` | `kv_consumer` | |
| `kv_port` | base (unused for bind) | base for the DP×TP port range | set the same value on both for clarity |
| `kv_connector_extra_config.transfer_backend` | `memfabric` | `memfabric` | **required**; missing → error |

`--additional-config` (D only):

```json
{"sparse_kv_offload_config": {"enabled": true, "topk_buffer_size": 4096, "dram_size_per_dp_GB": 128}}
```

* `dram_size_per_dp_GB` — CPU offload pool size **per DP rank**.
* `topk_buffer_size` must be `>= index_topk`.
* Requirements: sparse-attention model (`index_topk` in config), no context/pipeline
  parallel on D, and (unless `keep_device_kv_cache`) `kv_role=kv_consumer`.

## Run: P TP16 → D TP4·DP4

### 1. Prefill node (TP16, kv_producer)

```shell
vllm serve "/path/to/sfa-model" \
  --host <P_IP> --port 8100 \
  --tensor-parallel-size 16 \
  --trust-remote-code \
  --kv-transfer-config '{
    "kv_connector": "SFAPDRD2HConnector",
    "kv_role": "kv_producer",
    "kv_port": 20000,
    "kv_connector_extra_config": {"transfer_backend": "memfabric"}
  }'
```

Do **not** set `sparse_kv_offload_config` on P.

### 2. Decode node (TP4·DP4, kv_consumer)

```shell
vllm serve "/path/to/sfa-model" \
  --host <D_IP> --port 8200 \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --trust-remote-code \
  --kv-transfer-config '{
    "kv_connector": "SFAPDRD2HConnector",
    "kv_role": "kv_consumer",
    "kv_port": 20000,
    "kv_connector_extra_config": {"transfer_backend": "memfabric"}
  }' \
  --additional-config '{
    "sparse_kv_offload_config": {"enabled": true, "topk_buffer_size": 4096, "dram_size_per_dp_GB": 128}
  }'
```

Single-host `--data-parallel-size 4` lets vLLM assign global DP ranks 0–3 automatically.
For **multi-host DP**, launch each engine with the correct global rank
(`--data-parallel-start-rank`) so the port ranges stay disjoint.

### 3. Proxy

Use the **layerwise** proxy (this connector is layer-wise on both sides):

```shell
python load_balance_proxy_layerwise_server_example.py \
  --host <PROXY_IP> --port 9000 \
  --prefiller-hosts <P_IP> --prefiller-ports 8100 \
  --decoder-hosts  <D_IP> --decoder-ports 8200
```

`--host` must be a real reachable IP (not `0.0.0.0`): D calls back to
`http://<PROXY_IP>:9000/v1/metaserver` to rendezvous with P.

### 4. Infer

```shell
curl -s http://<PROXY_IP>:9000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "/path/to/sfa-model",
  "prompt": "The quick brown fox jumps over the lazy dog",
  "max_tokens": 64
}'
```

## Troubleshooting

| symptom | likely cause |
|---|---|
| `SFAPDRD2HConnector requires ...["transfer_backend"]` | `transfer_backend` missing from `kv_connector_extra_config` |
| `P/D tensor parallel sizes ... divisible` | `p_tp < d_tp` or `p_tp % d_tp != 0` |
| D port collision on startup | two D engines share global `data_parallel_rank` (multi-host DP not started with distinct ranks) |
| `SparseKVOffloadManager ... must run before` | D started without `sparse_kv_offload_config.enabled=true` |
| duplicate writes / `ratio ×` bandwidth | P and D not both on the contributor-group build (mixed versions) |
| `D is LIC8 ... no scale leg` / `scale layout mismatch` | P/D LIC8 (indexer scale) config mismatch |
