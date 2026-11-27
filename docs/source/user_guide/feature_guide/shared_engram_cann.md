# Shared host Engram over the A5 CANN side group

This opt-in DeepSeek V4.1 backend keeps one immutable Engram table in an external owner
process, like [Shared host Engram over HIXL](shared_engram_rdma.md), but retrieves rows
through the CANN `ElasticBuffer` fixed side group instead of HIXL. The owner holds the
complete table in DRAM; the readers fetch rows with device-side operators on the NPU
stream and synchronize with the decode graph through events.

The backend is intended for A5 deployments that need the lowest per-token lookup latency
and can satisfy the fixed-membership constraints below. It requires FP8 E4M3 codes with
E8M0 scales and a build of `cann_ops_transformer` that exposes `ElasticBuffer` and the
`engram_fetch`/`engram_fetch_wait` operators.

## Data path

```text
Consumer NPU hashes stay on the NPU
  -> one side HCCL group per Engram layer (owner rank 0 + one reader per model rank)
  -> device-side engram_fetch(ids)
  -> engram_fetch_wait
  -> NPU dequantization into BF16 rows
  -> existing TP/DP row exchange and graph-input preparation
  -> existing model execution
```

The model process keeps the IDs on the NPU and performs the lookup entirely on the
device, independently of the host control protocol used by the HIXL backend. The owner
registers the complete table with the side group once; readers write a one-row
placeholder to establish their slot.

## Start the owner

The owner is a separate process and needs one NPU for the side group. It loads both
Engram layers into ordinary DRAM.

```bash
python -m vllm_ascend.shared_engram_cann \
  --model /path/to/checkpoint \
  --endpoint OWNER_IP:29500 \
  --readers 8 --device 0
```

Wait for `CANN_OWNER_LISTENING`, then `CANN_OWNER_LAYER_READY` for each layer. The owner
prints `CANN_OWNER_CLOSED` after every reader reports completion.

## Enable consumers

Use the same Engram options as the HIXL backend, and select the CANN backend in
`--additional-config`:

```json
{
  "shared_engram": {
    "backend": "a5-cann",
    "endpoint": "OWNER_IP:29500",
    "capacity": 65536,
    "readers": 8,
    "rank_base": 1
  }
}
```

`readers` is the number of model ranks that join the side group; it defaults to the model
world size and must match the owner `--readers`. `rank_base` is the side-group rank of the
first model rank and defaults to 1 (rank 0 is the owner). All model ranks start and close
the side group collectively.

## Constraints

- The side group has fixed membership. Owners and readers start together, and the
  configured readers stay for the lifetime of the group.
- `a5-cann` requires tensor parallel size 1 so that each rank owns the complete table.
  Data parallelism is supported.
- Every layer uses one side group with `readers + 1` ranks; the maximum is eight readers.
- The backend supports FP8 E4M3 codes with E8M0 scales.
- Use only the options listed above with `a5-cann`; `library_path`, `listen_port`, and
  `local_comm_res` belong to the HIXL backend.
- Set a positive `--shutdown-timeout` so workers can close the side group. An unexpected
  reader failure holds the reader and the owner until any registration whose device
  accesses may still be in flight has retired. Failure recovery and owner failover are
  operator-driven.

## Deployment notes

The side group is a separate HCCL group created from the process HCCL environment. Both
the owner and every consumer need a valid platform topology file (for example
`/etc/cloud/topo/950/cloud_atlas_950_1.json` on A5) and the intended interface configured
through `HCCL_IF_IP`/`HCCL_SOCKET_IFNAME`. The owner needs one NPU even though the table
lives in DRAM, and the host must have room for both complete Engram layers.

## Performance

On A5 with DeepSeek V4.1 Flash, an eight-rank DP8/TP1/EP8 decode instance and a cross-node
owner, the CANN backend was measured against the shared HIXL backend under the same model,
placement and load (5- and 321-token prompts, concurrency 1 and 8, `max_tokens` 64,
`FULL_DECODE_ONLY` capture size 1):

| Configuration | CANN ITL / output tok/s | HIXL ITL / output tok/s |
|---|---:|---:|
| short prompt, concurrency 1 | 12.49 ms / 65.6 | 13.74 ms / 60.2 |
| short prompt, concurrency 8 | 12.75 ms / 432.9 | 14.65 ms / 392.6 |
| long prompt, concurrency 1 | 12.72 ms / 64.8 | 13.95 ms / 59.5 |
| long prompt, concurrency 8 | 12.92 ms / 464.5 | 14.94 ms / 386.3 |

Per-token latency is 8.8–13.5% lower and output throughput 8.9–20.3% higher than the shared
HIXL backend. These are single-model, single-workload A5 measurements; production SLOs
require separate validation. The PR description lists the raw samples and exact comparison
conditions.
