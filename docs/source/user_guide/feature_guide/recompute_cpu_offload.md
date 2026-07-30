# Recompute CPU Offload Guide

## Overview

`RecomputeCPUOffloadConnector` preserves the KV cache of requests that are
preempted by the Decode-side recompute scheduler. When HBM KV blocks are not
enough, `RecomputeScheduler` may preempt a running Decode request. Without this
connector, the request falls back to the original recompute path and may be sent
back to the Prefill node to run prefill again. With recompute CPU offload
enabled, the already-computed KV blocks are copied from HBM to CPU DRAM before
the HBM blocks are reused, and copied back to HBM when the request is scheduled
again.

This feature is designed for online P/D disaggregation workloads where Decode
nodes are tuned for decode throughput and cannot efficiently recompute long
prefills after preemption. It is opt-in and focuses on correctness first.

In a typical Decode deployment, `max_num_batched_tokens` is often sized around
`max_num_seqs * (1 + num_spec_tokens)` so that the Decode node mainly schedules
one new token, plus speculative tokens if MTP is enabled, per request. This is
efficient for decode but too small for recomputing long prompts after
preemption. Recompute CPU offload avoids sending such requests back through the
full prompt recompute path when their KV state can be preserved locally.

## Key Concepts

* **Recompute preemption**: When Decode-side HBM KV cache is exhausted,
  `RecomputeScheduler` can preempt a running request and later resume it.
* **CPU DRAM preservation**: `RecomputeCPUOffloadConnector` stores the
  preempted request's computed KV blocks in CPU memory.
* **H2D restore**: When the preempted request is scheduled again, the connector
  restores the preserved KV blocks before model forward.
* **Fallback behavior**: If the connector is not configured, the recompute
  scheduler is not enabled, or CPU offload capacity is insufficient, vLLM
  falls back to the original recompute behavior.
* **`MultiConnector` integration**: In P/D disaggregation, use
  `MultiConnector` to combine the P/D connector, such as `MooncakeConnectorV1`,
  with `RecomputeCPUOffloadConnector` on Decode nodes.

## Configuration Parameters

`RecomputeCPUOffloadConnector` is configured through `kv-transfer-config`.

| Parameter | Description |
| :--- | :--- |
| `kv_connector` | Must be set to `RecomputeCPUOffloadConnector`. |
| `kv_role` | Set to `kv_consumer` on Decode nodes. |
| `cpu_bytes_to_use_per_rank` | Optional and recommended. CPU memory budget in bytes used by each rank/card for recompute offload. If set, it overrides `cpu_bytes_to_use / world_size`. |
| `cpu_bytes_to_use` | Optional. Total CPU memory budget in bytes for this vLLM instance. The connector divides it by `world_size` to get the per-rank budget. This is less direct than `cpu_bytes_to_use_per_rank` and is easier to misconfigure in DP deployments. The default is 8 GiB total. |
| `enable_offload_prefix_caching` | Optional. Enables CPU block sharing for full hashed blocks with the same prefix-cache hash. The default is `false`; keep it disabled unless explicitly testing prefix sharing. |

Prefer `cpu_bytes_to_use_per_rank` when you want every rank/card to use the
same offload capacity. For example, set `cpu_bytes_to_use_per_rank` to
`17179869184` for 16 GiB per rank/card.

If you use `cpu_bytes_to_use`, remember that it is divided by `world_size`. In
typical P/D Decode deployments, this means the value is divided by the active
Decode DP size. For example, when starting a DP2TP8 Decode service, setting
`cpu_bytes_to_use` to 16 GiB gives each DP rank's cards about 8 GiB of recompute
offload space. To avoid ambiguity, use `cpu_bytes_to_use_per_rank` for new
deployments.

`scheduler_config.recompute_scheduler_enable` must also be enabled in `additional-config` on
P/D-disaggregated Decode nodes:

```bash
--additional-config '{"scheduler_config":{"recompute_scheduler_enable":true}}'
```

```{note}
`scheduler_config.recompute_scheduler_enable` is only valid in P/D-disaggregated mode
(`kv_role` is `kv_producer` or `kv_consumer`). Do not enable it in PD-mixed
mode (`kv_role` is `kv_both`).
```

## Docker Shared Memory

Recompute CPU offload allocates pinned CPU tensors for the offloaded KV blocks.
When running in Docker, make sure the container's shared memory is large enough.
If `--shm-size` is too small, a large per-card offload budget, such as 16 GiB
per card, can fail to allocate CPU tensors and may cause the service to OOM or
hang during startup.

For typical A3 deployments with a large recompute-offload budget, set
`--shm-size=1024g` when starting the container. The following snippet follows
the Docker style used by the DeepSeek-V4-Flash tutorial:

```bash

export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1024g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -it $IMAGE bash
```

## Usage with P/D Disaggregation

On Decode nodes, configure `MultiConnector` with both the P/D connector and
`RecomputeCPUOffloadConnector`. The P/D connector handles KV transfer from
Prefill to Decode, while `RecomputeCPUOffloadConnector` handles Decode-side
preemption preservation and restore.

The following example uses `MooncakeConnectorV1` for P/D KV transfer.

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --port 8200 \
    --trust-remote-code \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 128 \
    --max-num-batched-tokens 4096 \
    --additional-config '{"scheduler_config":{"recompute_scheduler_enable":true}}' \
    --kv-transfer-config \
    '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_consumer",
      "kv_connector_extra_config": {
        "connectors": [
          {
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "28000",
            "kv_connector_extra_config": {
              "prefill": {
                "dp_size": 1,
                "tp_size": 1
              },
              "decode": {
                "dp_size": 1,
                "tp_size": 1
              }
            }
          },
          {
            "kv_connector": "RecomputeCPUOffloadConnector",
            "kv_role": "kv_consumer",
            "kv_connector_extra_config": {
              "cpu_bytes_to_use_per_rank": 17179869184,
            }
          }
        ]
      }
    }'
```

For the Prefill node, keep the normal P/D connector configuration, for example
`MooncakeConnectorV1` with `kv_role` set to `kv_producer`.

When `kv_load_failure_policy` is also needed for the P/D connector, configure
it on the top-level `MultiConnector` `kv-transfer-config`, not inside child
connectors.

## Standalone Connector Example

`RecomputeCPUOffloadConnector` must be used together with the vLLM-Ascend
`RecomputeScheduler`. Because `RecomputeScheduler` is only supported on
P/D-disaggregated Decode nodes, recompute CPU offload is currently only
available for P/D-disaggregated Decode nodes as well.

The following standalone connector configuration is intended only as a minimal
configuration fragment for validating the recompute-offload path on a
P/D-disaggregated Decode node. It is not a PD-mixed deployment mode. In
PD-mixed or normal non-P/D deployments, do not enable recompute CPU offload; the
engine uses the normal vLLM recompute behavior instead.

```python
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="RecomputeCPUOffloadConnector",
    kv_role="kv_consumer",
    kv_connector_extra_config={
        "cpu_bytes_to_use_per_rank": 17179869184,
        "enable_offload_prefix_caching": False,
    },
)
```

For online serving:

```bash
vllm serve /path/to/model \
    --additional-config '{"scheduler_config":{"recompute_scheduler_enable":true}}' \
    --kv-transfer-config '{
        "kv_connector": "RecomputeCPUOffloadConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use_per_rank": 17179869184,
            "enable_offload_prefix_caching": false
        }
    }'
```

## How It Works

1. When `RecomputeScheduler` cannot allocate enough HBM KV blocks, it selects a
   Decode-side running request for preemption.
2. Before the request's HBM blocks are reused, the scheduler calls the
   connector's preemption hook. If enough CPU blocks are available, the
   connector records the HBM block to CPU block mapping.
3. The model runner calls `handle_preemptions()` before updating worker state.
   The worker copies the selected KV blocks from HBM to pinned CPU tensors.
4. After all workers report that the store is complete, the preempted request
   becomes restorable.
5. When the request is scheduled again, the connector reports the number of
   tokens that can be restored from CPU memory.
6. The scheduler allocates new HBM blocks and the connector builds the CPU block
   to HBM block reload mapping.
7. Before model forward, the worker copies the restored KV blocks back to HBM.
   Forward then continues from the restored KV state.

## Sliding-Window and MTP Support

Sliding-window models can contain logical block tables with null block IDs for
tokens outside the attention window. The connector preserves logical block
positions instead of compressing non-zero block IDs, so a block table such as
`[0, 0, 20, 21]` remains aligned as `[0, 0, cpu_20, cpu_21]` on the CPU side.
Only non-zero blocks are transferred.

With MTP or speculative decode enabled, the scheduler may allocate lookahead
blocks. The reload path clips the H2D range to the GPU block table actually
allocated for the resumed request.

## Notes and Limitations

* This feature requires the vLLM V1 engine and the vLLM-Ascend recompute
  scheduler.
* The feature is only intended for Decode nodes in P/D disaggregation. It is not
  supported in PD-mixed or normal non-P/D deployments.
* Do not enable `scheduler_config.recompute_scheduler_enable` in PD-mixed deployments. Without
  P/D-disaggregated Decode-side `RecomputeScheduler`, recompute CPU offload is
  not active and vLLM follows the normal recompute path.
* When running in Docker with a large per-rank offload budget, reserve enough
  shared memory. For typical A3 deployments with 16 GiB per-card offload, use
  `--shm-size=1024g`.
* `enable_offload_prefix_caching` is experimental and disabled by default.
* Current D2H and H2D transfers use basic torch copy operations. They are
  correctness-oriented and not yet optimized for transfer throughput.
* Qwen3.5 with async scheduling is not fully supported yet. Disable async
  scheduling when using recompute CPU offload with Qwen3.5 models.
* If CPU memory capacity is insufficient, the connector skips offload for that
  request and the scheduler falls back to the original recompute behavior.

## FAQ

### How much CPU memory should I configure?

Start with a budget that can hold the expected number of preempted request
blocks. Prefer `cpu_bytes_to_use_per_rank` because it directly controls the
offload budget for each rank/card. For example,
`cpu_bytes_to_use_per_rank=17179869184` gives each rank/card 16 GiB.

`cpu_bytes_to_use` is divided by `world_size`. In a DP2TP8 Decode service,
setting `cpu_bytes_to_use=17179869184` gives each DP rank's cards about 8 GiB
of offload space. Increase the budget if logs show that CPU cache free blocks
are insufficient.

### How do I know the offload path is active?

Look for logs similar to:

```text
Recompute preemption offload enabled for request ...
Created recompute offload state for request ...
Prepared recompute offload H2D load for request ...
```

If offload cannot be prepared, logs may show that CPU cache free blocks are
insufficient, and the request will fall back to the original recompute path.

### Is this the same as KV Cache CPU Offload?

No. `KV Cache CPU Offload` is a prefix-cache offload path for inactive KV cache
blocks. `RecomputeCPUOffloadConnector` is specifically for preserving KV blocks
of requests preempted by the Decode-side recompute scheduler.

## Reference

For design background and implementation details, see
[#10820: Recompute CPU Offload Connector](https://github.com/vllm-project/vllm-ascend/issues/10820).
