# NPU Device Selection

Use `ASCEND_RT_VISIBLE_DEVICES` to restrict the NPUs visible to a process, and
`--device-ids` to select devices for a vLLM multiprocessing executor. Setting
`--tensor-parallel-size` controls the number of workers; it does not select
particular NPUs.

The examples below use integer device IDs, a single node, and the `mp` executor.
Use matching vLLM and vLLM Ascend versions from the
[installation guide](../../getting_started/installation.md). Ray has a different selection
mechanism; see [Ray](#ray) below.

## How the two settings interact

- Set `ASCEND_RT_VISIBLE_DEVICES` **before** starting vLLM. It remains supported;
  explicit device selection does not require removing this environment variable.
- If the variable is unset, integer `--device-ids` values select NPU IDs directly
  in the current environment.
- If the variable is set, integer `--device-ids` values are **zero-based indices
  into that visible list**, in the specified order. They are not a second list
  of host device IDs.
- If `--device-ids` is omitted, vLLM allocates workers from the available visible
  devices according to the parallel configuration.

For TP2 on a machine exposing IDs `0` through `7`:

| `ASCEND_RT_VISIBLE_DEVICES` | `--device-ids` | Selected NPU IDs | Worker runtime devices |
| --- | --- | --- | --- |
| Unset | Omitted | `0,1` | `npu:0`, `npu:1` |
| Unset | `1,3` | `1,3` | `npu:1`, `npu:3` |
| `4,5,6,7` | Omitted | `4,5` | `npu:0`, `npu:1` |
| `4,5,6,7` | `1,3` | `5,7` | `npu:1`, `npu:3` |
| `4,5,6,7` | `3,1` | `7,5` | `npu:3`, `npu:1` |

For example, `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 --device-ids 3,1` selects entries
`7,5`, not devices `3,1`. Conversely, `ASCEND_RT_VISIBLE_DEVICES=1,3` combined
with `--device-ids 1,3` is invalid: index `3` is outside a two-device list.

The runtime device number is relative to the process's visibility, so `npu:0`
does not necessarily mean host device `0`. Container device mounts can further
restrict or remap devices; use IDs available inside the container. Neither
setting exposes devices that the container has not been granted.

## Tensor parallelism

Select two non-contiguous devices without a visibility override:

```bash
unset ASCEND_RT_VISIBLE_DEVICES
vllm serve Qwen/Qwen3-8B \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 \
  --device-ids 1,3
```

Alternatively, restrict the visible set and omit `--device-ids`:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,3 vllm serve Qwen/Qwen3-8B \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2
```

Run these as alternatives, not simultaneously on the same devices. Adjust the
model and memory settings for your hardware.

## Data parallelism

For a single launch with two local DP replicas, each using TP2, provide the full
four-device list. vLLM splits it between the replicas; do not pass only one
replica's two-device slice to this launch.

```bash
unset ASCEND_RT_VISIBLE_DEVICES
vllm serve Qwen/Qwen3-8B \
  --distributed-executor-backend mp \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --tensor-parallel-size 2 \
  --device-ids 1,3,5,7
```

Replica 0 uses `1,3`; replica 1 uses `5,7`. For this single-node DP/TP layout,
allocate `DP × TP` distinct devices. An equivalent visibility-only launch uses
`ASCEND_RT_VISIBLE_DEVICES=1,3,5,7` and omits `--device-ids`.

Multi-node or separately launched DP ranks have different launch arguments and
local device allocations. Follow the model's deployment tutorial rather than
copying this single-launch layout unchanged.

## Prefill-decode disaggregation

P and D are separate vLLM instances. On the same host, assign them disjoint
device sets and different service/connector ports. For example, with TP2 on
each side:

| Instance | Visibility-only selection | Alternative with visibility unset |
| --- | --- | --- |
| Prefill | `ASCEND_RT_VISIBLE_DEVICES=0,2`, omit `--device-ids` | `--device-ids 0,2` |
| Decode | `ASCEND_RT_VISIBLE_DEVICES=4,6`, omit `--device-ids` | `--device-ids 4,6` |

These are device assignments, not complete PD launch commands. Keep the model,
KV connector, proxy, and communication configuration from the
[single-node PD tutorial](../../tutorials/features/pd_disaggregation_mooncake_single_node.md)
or the [multi-node PD tutorial](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md).
On different hosts, the same numeric device IDs can refer to different NPUs.

## Ray

`--device-ids` has no effect with `--distributed-executor-backend ray`; vLLM
prints a warning. Ray placement groups determine which resources workers use.
Set each node's `ASCEND_RT_VISIBLE_DEVICES` before starting its Ray processes,
and follow the [Ray tutorial](../../tutorials/features/ray.md) for cluster and
environment configuration. Changing the variable in a later client shell does
not reconfigure an already running Ray cluster.

## Check the allocation

- Use distinct integer indices within the visible-list bounds. Duplicate IDs
  and out-of-range indices are rejected.
- Supply enough devices for the local parallel layout. For example, TP2 with
  `--device-ids 1` cannot start two workers.
- Check worker startup logs and the `npu-smi info` process table in the relevant
  host/container environment. Account for visibility remapping when comparing
  runtime device numbers with the selected NPUs.
- A successful HTTP response alone does not establish correct device placement.
  For DP and PD, verify every replica or instance, not just the API server.
