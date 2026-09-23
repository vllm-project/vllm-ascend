# RFork Guide

This guide explains how to use **RFork** as a model-loader plugin in **vLLM Ascend**.

---

## Overview

RFork is a warm-start weight loading path for vLLM Ascend. Instead of always reading model weights from storage, a new instance can request a compatible **seed** instance from an external planner, then pull weights directly from that seed through `YuanRong TransferEngine`.

The RFork loading flow in the current implementation is:

1. vLLM starts with `--load-format rfork`.
2. RFork builds a **seed key** from the model identity and deployment topology.
3. RFork asks the planner for an available seed matching that key.
4. If a seed is returned, the new instance initializes the model structure on its local NPU, registers local weight memory, fetches the remote transfer-engine metadata from the seed, and performs batch weight transfer into local parameter buffers.
5. If no seed is available, or any step fails, RFork releases its registered memory and seed service, then falls back to the default loader. Fallback is conditional on that cleanup succeeding: if registered memory or the seed service cannot be released, RFork raises instead of allocating a second model while the old weights are still pinned.
6. After the instance finishes loading, it starts a local seed service and periodically reports heartbeat to the planner, so later instances can reuse it.

## Flowchart

![rfork flowchart](./images/rfork_flowchart.jpg)

## Application Scenarios

- **Scale-out after a first successful load**: The first instance may still load from storage, but later instances with the same deployment identity can reuse it as a seed and shorten startup time.
- **Elastic serving clusters**: Because RFork asks a planner for available seeds, it fits clusters where instances are created and reclaimed dynamically.
- **Topology-sensitive deployments**: RFork derives the seed key from a configuration fingerprint, a structural digest of the built model's transferable tensors, and the shard topology (`tp_rank`, optional `pp_rank`, optional `ep_rank`, and an optional `draft` marker), so only structure- and topology-compatible instances are matched together.

---

## Usage

To enable RFork, pass `--load-format rfork` and provide RFork settings through `--model-loader-extra-config` as a JSON string.

### RFork Prerequisites

- Install the runtime dependency `YuanRong TransferEngine` on every RFork instance.
- Run a planner service that implements the RFork seed protocol. A simple mock planner script is provided at [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py).

### Configuration Fields

| Field Name | Type | Description | Allowed Values / Notes |
| ------------ | ---- | ----------- | ---------------------- |
| **model_url** | String | Logical model identifier used to build the RFork seed key. | Required for RFork transfer. Instances that should share seeds must use the same value. |
| **model_deploy_strategy_name** | String | Deployment strategy identifier used together with `model_url` to build the seed key. | Required for RFork transfer. Instances that should share seeds must use the same value. |
| **rfork_scheduler_url** | String | Base URL of the planner service used for seed allocation, release, and heartbeat. | Required for planner-based matching. Example: `http://127.0.0.1:1223`. |
| **rfork_seed_timeout_sec** | Number | Timeout for waiting until the local seed HTTP service becomes healthy after startup. | Optional. Default: `5.0`. Must be greater than `0`. Invalid values fall back to the default. |
| **rfork_request_timeout_sec** | Number | HTTP connect/read timeout for planner and seed requests. | Optional. Default: `10.0`. Must be greater than `0`. |
| **rfork_heartbeat_interval_sec** | Number | Interval between planner heartbeat reports. | Optional. Default: `30.0`. JSON configuration only. |
| **rfork_lease_release_max_attempts** | Integer | Fast-attempt count before transient lease-release failures continue at a slower background interval. | Optional. Default: `3`. Permanent planner rejection still stops retries. |
| **rfork_lease_release_retry_interval_sec** | Number | Interval between fast lease-release retries. | Optional. Default: `30.0`. JSON configuration only. |
| **rfork_seed_bind_host** | String | Local address used by the seed HTTP service. | Optional. Default: `0.0.0.0`. |
| **rfork_seed_port_base** | Integer | Base port for deterministic seed HTTP endpoints. | Optional. Default: `0`, which lets the OS assign a port. For a nonzero base, main and draft models reserve adjacent slots per worker: main uses `base + 2 * global_rank`, draft uses `base + 2 * global_rank + 1`. Ports above `65535` or ports already in use fall back to OS assignment. Keep this range separate from TransferEngine, its HIXL endpoint range, and other service ports. |
| **rfork_seed_advertise_host** | String | Seed address advertised to the planner. | Optional. Automatically detected when unset. |

The YuanRong TransferEngine RPC port is not a configuration field: RFork initializes the engine with port `0`, and the
engine binds and holds an OS-assigned port that RFork reads back via `get_rpc_port()` and publishes as the transfer
session endpoint. To constrain the engine to a fixed port range (for example, for firewall rules), set YuanRong's
`YR_TE_RPC_PORT_MIN` and `YR_TE_RPC_PORT_MAX` environment variables (both required, valid within `1024-65535` and
outside the ephemeral range `32768-60999`); invalid values fall back to OS assignment. Keep this range separate from
the seed HTTP ports and the HIXL endpoint range. The HIXL data-plane endpoints default to base port `22000`
(`YR_TE_HIXL_BASE_PORT`, 100 ports per physical device), which stays clear of Mooncake ADXL's default `20000`-based
segments on nodes that co-locate both engines.

### How RFork Matches Seeds

RFork does not match instances by `model_url` alone. The local seed key is derived from three layers:

- a **compatibility fingerprint** over value-semantic configuration,
- a **structural digest** over the built model's transferable tensors, and
- the **shard topology**: `tp_rank`, optional `pp_rank`, optional `ep_rank`, a conditional `sharded_dp_rank`, and an optional `draft` marker.

The compatibility fingerprint covers configuration that changes tensor *contents* without changing tensor *shapes*: the RFork protocol version, `model_url`, `model_deploy_strategy_name`, the resolved model revision, the checkpoint quantization config digest, the architecture list, RoPE settings, the effective P/D role (`kv_role`), and `mix_placement`.

The structural digest is computed from the same transferable tensor set the transfer manifest validates — names, shapes, dtypes, and NPU formats. Any setting that changes tensor structure is therefore captured automatically: weight dtype, TP/PP/EP sharding, NZ and fused MC2 layout switches, quantization post-processing, and speculative draft topology. New Ascend weight-layout switches do not need to be enumerated in the fingerprint.

Shard ranks distinguish same-shaped shards: `tp_rank` picks the TP slice, `pp_rank` the pipeline stage, `ep_rank` the expert set (expert weights are same-shaped across EP ranks), and `sharded_dp_rank` the DP position — but only when weight content is actually sharded across the DP dimension. Two deployment forms break DP replication: fine-grained TP (`finegrained_tp_config`) shards selected modules across the DP dimension, and MoE models with `data_parallel_size > 1` shard expert weights across the DP×TP plane — with or without expert parallelism enabled, since with EP off the DP position is the only thing distinguishing same-shaped different-content expert shards. In those cases the DP rank joins the seed key as a shard selector; replicated DP ranks leave it unset and keep sharing one seed pool.

Because the structural digest exists only after the model is built and its memory is registered, the seed key is computed lazily and logged together with the digest right after registration (INFO on TP rank 0, DEBUG elsewhere). A seed miss therefore leaves a debuggable trail of the exact key and digest that did not match.

This means two instances must agree on model identity, tensor structure, and deployment topology before the planner will treat them as interchangeable seeds.
The final manifest is still validated before transfer, and RFork falls back rather than copying incompatible tensors.

### Planner Responsibilities

The planner treats the seed key as an opaque compatibility identifier. RFork derives it from the model/configuration fingerprint, the structural digest of transferable tensors, and shard topology. A planner should:

- Match exact seed keys, track seed availability through heartbeats, and expire stale advertisements.
- Allocate leases according to seed capacity, accept lease releases, and reclaim abandoned or expired leases.
- Make seed selection and lease state observable for operations.

Lease release is asynchronous. An unresolved release delays advertising the receiver as a seed, but does not block inference. The bundled planner is a functional example, not a production scheduler.

### Quantized Models

For quantized models, RFork transfers tensors after Ascend weight post-processing instead of raw checkpoint parameters. The receiver first builds the same post-load tensor layout as the seed, then RFork copies the live NPU tensors used by inference.

This path handles Ascend quantization changes such as weight transposition, NZ format conversion, packed weights, derived scale tensors, and MLA/SFA runtime tensors such as `W_UV` and `W_UK_T`. Empty tensors that were released during post-processing are not included in the transfer manifest.

When validating RFork for a quantized model:

- Apply the same vLLM Ascend code to both the seed instance and the receiver instance.
- Restart the planner and all vLLM instances after changing RFork code, because existing seeds keep their old transfer metadata.
- Use a new `model_deploy_strategy_name` after changing model arguments or RFork code, so the planner does not match a receiver with an incompatible old seed.
- A successful TP0 RFork transfer logs elapsed time, bytes, chunks, and throughput at INFO. Other TP ranks and
  per-chunk details remain at DEBUG. The two fallback paths are logged differently: a plain seed miss logs
  `seed acquisition was unsuccessful; loading locally` (INFO on the summary rank, DEBUG elsewhere) and reports
  `source=local`, while a failed transfer logs `RFork transfer failed` at WARNING and reports `source=fallback`.
  Treat only the latter as a transfer defect.

### Intentional transfer contracts

The following behaviors are deliberate RFork protocol choices, not missing
manifest checks:

- **Dense transpose and reshape:** RFork transfers the one continuous byte range
  covered by a non-overlapping dense tensor. Before the native read, the receiver
  may replace its tensor metadata with a storage-preserving view of the seed
  shape. RFork therefore does not require a separate seed/receiver stride-equality
  check. Compatible instances are expected to construct the same dense byte
  order through the fingerprinted configuration and the same post-load path;
  semantic and physical layout digests are diagnostic evidence for that contract.
  Gapped or overlapping layouts are still rejected. A new layout implementation
  that changes byte order must update the compatibility descriptor or extend the
  transfer protocol instead of relying on the existing dense-view contract.
- **Processed NZ payload length:** RFork intentionally reads exactly
  `numel * element_size` bytes for each named tensor, including processed NZ,
  packed-weight, and derived-scale tensors. Allocation capacity, descriptor-only
  padding outside the tensor's dense logical range, and adjacent or shared storage
  are not part of that tensor's payload and are not copied implicitly. A future
  NPU format that requires bytes outside this range needs explicit manifest and
  transfer-protocol support; the diagnostic physical-size fields do not silently
  widen a read.

## Supported Models

Mainstream DeepSeek/Qwen/GLM series are supported.

## Performance Considerations

RFork indexes tensor address ranges to find backing allocations and uses binary search for memory-coverage checks during registration. Registration still adds seed startup cost, so RFork is most beneficial when later replicas reuse a seed rather than for a single cold start.

Performance depends on model size, shard topology, NPU tensor layout, TransferEngine registration time, network bandwidth, and the number of concurrent destinations. Compare with the default loader using:

- Time from process start until the service is ready.
- Registration, metadata, and read timings in the DEBUG `RFork transfer stages` log, plus bytes and throughput in `RFork weight transfer completed` (INFO on TP rank 0, DEBUG on other ranks).
- Seed inference latency and storage/host-memory traffic while transfers are active.
- Seed hit rate, transfer failure rate, and fallback rate.

---

## Example Commands & Placeholders

> Replace parts in `<...>` before running.

### 1. Install YuanRong Datasystem

```shell
pip install openyuanrong-datasystem
```

### 2. Start the Planner

A simple planner implementation is provided at [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py).

```shell
python rfork_planner.py \
  --host 0.0.0.0 \
  --port <planner_port>
```

### 3. Start vLLM Instances

Use the same RFork startup command for both the first instance and later instances in the same deployment.

For the first instance, the planner usually has no compatible seed yet, so RFork falls back to the default loader. After loading finishes, that instance starts its local seed service and reports itself to the planner.

For later instances, if the planner can allocate a compatible seed, RFork will try to transfer weights from the existing seed instance before falling back to the default loader.

```shell
export RFORK_CONFIG='{
  "model_url": "<model_url>",
  "model_deploy_strategy_name": "<deploy_strategy>",
  "rfork_scheduler_url": "http://<planner_ip>:<planner_port>"
}'

vllm serve <model_path> \
  --tensor-parallel-size 1 \
  --served-model-name <served_model_name> \
  --port <port> \
  --load-format rfork \
  --model-loader-extra-config "${RFORK_CONFIG}"
```

### Placeholder Descriptions

- `<model_path>`: Model path or model identifier passed to `vllm serve`.
- `<served_model_name>`: Service name exposed by vLLM.
- `<planner_ip>`: IP address or hostname of the RFork planner.
- `<planner_port>`: Listening port of the RFork planner.
- `<model_url>`: Stable model identity string used to build the RFork seed key.
- `<deploy_strategy>`: Stable deployment-strategy name used to build the RFork seed key.
- `<port>`: Serving port of the vLLM instance being started.

Successful loads log `source=transfer`, `local`, `fallback`, or `shared_target`.
Successful TP0 weight reads also log transfer elapsed time, bytes, chunks, and
throughput at INFO; other TP ranks and per-chunk timings remain at DEBUG.
Every successful registration, receiver-before-read, and final receiver stage
emits one bounded `RFork tensor layout summary` at INFO per rank. The summary hashes all tensor
names, shapes, strides, dtypes, NPU formats, logical byte counts, storage byte
capacities, storage offsets, and NPU descriptor element counts into fixed-size
semantic and physical digests. It also reports aggregate counts and at most
three representative tensors, preferring storage views or tensors whose NPU
descriptor size differs from logical `numel`. Match a receiver's `peer_session`
to the seed's `session`. Digest differences are diagnostic and do not by
themselves reject a transfer. On checkpoint-layout transfers, compare
`receiver_before_read` with `receiver_after_post_load` to determine whether the
post-load hook rebuilt the layout. Processed-layout transfers instead emit
`receiver_after_transfer_finalize`, because their layout processing happened
before the read. The summary does not copy tensor data or prove value equality;
validate output accuracy separately on NPU hardware.
Set `VLLM_LOGGING_LEVEL=DEBUG` for per-rank registration, metadata, transfer,
lease-release, and publication timing.

---

## Note & Caveats

- RFork requires `YuanRong TransferEngine` at runtime. If the package is missing, RFork cannot initialize the transfer backend.
- If RFORK is used, **each worker process** must bind a listening port. That port is assigned randomly.
- RFork transfer is bypassed entirely — the model loads through the default loader and no seed is advertised — under any of the configurations below. RFork registers live NPU weight memory and serves it to other instances, so it cannot be used when that memory may later be released or rewritten, nor when expert placement is not fixed at load time. The bypass is decided before any session or TransferEngine setup and is logged as `RFork transfer is disabled when <reason> is enabled; using the default model loader.`
    - **Sleep mode** (`enable_sleep_mode` on either the model config or `vllm_config.model_config`), because sleeping discards and reallocates weight storage.
    - **Online weight transfer** (`weight_transfer_config` is set), because weights can be replaced after loading.
    - **Static expert placement** (`eplb_config.expert_map_path` is set), because the placement map is applied outside the transferred tensor set.
    - **Dynamic EPLB**, because expert weights and placement can change after the seed service starts. This covers `parallel_config.enable_eplb`, `eplb_config.dynamic_eplb`, and `eplb_config.expert_map_record_path`.
- Falling back to the default loader requires releasing the RFork seed service and registered memory first. RFork retries that cleanup (`FALLBACK_CLEANUP_MAX_ATTEMPTS`, currently 2, with a short backoff) and only then reclaims memory and reruns the default loader. If cleanup still fails, tensor owners remain pinned and RFork raises rather than allocating a second copy of the model; a session that already reached the finalized state raises as well. Both cases abort startup instead of degrading to a slow load.
- The example [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py) is only a simple mock implementation. If you need stronger scheduling, capacity management, or production-grade availability behavior, implement your own planner based on the RFork seed protocol.
- The planner protocol headers `SEED_REFCNT` and `SEED_RANK` carry no scheduling meaning, but they are not optional on the wire today:
    - `SEED_REFCNT` is always sent as `0` on advertisement and is ignored for capacity decisions (seed capacity is controlled by planner configuration). A planner must still accept the header.
    - `SEED_RANK` remains a **required** field. RFork rejects a `/get_seed` response that omits it or carries a negative value, and sends it on every seed advertisement, lease renewal, lease release, and seed removal request. A planner must echo it back on `/get_seed` and accept it on the other routes. What is deprecated is using it for compatibility decisions: it duplicates the `tp_rank` already encoded in the seed key, so a planner must match on the seed key alone and treat `SEED_RANK` as an opaque part of the seed's identity tuple.
- Each heartbeat verifies that the seed HTTP service remains alive. If the service exits, RFork stops heartbeats and
  attempts to withdraw the advertisement while leaving the loaded model available for inference.
- Temporary planner outages do not stop inference. Retryable initial advertisements and lease releases continue in the
  background, while permanent planner rejections prevent seed promotion. Outage logs are limited to the first failure,
  periodic summaries, and recovery events.
- Validate transfer accuracy and the logical-payload contract for newly supported NPU formats on the intended NPU and
  model combination; CPU tests cannot establish NPU storage correctness. This qualification is required when adding a
  format, but it is not a runtime requirement to copy allocator or descriptor padding outside a tensor's declared payload.
