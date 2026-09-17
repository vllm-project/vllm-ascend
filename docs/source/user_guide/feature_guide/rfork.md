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
5. If no seed is available, or any step fails, RFork cleans up and falls back to the default loader.
6. After the instance finishes loading, it starts a local seed service and periodically reports heartbeat to the planner, so later instances can reuse it.

## Flowchart

![rfork flowchart](./images/rfork_flowchart.jpg)

## Application Scenarios

- **Scale-out after a first successful load**: The first instance may still load from storage, but later instances with the same deployment identity can reuse it as a seed and shorten startup time.
- **Elastic serving clusters**: Because RFork asks a planner for available seeds, it fits clusters where instances are created and reclaimed dynamically.
- **Topology-sensitive deployments**: RFork encodes `kv_role`, `node_rank`, optional `pp_rank`, `tp_rank`, optional `ep_rank`, and optional `draft` role into the seed key, so only topology-compatible instances are matched together.

---

## Usage

To enable RFork, pass `--load-format rfork` and provide RFork settings through `--model-loader-extra-config` as a JSON string.

### RFork Prerequisites

- Install the runtime dependency `YuanRong TransferEngine` on every RFork instance.
- Run a planner service that implements the RFork seed protocol. A simple mock planner script is provided at [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py).

### Configuration Fields

| Field Name | Type | Description | Allowed Values / Notes |
|------------|------|-------------|------------------------|
| **model_url** | String | Logical model identifier used to build the RFork seed key. | Required for RFork transfer. Instances that should share seeds must use the same value. |
| **model_deploy_strategy_name** | String | Deployment strategy identifier used together with `model_url` to build the seed key. | Required for RFork transfer. Instances that should share seeds must use the same value. |
| **rfork_scheduler_url** | String | Base URL of the planner service used for seed allocation, release, and heartbeat. | Required for planner-based matching. Example: `http://127.0.0.1:1223`. |
| **rfork_seed_timeout_sec** | Number | Timeout for waiting until the local seed HTTP service becomes healthy after startup. | Optional. Default: `5.0`. Must be greater than `0`. Invalid values fall back to the default. |
| **rfork_seed_key_separator** | String | Separator used when building the RFork seed key string. | Optional. Default: `$`. Keep the same value across compatible instances. |
| **rfork_request_timeout_sec** | Number | HTTP connect/read timeout for planner and seed requests. | Optional. Default: `10.0`. Must be greater than `0`. |
| **rfork_heartbeat_interval_sec** | Number | Interval between planner heartbeat reports. | Optional. Default: `30.0`. JSON configuration only. |
| **rfork_lease_release_max_attempts** | Integer | Fast-attempt count before transient lease-release failures continue at a slower background interval. | Optional. Default: `3`. Permanent planner rejection still stops retries. |
| **rfork_lease_release_retry_interval_sec** | Number | Interval between fast lease-release retries. | Optional. Default: `30.0`. JSON configuration only. |
| **rfork_seed_bind_host** | String | Local address used by the seed HTTP service. | Optional. Default: `0.0.0.0`. |
| **rfork_seed_advertise_host** | String | Seed address advertised to the planner. | Optional. Automatically detected when unset. |

### How RFork Matches Seeds

RFork does not match instances by `model_url` alone. The local seed key is composed from:

- `model_url`
- `model_deploy_strategy_name`
- disaggregation mode derived from `kv_transfer_config.kv_role` or `kv_both`
- `node_rank`
- `pp_rank` when pipeline parallel size is greater than 1
- `tp_rank`
- `ep_rank` when expert parallelism is enabled for an MoE model
- optional `draft` suffix when the worker runs as a draft model

This means two instances must agree on both model identity and deployment topology before the planner will treat them as interchangeable seeds.
For deployments without pipeline or expert parallelism, the existing seed-key format is unchanged.

The compatibility fingerprint also includes the normalized model configuration, model revision, parallel topology,
device type, effective P/D role, quantization, speculative decoding, model-runner generation, and Ascend weight-layout
settings that can change transferable tensor names, shapes, dtypes, formats, or derived contents. Runtime-only settings
such as planner addresses, request scheduling policy, logging paths, and ACLGraph capture sizes are excluded. The final
manifest is still validated before transfer, and RFork falls back rather than copying incompatible tensors.

### Quantized Models

For quantized models, RFork transfers tensors after Ascend weight post-processing instead of raw checkpoint parameters. The receiver first builds the same post-load tensor layout as the seed, then RFork copies the live NPU tensors used by inference.

This path handles Ascend quantization changes such as weight transposition, NZ format conversion, packed weights, derived scale tensors, and MLA/SFA runtime tensors such as `W_UV` and `W_UK_T`. Empty tensors that were released during post-processing are not included in the transfer manifest.

When validating RFork for a quantized model:

- Apply the same vLLM Ascend code to both the seed instance and the receiver instance.
- Restart the planner and all vLLM instances after changing RFork code, because existing seeds keep their old transfer metadata.
- Use a new `model_deploy_strategy_name` after changing model arguments or RFork code, so the planner does not match a receiver with an incompatible old seed.
- A successful TP0 RFork transfer logs elapsed time, bytes, chunks, and throughput at INFO. Other TP ranks and
  per-chunk details remain at DEBUG. The fallback path logs `RFork transfer failed`.

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

## Tested Models

The following table records models that have been explicitly tested with RFork weight transfer. A model should be added here only after RFork transfer succeeds and the loaded instance passes basic inference validation.

| Model | Precision / Quantization | Hardware | Validation Status | Notes |
|-------|--------------------------|----------|-------------------|-------|
| Qwen2.5-7B | BF16 | A2 | Tested | RFork transfer has been validated. |
| Qwen3-32B | BF16 | A2 | Tested | RFork transfer has been validated. |
| Qwen3-235B-A22B | BF16 | A2 | Tested | RFork transfer has been validated. |
| DeepSeek-V4-Flash-W8A8-MTP | W8A8 | A2 | Tested | RFork transfer with MTP draft model has been validated. |
| GLM5-W4A8 | W4A8 | A2 | Tested | RFork transfer has been validated. |
| Kimi2.5-W4A8 | W4A8 | A2 | Tested | RFork transfer has been validated. |

---

## Example Commands & Placeholders

> Replace parts in `<...>` before running.

### 1. Install YuanRong TransferEngine

```shell
pip install openyuanrong-transfer-engine
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
- RFork weight transfer does not support dynamic EPLB because expert weights and placement can change after the seed service starts. If `eplb_config.dynamic_eplb` or `eplb_config.expert_map_record_path` enables dynamic EPLB, RFork transfer is bypassed and the model is loaded through the default model loader.
- The example [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py) is only a simple mock implementation. If you need stronger scheduling, capacity management, or production-grade availability behavior, implement your own planner based on the RFork seed protocol.
- Each heartbeat verifies that the seed HTTP service remains alive. If the service exits, RFork stops heartbeats and
  attempts to withdraw the advertisement while leaving the loaded model available for inference.
- Temporary planner outages do not stop inference. Retryable initial advertisements and lease releases continue in the
  background, while permanent planner rejections prevent seed promotion. Outage logs are limited to the first failure,
  periodic summaries, and recovery events.
- Validate transfer accuracy and the logical-payload contract for newly supported NPU formats on the intended NPU and
  model combination; CPU tests cannot establish NPU storage correctness. This qualification is required when adding a
  format, but it is not a runtime requirement to copy allocator or descriptor padding outside a tensor's declared payload.
