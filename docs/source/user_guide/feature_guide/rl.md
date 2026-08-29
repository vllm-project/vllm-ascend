# vLLM-Ascend for RL

## Overview

vLLM is commonly used as the rollout engine in reinforcement learning (RL) and
post-training workflows. A training framework generates samples with
vLLM-Ascend, updates the policy from those samples, and then synchronizes the
new weights back to the rollout engine. See the upstream vLLM
[RLHF guide](https://docs.vllm.ai/en/latest/training/rlhf/) for integrations
with RL frameworks.

RL workloads commonly need the following capabilities:

- release NPU memory while a colocated trainer is running;
- update model weights without restarting the rollout engine;
- pause generation while weights are being updated;
- return token log probabilities and, for supported MoE workflows, expert
  routing decisions;
- accept and return token IDs without redundant tokenization; and
- produce reproducible results across different batch compositions.

This page describes how these features fit together on Ascend. Follow the
linked feature guides and examples for complete configuration details.

## Enable RL mode

vLLM-Ascend provides a unified RL configuration under
`additional_config.rl_config`. Set its `enabled` field to `true` for both
same-NPU and cross-NPU deployments:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --additional-config '{"rl_config": {"enabled": true}}'
```

The master switch applies the common RL defaults: it disables FRACTAL_NZ,
enables the development endpoints, and removes `expandable_segments` from
`PYTORCH_NPU_ALLOC_CONF`. The optional fields are:

| Field | When to enable it |
| --- | --- |
| `sleep_mode_extra_cleanup` | Release HCCL process groups and ACL graph workspaces during sleep in a same-NPU deployment |
| `enable_training_consistency` | Select the FA3 attention backend for training-inference consistency |
| `enable_batch_invariant` | Enable batch-invariant kernels and deterministic communication settings |

All optional fields default to `false` and are ignored unless `enabled` is
`true`. See [Additional Configuration](../configuration/additional_config.md)
for field definitions, prerequisites, and migration from legacy settings.

## Choose a deployment mode

| Deployment mode | Memory management | Weight transfer backend |
| --- | --- | --- |
| Trainer and rollout engine share NPUs | Enable sleep mode so the rollout engine can release NPU memory | `ipc` |
| Trainer and rollout engine use different NPUs | Pause generation during the update; sleep mode is normally unnecessary | `hccl` |

The exact lifecycle is framework-dependent. In particular, level 2 sleep
discards the weights. Wake the `weights` allocation before loading or
transferring new weights, then wake `kv_cache` after the update. See
[Sleep Mode](./sleep_mode.md) for the two-phase wake-up sequence.

## Engine lifecycle

### Sleep mode

Sleep mode lets a colocated rollout engine release NPU memory without exiting.
Level 1 offloads model weights to CPU and discards the KV cache. Level 2
discards both model weights and KV cache and is appropriate when new weights
will be loaded before inference resumes.

For online control, enable the development endpoints and start the server with
sleep mode enabled:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve Qwen/Qwen3-0.6B \
    --enable-sleep-mode \
    --additional-config '{"rl_config": {"enabled": true}}'
```

The sleep level is a query parameter; a JSON request body is ignored:

```bash
# Release memory.
curl -X POST "http://127.0.0.1:8000/sleep?level=1"
curl http://127.0.0.1:8000/is_sleeping

# Restore all tagged allocations.
curl -X POST http://127.0.0.1:8000/wake_up
```

For level 2 sleep, wake the tags in the order documented in
[Sleep Mode](./sleep_mode.md):

```bash
curl -X POST "http://127.0.0.1:8000/sleep?level=2"
curl -X POST "http://127.0.0.1:8000/wake_up?tags=weights"
# Load or transfer the new weights here.
curl -X POST "http://127.0.0.1:8000/wake_up?tags=kv_cache"
```

To release HCCL process groups and ACL graph workspaces as well, enable extra
cleanup:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve Qwen/Qwen3-0.6B \
    --enable-sleep-mode \
    --additional-config '{"rl_config": {"enabled": true, "sleep_mode_extra_cleanup": true}}'
```

Extra cleanup reduces sleep-time NPU memory use, but HCCL groups must be
restored and ACL graphs must be recaptured during wake-up. RL mode removes
`expandable_segments` from `PYTORCH_NPU_ALLOC_CONF` because it is incompatible
with the sleep-mode memory pool. The former top-level
`enable_sleep_mode_extra_cleanup` option has been removed.

### Weight transfer

vLLM-Ascend exposes two Ascend-specific weight-transfer backends:

| Backend | Use case | Transport |
| --- | --- | --- |
| `ipc` | Trainer and rollout engine use the same physical NPU | NPU IPC handles |
| `hccl` | Trainer and rollout engine use separate NPUs | HCCL collectives |

Start an HCCL-enabled server with:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --weight-transfer-config '{"backend": "hccl"}' \
    --additional-config '{"rl_config": {"enabled": true}}'
```

For same-NPU transfer, use the `ipc` backend. On Ascend, vLLM-Ascend maps this
user-facing backend name to `NPUIPCWeightTransferEngine`:

```bash
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
vllm serve Qwen/Qwen3-0.6B \
    --weight-transfer-config '{"backend": "ipc"}' \
    --additional-config '{"rl_config": {"enabled": true}}'
```

`VLLM_ALLOW_INSECURE_SERIALIZATION=1` is required by the HTTP NPU IPC example
because it serializes IPC handles. Only enable it in a trusted environment.

The control flow is:

1. initialize the transfer engine;
2. pause generation;
3. start the weight update;
4. transfer one or more groups of weights;
5. finish the weight update; and
6. resume generation.

The repository contains runnable examples for both backends:

- [`rlhf_http_hccl.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rl/rlhf_http_hccl.py)
- [`rlhf_http_npu_ipc.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rl/rlhf_http_npu_ipc.py)
- [`rlhf_async_new_apis.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rl/rlhf_async_new_apis.py)

FRACTAL_NZ must be disabled for weight updates. RL mode forces
`weight_nz_mode=0` and synchronizes `VLLM_ASCEND_ENABLE_NZ=0`; the
weight-transfer start path validates the effective Ascend configuration.

### Pause and resume generation

With `rl_config.enabled=true`, the server exposes `/pause` and `/resume`.
These are development endpoints and must not be exposed to untrusted
networks. Pause parameters are query parameters:

```bash
# Drain in-flight requests and clear caches before the update.
curl -X POST \
    "http://127.0.0.1:8000/pause?mode=wait&clear_cache=true"

# Transfer weights here.

curl -X POST http://127.0.0.1:8000/resume
```

The pause modes are:

| Mode | Behavior |
| --- | --- |
| `abort` | Abort in-flight requests immediately; this is the default |
| `wait` | Wait for in-flight requests to complete before pausing |
| `keep` | Freeze requests so they continue after `/resume` |

`clear_cache` is deprecated and is ignored for `mode=keep`. A kept request can
therefore contain tokens and KV-cache entries produced with the old weights and
continue with the new weights after resume. Use `abort` or `wait` when a rollout
must not span weight versions.

See the upstream [Async RL guide](https://docs.vllm.ai/en/latest/training/async_rl/)
for the API lifecycle.

## Training data returned by the engine

### Token log probabilities

Set `logprobs` and, when needed, `prompt_logprobs` on the completions request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Your prompt here",
        "max_tokens": 32,
        "temperature": 1.0,
        "logprobs": 1,
        "prompt_logprobs": 1
    }'
```

`prompt_logprobs` is not available for streaming completion requests. See the
upstream [Sampling Parameters](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)
documentation for the corresponding engine-level options.

### Router Replay (R3)

For supported MoE workflows, start the server with routed-expert capture:

```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-return-routed-experts
```

For a non-streaming completion response, `choices[].routed_experts` is a
base64-encoded NumPy array, not an inline tensor. After decoding, its shape is
`(num_tokens - 1, num_layers, num_experts_per_tok)`. The final sampled token
has no routing record because it has not passed through the model yet. The
field is `null` when capture is disabled or the request is aborted before a
forward pass.

Enable this option only when the trainer consumes and replays the captured
expert choices. See the upstream
[routed experts example](https://docs.vllm.ai/en/latest/examples/rl/routed_experts_e2e/)
for decoding and replay logic.

## Deterministic rollouts

Batch invariance reduces output differences caused by changes in batch shape
or request order. On Atlas A2 and A3, build vLLM-Ascend with
`COMPILE_CUSTOM_KERNELS=1`, then enable batch invariance through the unified
RL configuration:

```bash
vllm serve Qwen/Qwen3-8B \
    --additional-config '{"rl_config": {"enabled": true, "enable_batch_invariant": true}}' \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

RL mode disables FRACTAL_NZ. When batch invariance is enabled, worker
initialization also sets `HCCL_DETERMINISTIC=strict` and
`LCCL_DETERMINISTIC=1`. See
[Batch Invariance](./batch_invariance.md) for supported hardware, models, and
limitations.

## Tokens in, tokens out

The experimental tokens-only endpoint accepts pre-tokenized prompts and
returns generated token IDs. Start the normal `serve` command with
`--tokens-only`:

```bash
vllm serve Qwen/Qwen3-0.6B --tokens-only
```

Send requests to `/inference/v1/generate`:

```bash
curl http://127.0.0.1:8000/inference/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
        "request_id": "rollout-001",
        "token_ids": [151644, 8948, 198],
        "sampling_params": {
            "temperature": 1.0,
            "max_tokens": 32,
            "logprobs": 1
        }
    }'
```

A non-streaming response has the following shape:

```json
{
  "request_id": "rollout-001",
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop",
      "token_ids": [123, 456],
      "routed_experts": null
    }
  ],
  "prompt_logprobs": null,
  "usage": null
}
```

The exact logprob and usage values depend on the sampling parameters and
server configuration. The API is intended for disaggregated serving and may
change; track the upstream [Tokens API RFC](https://github.com/vllm-project/vllm/issues/22817)
for its status.

## Tool calling for agentic RL

Agentic rollout environments can use the standard OpenAI-compatible tool-call
API. For a model with a supported parser:

```bash
vllm serve Qwen/Qwen3-8B \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

The environment supplies `tools` in `/v1/chat/completions`, executes returned
tool calls, appends tool results to the conversation, and submits the next
turn. See upstream [Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
for request examples and the supported parser/model combinations.

## Data-parallel request routing

vLLM's OpenAI serving layer recognizes the `X-data-parallel-rank` header and
passes its integer value to the scheduler. A DP-aware external router can use
this header to target a rank:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "X-data-parallel-rank: 2" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Your prompt here",
        "max_tokens": 32
    }'
```

vLLM-Ascend does not provide a `vllm serve router` command or a
`--dp-routing-config` option. Configure the vLLM data-parallel deployment and
an external router separately. See upstream
[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
for the supported load-balancing modes.

## Configuration checklist

| Setting | When to use it | Purpose |
| --- | --- | --- |
| `rl_config.enabled=true` | All RL deployments | Apply RL defaults: disable NZ, enable development endpoints, and remove `expandable_segments` |
| `--enable-sleep-mode` | Trainer and rollout engine share NPUs | Enable the CaMemAllocator memory pool |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Sleep mode | Use the supported worker start method |
| `rl_config.sleep_mode_extra_cleanup=true` | Optional for sleep mode | Release HCCL and ACL graph resources at the cost of slower wake-up |
| `rl_config.enable_training_consistency=true` | Training-inference consistency | Select FA3; requires the `flash_attn_npu_v3` package |
| `rl_config.enable_batch_invariant=true` | Reproducible rollouts | Enable batch-invariant kernels and deterministic communication settings |
| `--weight-transfer-config '{"backend": "hccl"}'` | Cross-NPU weight transfer | Register the Ascend HCCL transfer engine |
| `--weight-transfer-config '{"backend": "ipc"}'` | Same-NPU weight transfer | Select the Ascend NPU IPC transfer engine |
| `--enable-return-routed-experts` | MoE router replay | Return encoded expert routing decisions |

For the Ascend-specific option definitions, see
[Additional Configuration](../configuration/additional_config.md) and
[Environment Variables](../configuration/env_vars.md).
