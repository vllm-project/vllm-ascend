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

vLLM-Ascend exposes the following Ascend-specific weight-transfer backends:

| Backend | Use case | Transport |
| --- | --- | --- |
| `ipc` | Trainer and rollout engine use the same physical NPU | NPU IPC handles |
| `hccl` | Trainer and rollout engine use separate NPUs | HCCL collectives |
| `sparse_hccl` | Replace selected checkpoint weight values on separate NPUs | HCCL collectives |

The table describes transport choices. Sparse transfer requires the trainer and
rollout engine to start from the same checkpoint.

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
5. finish the weight update;
6. resume generation.

The repository contains runnable examples for weight transfer:

- [`rlhf_http_hccl.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rl/rlhf_http_hccl.py)
- [`rlhf_http_npu_ipc.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rl/rlhf_http_npu_ipc.py)
- [`rlhf_async_new_apis.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rl/rlhf_async_new_apis.py)
- [`rlhf_sparse_hccl.py`](https://github.com/vllm-project/vllm-ascend/blob/182ea0448550a970e4f0ae3624fe35e38b37bc80/examples/rl/rlhf_sparse_hccl.py)

FRACTAL_NZ must be disabled for weight updates. RL mode forces
`weight_nz_mode=0`; the weight-transfer start path validates the effective
Ascend configuration.

#### Sparse HCCL

For sparse checkpoint updates, follow the upstream [Sparse NCCL
guide](https://docs.vllm.ai/en/latest/training/weight_transfer/nccl/#sparse-nccl)
for patch semantics and the update lifecycle. On Ascend, select `sparse_hccl`
and enable RL mode:

```bash
vllm serve /path/to/checkpoint \
    --weight-transfer-config '{"backend": "sparse_hccl"}' \
    --additional-config '{"rl_config": {"enabled": true}}'
```

Run `rlhf_sparse_hccl.py` to initialize the trainer and send sparse checkpoint
updates. Its TP2 launch commands explicitly enable expert parallelism. Reserve
a separate NPU for the trainer; the transfer group size is `1 + TP * DP`, with
no additional EP factor. Pipeline parallelism is currently unsupported.

Sparse HCCL requires unquantized ND weights and static expert placement with
EPLB disabled. FRACTAL_NZ, fused MC2 expert weight lists, draft models, and packed
sparse updates are unsupported. Updates apply in place: after a failed update,
keep generation paused until the baseline is restored or the engine restarts.

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

### Sampling Mask (Distribution Replay)

During RL rollout inference, top-k/top-p/min-p sampling truncates the model's
distribution. Sampling Mask returns the retained token IDs for each generated
position, so training can normalize policy probabilities over the same support
used during rollout. See the upstream
[Sampling Mask (Distribution Replay) guide](https://docs.vllm.ai/en/latest/training/sampling_mask/)
for training-side usage.

`--return-sampling-mask` enables the feature and is disabled by default.
`--logprobs-mode processed_logprobs` is required.
The following example also uses `--tokens-only` to expose the token-in/token-out
API and skip tokenizer initialization. Use `--enable-scale-out` instead to expose the
same endpoint while retaining the normal serving setup and tokenizer.
See [Tokens in, tokens out](#tokens-in-tokens-out) for serving details.

```bash
vllm serve Qwen/Qwen3-0.6B \
    --tokens-only \
    --return-sampling-mask \
    --logprobs-mode processed_logprobs
```

With sampling-mask capture enabled, requests must use
`temperature > 0` and `top_k > 0`. Send pre-tokenized input to
`/inference/v1/generate`:

```bash
curl -sS -X POST http://127.0.0.1:8000/inference/v1/generate \
    -H 'Content-Type: application/json' \
    -d '{
        "token_ids": [785, 6722, 315, 9625, 374],
        "sampling_params": {
            "temperature": 1.0,
            "top_k": 20,
            "top_p": 0.95,
            "max_tokens": 2,
            "logprobs": 1
        },
        "output_mode": "tokens",
        "stream": false
    }'
```

Each choice returns one support set per generated token. The following tested
Qwen3-0.6B response excerpt omits metadata, ranks, and `top_logprobs`:

```json
{
  "finish_reason": "length",
  "token_ids": [12095, 11],
  "sampling_mask": [
    [220, 264, 279, 304, 320, 510, 1112, 1181, 1304, 2130,
     2598, 7407, 7436, 9625, 12095, 21718, 24209, 30743, 32671, 55201],
    [11, 13, 323, 382]
  ],
  "logprobs": {
    "content": [
      {"token_id": 12095, "logprob": -0.2592906951904297},
      {"token_id": 11, "logprob": -0.7923774719238281}
    ]
  }
}
```

Basic distribution replay needs the complete mask and the sampled token's old
logprob. During training, apply the saved mask to the current policy's logits
to compute that token's new logprob. Old logprobs for every candidate token are
not required.

The sampled token belongs to its corresponding support. Mask size is independent
of `logprobs`, and top-k ties can retain more than `top_k` tokens. With
`stream=true`, concatenate token and mask chunks by choice index and token
position.

For an MTP-capable model, speculative decoding is supported with fixed
verification boundaries. Add the following configuration to the server command:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"enable_adaptive_verification":false}'
```

Each committed token receives the processed target-model support at its
position. The mask does not record draft or acceptance probabilities.

On Ascend, supports are packed on the NPU and decoded on the CPU using the
exact bitmask fallback. This preserves the complete support but adds transfer,
CPU decoding, and response-size overhead. The adapter requires the upstream
request-major sampling-mask interface.

Diffusion models, custom logits processors, adaptive verification, and the
batch-sharded LM-head gather path are not supported for this feature.
Sampling masks contain neither probability values nor RNG state; they do not
force a token sequence as [Trace Replay](#trace-replay) does.

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

The experimental token-in/token-out (TITO) endpoint accepts pre-tokenized
prompts and, by default (`output_mode="tokens"`), returns generated token IDs.
Apply tokenization and any chat template on the client before sending
`token_ids`. Choose one of two serving modes:

```bash
# Dedicated token-only serving: skips tokenizer initialization.
vllm serve Qwen/Qwen3-0.6B --tokens-only

# Enable TITO while retaining the normal serving setup and tokenizer.
vllm serve Qwen/Qwen3-0.6B --enable-scale-out
```

`--tokens-only` uses token output and does not support stop strings; use
`sampling_params.stop_token_ids` instead. A normal server started without
either flag returns HTTP 403 for TITO requests, with a message to enable
`--enable-scale-out`.

Send a non-streaming request to `/inference/v1/generate`:

```bash
curl http://127.0.0.1:8000/inference/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
        "request_id": "rollout-001",
        "token_ids": [151644, 8948, 198],
        "output_mode": "tokens",
        "sampling_params": {
            "temperature": 1.0,
            "max_tokens": 2,
            "logprobs": 1
        }
    }'
```

The following non-streaming response excerpt illustrates the structure for a
request that reaches the two-token limit. Token IDs and logprob values are
illustrative, not captured model output; other response fields are omitted:

```json
{
  "request_id": "generate-tokens-rollout-001",
  "model": "Qwen/Qwen3-0.6B",
  "output_mode": "tokens",
  "choices": [
    {
      "index": 0,
      "logprobs": {
        "content": [
          {
            "token_id": 123,
            "logprob": -0.5,
            "rank": 1,
            "top_logprobs": [
              {"token_id": 123, "logprob": -0.5, "rank": 1}
            ]
          },
          {
            "token_id": 456,
            "logprob": -0.7,
            "rank": 1,
            "top_logprobs": [
              {"token_id": 456, "logprob": -0.7, "rank": 1}
            ]
          }
        ]
      },
      "finish_reason": "length",
      "stop_reason": null,
      "token_ids": [123, 456],
      "routed_experts": null
    }
  ],
  "prompt_logprobs": null,
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 2,
    "total_tokens": 5
  }
}
```

In token output mode, `choices[].logprobs.content` contains an integer
`token_id`, its `logprob` and `rank`, and `top_logprobs` for each generated
position. Non-streaming responses include `usage` without requiring
`stream_options.include_usage`. Here, three prompt tokens and two completion
tokens give five total tokens; `stop_reason` is `null` because generation
reached `max_tokens`.

Set `stream=true` to receive server-sent events (SSE), ending with
`data: [DONE]`. To cancel an in-flight request, send a POST to
`/inference/v1/abort_requests` with a JSON body containing `request_ids`, a
list of IDs returned by the server (including the `generate-tokens-` prefix),
rather than the original client-supplied IDs.

For the full protocol and the `text` output mode, which requires a tokenizer,
see the upstream
[TITO guide](https://docs.vllm.ai/en/latest/serving/online_serving/token_in_token_out/).
The [Tokens API RFC](https://github.com/vllm-project/vllm/issues/22817)
provides design background.

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

## Trace Replay

Trace replay forces the engine to emit a predetermined token sequence during
decoding while computing real logprobs from the model's unmodified logit
distribution. When using the normal
`vllm serve` command with the token-in/token-out endpoint, enable the scale-out
endpoints:

```bash
vllm serve Qwen/Qwen3-8B \
    --enable-scale-out \
    --enable-trace-replay
```

`--enable-scale-out` registers the
`/inference/v1/generate` endpoint for a normal serving process.
`--enable-trace-replay` opts the server into accepting trace replay requests.

Send requests to `/inference/v1/generate`:

```bash
curl http://127.0.0.1:8000/inference/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
        "request_id": "rollout-001",
        "token_ids": [151644, 8948, 198],
        "sampling_params": {
            "trace_decode_token_ids": [15, 284, 1026, 374],
            "logprobs": 1
        }
    }'
```

The output tokens will always be `[15, 284, 1026, 374]`. The logprobs reflect the model's true probability for each forced token under the current inference configuration.

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
| `--return-sampling-mask` | Sampling Mask (Distribution Replay) | Return the support set for each generated token; disabled by default |
| `--enable-scale-out` | Token-in/token-out requests on a normal server | Register `/inference/v1/generate`; `--tokens-only` also registers this endpoint |
| `--enable-trace-replay` | trace replay | Forces decoding to follow a predetermined token sequence |

For the Ascend-specific option definitions, see
[Additional Configuration](../configuration/additional_config.md) and
[Environment Variables](../configuration/env_vars.md).
