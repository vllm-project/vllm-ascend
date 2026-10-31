# Sleep Mode Guide

## Overview

Sleep Mode is an API designed to offload model weights and discard KV cache from NPU memory. This functionality is essential for reinforcement learning (RL) post-training workloads, particularly in online algorithms such as PPO, GRPO, or DPO. During training, the policy model typically performs autoregressive generation using inference engines like vLLM, followed by forward and backward passes for optimization.

Since the generation and training phases may employ different model parallelism strategies, it becomes crucial to free KV cache and even offload model parameters stored within vLLM during training. This ensures efficient memory utilization and avoids resource contention on the NPU.

## Getting started

With `enable_sleep_mode=True`, the way we manage memory (malloc, free) in vLLM is under a specific memory pool. During model loading and KV cache initialization, we tag the memory as a map: `{"weight": data, "kv_cache": data}`.

The engine (v0/v1) supports two sleep levels to manage memory during idle periods:

- Level 1 Sleep
    - Action: Offloads model weights and discards the KV cache.
    - Memory: Model weights are moved to CPU memory; KV cache is forgotten.
    - Use Case: Suitable when reusing the same model later.
    - Note: Ensure sufficient CPU memory is available to hold the model weights.

- Level 2 Sleep
    - Action: Discards both model weights and KV cache.
    - Memory: The content of both the model weights and KV cache is forgotten.
    - Use Case: Ideal when switching to a different model or updating the current one.

Since this feature uses the low-level API [AscendCL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/appdevgapi/appdevgapi_07_0000.html), in order to use sleep mode, you should follow the [installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) and build from source. If you are using < v0.12.0rc1, remember to set `export COMPILE_CUSTOM_KERNELS=1`.

## Optional extra cleanup

By default, sleep mode only releases memory managed by the sleep-mode allocator. For RL workloads that need to return more NPU memory to the trainer, vLLM Ascend also provides an optional extra cleanup path:

```python
llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    enable_sleep_mode=True,
    additional_config={
        "rl_config": {
            "enabled": True,
            "sleep_mode_extra_cleanup": True,
        }
    },
)
```

For online serving, pass the same option through `--additional-config`:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-sleep-mode \
    --additional-config '{"rl_config": {"enabled": true, "sleep_mode_extra_cleanup": true}}'
```

When `rl_config.sleep_mode_extra_cleanup` is enabled, `sleep()` additionally:

- clears ACL graph attention workspaces and invalidates captured ACL graph caches when ACL graph is enabled;
- resets the model runner graph manager so ACL graphs can be captured again after wakeup;
- waits for pending pipeline-parallel send work, synchronizes the NPU, and destroys HCCL process groups.

During `wake_up()`, vLLM Ascend restores the HCCL process groups, refreshes MoE dispatcher HCCL metadata, restores sleep-mode allocator memory, and recaptures ACL graphs when needed.

!!! note

    Extra cleanup trades lower sleep-time NPU memory usage for longer wakeup latency. By default, if ACL graph is enabled, `wake_up()` must call `capture_model()` again after the model state has been restored. The experimental graph restoration option below can avoid this recapture for eligible graphs. The former top-level `enable_sleep_mode_extra_cleanup` option has been removed.

For level 2 sleep, wakeup can be split into two phases:

```python
llm.wake_up(tags=["weights"])
# Reload or update model weights here.
llm.wake_up(tags=["kv_cache"])
```

With extra cleanup enabled, ACL graphs are recaptured only when `tags` is `None` or contains `"kv_cache"`. This avoids recapturing graphs before externally reloaded weights and KV-cache state are ready.

## Experimental A5 MRV2 graph restoration

On shutdown, the reuse worker retires its graphs and explicitly releases managed
physical allocations before Python finalization. Late tensor owners retain only
virtual allocation records until their normal free callbacks run. Shutdown is
terminal: inference and wake-up require a new worker afterward.

Worker initialization and compilation preserve the caller's SIGTERM/SIGINT
handlers across lazy backend imports. Some torch-npu backend versions replace
these handlers at import time, preventing the default single-card EngineCore
from entering its shutdown protocol. Backend atexit cleanup remains registered.

Resource teardown and request draining have separate deadlines. On the tested
A5 environment, device finalization took longer than the default best-effort
process exit window, with graph reuse both enabled and disabled. Offline
validation sets the existing vLLM `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=35`
before starting the engine and explicitly waits up to 45 seconds with
`llm.llm_engine.engine_core.shutdown(timeout=45)` on the pinned vLLM revision.
This internal engine-client call is a validation cleanup step, not a new public
API. Record actual exit times and process exit codes; extending a deadline
alone does not establish successful cleanup. Use the serving application's
supported shutdown lifecycle in production.

Set `additional_config.rl_config.sleep_mode_graph_reuse=True` to enable this
experimental path. It defaults to `False` and requires `enable_sleep_mode=True`,
`rl_config.enabled=True`, and `rl_config.sleep_mode_extra_cleanup=True`.
Select Model Runner V2 with `VLLM_USE_V2_MODEL_RUNNER=1` before starting the engine.

The initial support boundary is A5, Model Runner V2, unquantized BF16 Qwen2,
Qwen3, or Qwen3 MoE with ordinary Ascend attention, `FULL_DECODE_ONLY`, and fixed
capture sizes drawn from `[1, 2, 4, 8]`. Tensor parallelism is limited to one or
two ranks, with expert parallelism disabled. MLA, multimodal models, speculative
decoding, LoRA, PP/CP/DP, weight offload, transfer connectors, dynamic EPLB, and
super kernels use recapture. Other attention backends and model families also
use recapture; the decision log includes the reason.

```python
llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype="bfloat16",
    enable_sleep_mode=True,
    compilation_config={
        "mode": 3,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    },
    additional_config={
        "rl_config": {
            "enabled": True,
            "sleep_mode_extra_cleanup": True,
            "sleep_mode_graph_reuse": True,
        },
    },
)
```

Eligible captures use a dedicated CaMem pool. Sleep copies the complete managed
graph pool to pinned CPU memory and unmaps its physical memory, preserving the
virtual addresses. The graph objects, bound events, and business HCCL groups
remain alive. Consequently, sleep retains communication and other unmanaged
memory, and uses additional CPU memory for the graph backup. This is a latency
and memory tradeoff; it does not release all graph or communication resources.

Every rank checks the capture configuration, tensor layouts and addresses,
graph/event ownership, pool records, and communicator identities before sleep
and again after restoration. Same-address weight updates are allowed. Changed
storage, events, or configuration invalidate the certificate, and all ranks
recapture together before inference resumes. Known graph-only restoration
failures also recapture. An unknown native mapping state fails the worker and
requires restart; the old graph is never replayed in that state.

The transaction moves from `READY` to `ASLEEP`, optionally `PARTIAL`, then
`RESTORING`, and finally `READY`. Inference is rejected while the transaction
is incomplete. Repeating sleep at the same level or wake after complete restore
is idempotent. Level 2 supports the `weights` / external weight reload /
`kv_cache` sequence above: restoring weights alone never enables replay.
Control transitions and worker inference entry points share a lock; device
synchronization drains work queued before sleep.

To return to the existing cleanup and recapture behavior, set
`sleep_mode_graph_reuse=False` and start a new engine. No additional HTTP
endpoint is introduced.

## Prepare Model Weights

Use the `Qwen2.5-0.5B-Instruct` model weights. With `VLLM_USE_MODELSCOPE=True`, the model will be downloaded automatically from ModelScope.

```{list-table}
:header-rows: 1

* - Model
  - ModelScope Link
* - Qwen2.5-0.5B-Instruct
  - [Qwen/Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)
```

## Usage

The following is a simple example of how to use sleep mode.

- Offline inference:

    ```python
    import os

    import torch
    from vllm import LLM, SamplingParams
    from vllm.utils.mem_constants import GiB_bytes

    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if __name__ == "__main__":
        prompt = "How are you?"

        free, total = torch.npu.mem_get_info()
        print(f"Free memory before sleep: {free / 1024 ** 3:.2f} GiB")
        # record npu memory use baseline in case other process is running
        used_bytes_baseline = total - free
        llm = LLM(
            "Qwen/Qwen2.5-0.5B-Instruct",
            enable_sleep_mode=True,
            additional_config={"weight_nz_mode": 0},
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        output = llm.generate(prompt, sampling_params)

        llm.sleep(level=1)

        free_npu_bytes_after_sleep, total = torch.npu.mem_get_info()
        print(f"Free memory after sleep: {free_npu_bytes_after_sleep / 1024 ** 3:.2f} GiB")
        used_bytes = total - free_npu_bytes_after_sleep - used_bytes_baseline
        # now the memory usage should be less than the model weights
        # (0.5B model, 1GiB weights)
        assert used_bytes < 1 * GiB_bytes

        llm.wake_up()
        output2 = llm.generate(prompt, sampling_params)
        # cmp output
        assert output[0].outputs[0].text == output2[0].outputs[0].text
    ```

- Online serving:
    !!! note

            Considering there may be a risk of malicious access, please make sure you are under a dev-mode, and explicitly specify the dev environment `VLLM_SERVER_DEV_MODE` to expose these endpoints (sleep/wake up).

    ```bash
    export VLLM_SERVER_DEV_MODE="1"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export VLLM_USE_MODELSCOPE="True"

    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
        --enable-sleep-mode \
        --additional-config='{"weight_nz_mode": 0}'

    # after serving is up, post to these endpoints.
    # /sleep reads level from the query string (JSON body is ignored).

    # --- Level 1: offload weights, discard KV cache ---
    curl -X POST "http://127.0.0.1:8000/sleep?level=1"
    curl -X GET http://127.0.0.1:8000/is_sleeping

    # wake all tags (weights + kv_cache)
    curl -X POST http://127.0.0.1:8000/wake_up
    curl -X GET http://127.0.0.1:8000/is_sleeping

    # serving is available again after Level-1 wake_up
    curl http://127.0.0.1:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt": "The future of AI is",
            "max_tokens": 7,
            "temperature": 0
        }'

    # --- Level 2: discard weights and KV cache ---
    # tags must be in ["weights", "kv_cache"]. After waking weights,
    # reload the checkpoint on every worker before waking kv_cache.
    curl -X POST "http://127.0.0.1:8000/sleep?level=2"
    curl -X POST "http://127.0.0.1:8000/wake_up?tags=weights"
    curl -X POST http://127.0.0.1:8000/collective_rpc \
        -H "Content-Type: application/json" \
        -d '{
            "method": "reload_weights",
            "kwargs": {"weights_path": "Qwen/Qwen2.5-0.5B-Instruct"}
        }'
    curl -X POST "http://127.0.0.1:8000/wake_up?tags=kv_cache"
    ```
