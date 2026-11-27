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

## Optional communicator memory cleanup

By default, sleep releases memory managed by the sleep-mode allocator. To also
release reclaimable HCCL process-group memory, use the upstream communicator
lifecycle option:

```python
llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    enable_sleep_mode=True,
    enable_nccl_comm_suspend=True,
    additional_config={"weight_nz_mode": 0},
)
```

For online serving:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-sleep-mode \
    --enable-nccl-comm-suspend \
    --additional-config '{"weight_nz_mode": 0}'
```

The option retains its upstream name on Ascend. When a cycle has HCCL PGs to
rebuild, the current backend
**destroys and restores process groups** through the communicator
suspend/resume interface. It does not use native `HcclCommSuspend` for memory
reclamation, and it does not preserve native communication-domain identity.
A temporary lifecycle anchor remains alive during sleep and recovery.

When ACL Graph is enabled and the cycle will rebuild HCCL, sleep invalidates
captured graphs and their workspaces before communication teardown. Wake restores communication groups
and MoE runtime metadata, then recaptures graphs once their inputs are ready.
Existing graph entries are not reused across a destroy/restore cycle. Eager
execution remains eager. This trades lower residual sleep memory for increased
wake latency; actual HBM savings depend on the workload and software stack.

When there are no HCCL PGs to rebuild, including on a single-rank worker,
communication cleanup leaves existing graphs and workspaces intact and wake
does not recapture them. The decision is made separately for every sleep cycle.

The legacy combination `rl_config.enabled=True` and
`rl_config.sleep_mode_extra_cleanup=True` remains accepted with a deprecation
warning and maps to the same upstream option. Enabling both old and new options
runs cleanup once. The proposed `sleep_hccl_policy`, `sleep_acl_graph_policy`,
and `sleep_allow_fallback` fields are not supported configuration options.

For staged wake, restore weights before KV cache:

```python
llm.wake_up(tags=["weights"])
# Reload or update model weights here for level-2 sleep.
llm.wake_up(tags=["kv_cache"])
```

Required graph recapture and anchor release happen at the final KV-cache wake.
Waking KV cache before weights is rejected when communicator cleanup is active. A full
`wake_up()` restores both. If communication or graph recovery fails, the error
is propagated; restart the worker rather than retrying a partially failed
lifecycle.

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
