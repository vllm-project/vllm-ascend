# Sleep Mode Guide

## Overview

Sleep Mode is an API designed to offload model weights and discard KV cache from NPU memory. This functionality is essential for reinforcement learning (RL) post-training workloads, particularly in online algorithms such as PPO, GRPO, or DPO. During training, the policy model typically performs auto-regressive generation using inference engines like vLLM, followed by forward and backward passes for optimization.

Since the generation and training phases may employ different model parallelism strategies, it becomes crucial to free KV cache and even offload model parameters stored within vLLM during training. This ensures efficient memory utilization and avoids resource contention on the NPU.

## Getting started

With `enable_sleep_mode=True`, the way we manage memory (malloc, free) in vllm is under a specific memory pool. During model loading and KV cache initialization, we tag the memory as a map: `{"weight": data, "kv_cache": data}`.

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

Since this feature uses the low-level API [AscendCL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/appdevgapi/appdevgapi_07_0000.html), in order to use sleep mode, you should follow the [installation guide](https://vllm-ascend.readthedocs.io/en/latest/installation.html) and build from source. If you are using v0.7.3, remember to set `export COMPILE_CUSTOM_KERNELS=1`. For the latest version (v0.9.x+), the environment variable `COMPILE_CUSTOM_KERNELS` will be set to 1 by default while building from source.

## Usage

The following is a simple example of how to use sleep mode.

- Offline inference:

    ```python
    import os

    import torch
    from vllm import LLM, SamplingParams
    from vllm.utils import GiB_bytes


    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if __name__ == "__main__":
        prompt = "How are you?"

        free, total = torch.npu.mem_get_info()
        print(f"Free memory before sleep: {free / 1024 ** 3:.2f} GiB")
        # record npu memory use baseline in case other process is running
        used_bytes_baseline = total - free
        llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)
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
    :::{note}
    Considering there may be a risk of malicious access, please make sure you are under a dev-mode, and explicit specify the dev environment `VLLM_SERVER_DEV_MODE` to expose these endpoints (sleep/wake up).
    :::

    ```bash
    export VLLM_SERVER_DEV_MODE="1"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export VLLM_USE_MODELSCOPE="True"

    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-sleep-mode

    # after serveing is up, post these endpoints

    # sleep level 1
    curl -X POST http://127.0.0.1:8000/sleep \
        -H "Content-Type: application/json" \
        -d '{"level": "1"}'

    curl -X GET http://127.0.0.1:8000/is_sleeping

    # sleep level 2
    curl -X POST http://127.0.0.1:8000/sleep \
        -H "Content-Type: application/json" \
        -d '{"level": "2"}'

    # wake up
    curl -X POST http://127.0.0.1:8000/wake_up

    # wake up with tag, tags must be in ["weights", "kv_cache"]
    curl -X POST "http://127.0.0.1:8000/wake_up?tags=weights"

    curl -X GET http://127.0.0.1:8000/is_sleeping

    # after sleep and wake up, the serving is still available
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt": "The future of AI is",
            "max_tokens": 7,
            "temperature": 0
        }'
    ```
