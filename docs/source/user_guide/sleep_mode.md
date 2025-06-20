# Sleep Mode

## Overview

Sleep Mode is the API which can selectively exposed to offload weight, discard kv cache from NPU memory, and it is strong needed for RL post-training workloads, in online PPO (or GRPO, online DPO), the policy model will perform auto-regressive generation (using vLLM or other inference engines) and fwd + bwd computation with training infrastructure. Therefore, in the training stage, so it is necessary to free the KVCache and even offload the model parameter stored in the vLLM (as the model parallel strategies during generation and training could be different).

## Getting started

This module provides a custom memory allocator for Ascend NPUs using the [CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/appdevgapi/appdevgapi_07_0000.html) runtime. It integrates tightly with PyTorch via `torch.npu.memory.NPUPluggableAllocator` and supports a "sleep mode", which allows tensors to offload memory to the CPU and release NPU memory when it's no longer immediately needed. This improves memory efficiency and allows large-scale inference to run in constrained environments.

With `enable_sleep_mode=True`, the way we manage memory(malloc, free) in vllm will under the `use_memory_pool` Context Managers, and all memory allocation created inside the context will be allocated, in the memory pool, and has the specified tag.

```bash
+-------------------+            +---------------------------+          +----------------------------+
|    Python Layer   |  ----->    |   CaMemAllocator (class)  |  --->    | C Extension (vllm_ascend_C)|
+-------------------+            +---------------------------+          +----------------------------+
    ⬇ Registers                      ⬇ Tracks & Tags                         ⬇ Calls into CANN
init_module(malloc, free)         pointer_to_data[ptr] = data         aclrtMallocPhysical, aclrtMapMem, etc.
```

Since this feature uses the AscendCL API, in order to use sleep mode, you should follow the [installation guide](https://vllm-ascend.readthedocs.io/en/latest/installation.html) and building from source, if you are using v0.7.3, remember to set `export COMPILE_CUSTOM_KERNELS=1`, for the latest version(v0.9.x+), the environment variable COMPILE_CUSTOM_KERNELS will be set 1 by default while building from source.

## Usage

Let's take the default parameters of v1 engine as an example

- For offline inference:
```python
import os

import torch
from vllm import LLM, SamplingParams
from vllm.utils import GiB_bytes


os.environ["VLLM_USE_V1"] = "1"
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

- For online serving:
Considering there may be a risk of malicious access, please make sure you are under a dev-mode, and explicit specify the develop env: `VLLM_SERVER_DEV_MODE` to expose these endpoints(sleep/wake up).

server command:
```bash
export VLLM_SERVER_DEV_MODE="1"
export VLLM_USE_V1="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_MODELSCOPE="True"

vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-sleep-mode

# after serveing is up, post these endpoints

# 1. sleep
curl -X POST http://127.0.0.1:8000/sleep

curl -X POST http://127.0.0.1:8000/is_sleeping

# 2. wake up
curl -X POST http://127.0.0.1:8000/wake_up
curl -X POST http://127.0.0.1:8000/is_sleeping

```