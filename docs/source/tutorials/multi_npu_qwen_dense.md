# Multi-NPU (Qwen3-32B-W8A8)

## Getting Start
Welcome to the tutorial on optimizing Qwen Dense models in the vLLM-Ascend environment. This guide will help you configure the most effective settings for your use case, with practical examples that highlight key optimization points. We will also explore how adjusting service parameters can maximize throughput performance across various scenarios.

By the end of this tutorial, you’ll be equipped to fine-tune Qwen Dense models for optimal performance in different production environments. Let’s get started!

## Key Optimization Points
In this section, we will cover the key optimization points that can significantly improve the performance of Qwen Dense models. These techniques are designed to enhance throughput and efficiency across various scenarios.

### 1. Rope Optimization
Rope optimization enhances the model's efficiency by modifying the position encoding process. Specifically, it ensures that the cos_sin_cache and the associated index selection operation are only performed during the first layer of the forward pass. For subsequent layers, the position encoding is directly reused, eliminating redundant calculations and significantly speeding up inference in decode phase.

This optimization is enabled by default and does not require any additional environment variables to be set.

### 2. AddRMSNormQuant Fusion
AddRMSNormQuant fusion merges the Address-wise Multi-Scale Normalization and Quantization operations, allowing for more efficient memory access and computation, thereby enhancing throughput.

This optimization is enabled by default and does not require any additional environment variables to be set.

### 3. FlashComm_v1
FlashComm_v1 significantly improves performance in large-batch scenarios by decomposing the traditional allreduce collective communication into reduce-scatter and all-gather. This breakdown helps reduce the computation of the RMSNorm token dimensions, leading to more efficient processing. In quantization scenarios, FlashComm_v1 also reduces the communication overhead by decreasing the bit-level data transfer, which further minimizes the end-to-end latency during the prefill phase.

It is important to note that the decomposition of the allreduce communication into reduce-scatter and all-gather operations only provides benefits in high-concurrency scenarios, where there is no significant communication degradation. In other cases, this decomposition may result in noticeable performance degradation. To mitigate this, the current implementation uses a threshold-based approach, where FlashComm_v1 is only enabled if the actual token count for each inference schedule exceeds the threshold. This ensures that the feature is only activated in scenarios where it improves performance, avoiding potential degradation in lower-concurrency situations.

This optimization requires setting the environment variable `VLLM_ASCEND_ENABLE_FLASHCOMM1 = 1` to be enabled.

### 4. Matmul and ReduceScatter Fusion
Once FlashComm_v1 is enabled, an additional optimization can be applied. This optimization fuses matrix multiplication and ReduceScatter operations, along with tiling optimization. The Matmul computation is treated as one pipeline, while the ReduceScatter and dequant operations are handled in a separate pipeline. This approach significantly reduces communication steps, improves computational efficiency, and allows for better resource utilization, resulting in enhanced throughput, especially in large-scale distributed environments.

This optimization is automatically enabled once FlashComm_v1 is activated. However, due to an issue with performance degradation in small-concurrency scenarios after this fusion, a threshold-based approach is currently used to mitigate this problem. The optimization is only applied when the token count exceeds the threshold, ensuring that it is not enabled in cases where it could negatively impact performance.

### 5. Weight Prefetching
Weight prefetching optimizes memory usage by preloading weights into the cache before they are needed, minimizing delays caused by memory access during model execution.

In dense model scenarios, the MLP's gate_up_proj and down_proj linear layers often exhibit relatively high MTE utilization. To address this, we create a separate pipeline specifically for weight prefetching, which runs in parallel with the original vector computation pipeline, such as RMSNorm and SiLU, before the MLP. This approach allows the weights to be preloaded to L2 cache ahead of time, reducing MTE utilization during the MLP computations and indirectly improving Cube computation efficiency by minimizing resource contention and optimizing data flow.

It is important to emphasize that, since we use vector computations to hide the weight prefetching pipeline, the setting of the prefetch buffer size is crucial. If the buffer size is too small, the optimization benefits will not be fully realized, while a larger buffer size may lead to resource contention, resulting in performance degradation. To accommodate different scenarios, we have exposed two environment variables `VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE` and `VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE` to allow for flexible buffer size configuration based on the specific workload.

This optimization requires setting the environment variable `VLLM_ASCEND_ENABLE_PREFETCH_MLP = 1` and `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE = 1` to be enabled.

### 6. Zerolike Elimination
This elimination removes unnecessary operations related to zero-like tensors in Attention forward, improving the efficiency of matrix operations and reducing memory usage.

This optimization is enabled by default and does not require any additional environment variables to be set.

### 7. FullGraph Optimization
ACLGraph offers several key optimizations to improve model execution efficiency. By replaying the entire model execution graph at once, we significantly reduce dispatch latency compared to multiple smaller replays. This approach also stabilizes multi-device performance, as capturing the model as a single static graph mitigates dispatch fluctuations across devices. Additionally, consolidating graph captures frees up streams, allowing for the capture of more graphs and optimizing resource usage, ultimately leading to improved system efficiency and reduced overhead.

The configuration compilation_config = { "cudagraph_mode": "FULL_DECODE_ONLY"} is used when starting the service. This setup is necessary to enable the aclgraph's full decode-only mode.

### 8. Asynchronous Scheduling
Asynchronous scheduling is a technique used to optimize inference efficiency. It allows non-blocking task scheduling to improve concurrency and throughput, especially when processing large-scale models.

This optimization is enabled by setting --async-scheduling

## Real-World Example

In this section, we will demonstrate best practices for adjusting hyperparameters in vLLM-Ascend to maximize inference throughput performance. By tailoring service-level configurations to fit different use cases, you can ensure that your system performs optimally across various scenarios. We will guide you through how to fine-tune hyperparameters based on observed phenomena, such as max_model_len, max_num_batched_tokens, and cudagraph_capture_sizes, to achieve the best performance.

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

Setup environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

### Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

For an Atlas A2 with 64GB of NPU card memory, tensor-parallel-size should be at least 2, and for 32GB of memory, tensor-parallel-size should be at least 4.

```bash
vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 4 --enable_expert_parallel
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-30B-A3B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 4096
}'
```

### Offline Inference on Multi-NPU

Run the following script to execute offline inference on multi-NPU:

```python
import gc
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)
llm = LLM(model="Qwen/Qwen3-30B-A3B",
          tensor_parallel_size=4,
          distributed_executor_backend="mp",
          max_model_len=4096,
          enable_expert_parallel=True)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: " Lucy. I'm from the UK and I'm 11 years old."
Prompt: 'The future of AI is', Generated text: ' a topic that has captured the imagination of scientists, philosophers, and the general public'
```
