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

The specific example scenario is as follows:
- The machine environment is an Atlas 800 A3 (64G*16)
- The LLM is Qwen3-32B-W8A8
- The data scenario is a fixed-length input of 3.5K and an output of 1.5K.
- The parallel configuration requirement is DP=1&TP=4

### Run docker container:

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

### Setup environment variables

```bash
# Set vLLM to Engine V1
export VLLM_USE_V1=1

# set the NPU device number
export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15

# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1

# Performance optimization of memory management
# Install it using `apt install libjemalloc2`
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# Enable the AIVector core to directly schedule ROCE communication
export HCCL_OP_EXPANSION_MODE="AIV"

# Enable dense model and general optimizations for better performance.
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1

# Enable FlashComm_v1 optimization when tensor parallel is enabled.
export VLLM_ASCEND_ENABLE_FLASHCOMM=1

# Enable MLP weight prefetch, only used in small concurrency.
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# Set buffer size to 18 MB for gate up prefetch
export VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE=18874368

# Set buffer size to 18 MB for down proj prefetch
export VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE=18874368
```

### Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

This script is configured to achieve optimal performance under the above specific example scenarios, with a single card throughput of 650 TPS, TPOT < 50ms, TTFT < 2s and batchsize = 72 on two A3 cards.

```bash
vllm serve /model/Qwen3-32B-W8A8 \
  --served-model-name qwen3 \
  --trust-remote-code \
  --async-scheduling \
  --quantization ascend \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,8,24,48,60,64,72,76]}' \
  --port 8113 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8113/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3",
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
llm = LLM(model="Qwen/Qwen3-32B-W8A8",
          tensor_parallel_size=4,
          trust_remote_code=True,
          distributed_executor_backend="mp",
          max_model_len=5500,
          max_num_batched_tokens=5500,
          quantization="ascend",
          compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
```

## Optimization Highlights

Building on the specific example scenarios outlined earlier, this section highlights the key tuning points that played a crucial role in achieving optimal performance. By focusing on the most impactful adjustments to hyperparameters and optimizations, we’ll emphasize the strategies that can be leveraged to maximize throughput, minimize latency, and ensure efficient resource utilization in various environments. These insights will help guide you in fine-tuning your own configurations for the best possible results.

### 1.Prefetch Buffer Size
Setting the right prefetch buffer size is essential for optimizing weight loading and the size of this prefetch buffer is directly related to the time that can be hidden by vector computations. To achieve a near-perfect overlap between the prefetch and computation streams, you can flexibly adjust the buffer size by profiling and observing the degree of overlap at different buffer sizes.

For example, in the real-world scenario mentioned above, I set the prefetch buffer size for the gate_up_proj and down_proj in the MLP to 18MB. The reason for this is that, at this value, the vector computations of RMSNorm and SiLU can effectively hide the prefetch stream, thereby accelerating the Matmul computations of the two linear layers.

### 2.Max-model-len
The max-model-len parameter controls the maximum sequence length that the model can handle. Adjusting this value allows you to optimize both memory usage and processing time. Generally speaking, I recommend setting this value to the max_input_len + max_output_len in your data scenario, if these values are known. This helps reduce the input concurrency in the profile-run stage, thereby lowering GPU memory usage and leaving more memory space for the kv_cache and activation values.

For example, in the real-world scenario above, I set it to 5500, which is nearly the sum of 3.5K for the input and 1.5K for the output. After setting this, it’s clear from the service startup logs that more memory is allocated for the kv_cache. This also allows for a higher maximum concurrency during actual inference, which significantly improves end-to-end throughput.

### 3.Max-num-batched-tokens
The max-num-batched-tokens parameter determines the maximum number of tokens that can be processed in a single batch. Adjusting this value helps to balance throughput and memory usage. Setting this value too small can negatively impact end-to-end performance, as fewer tokens are processed per batch, potentially leading to inefficiencies. Conversely, setting it too large increases the risk of Out of Memory (OOM) errors due to excessive memory consumption.

In the above real-world scenario, we not only conducted extensive testing to determine the most cost-effective value, but also took into account the accumulation of decode tokens when enabling chunked prefill. If the value is set too small, a single request may be chunked multiple times, and during the early stages of inference, a batch may contain only a small number of decode tokens. This can result in the end-to-end throughput falling short of expectations.

### 4.Cudagraph_capture_sizes
The cudagraph_capture_sizes parameter controls the granularity of graph captures during the inference process. Adjusting this value determines how much of the computation graph is captured at once, which can significantly impact both performance and memory usage.

If this list is not manually specified, it will be filled with a series of evenly distributed values, which typically ensures good performance. However, if you want to fine-tune it further, manually specifying the values will yield better results. This is because if the batch size falls between two sizes, the framework will automatically pad the token count to the larger size. This often leads to actual performance deviating from the expected or even degrading.

Therefore, like the above real-world scenario, when adjusting the benchmark request concurrency, we always ensure that the concurrency is actually included in the cudagraph_capture_sizes list. This way, during the decode phase, padding operations are essentially avoided, ensuring the reliability of the experimental data. 

It’s important to note that if you enable FlashComm_v1, the values in this list must be integer multiples of the TP count. Any values that do not meet this condition will be automatically filtered out. Therefore, I recommend incrementally adding concurrency based on the TP count after enabling FlashComm_v1.