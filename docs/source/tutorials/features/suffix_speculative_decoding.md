# Suffix Speculative Decoding

## **Introduction**

Suffix Decoding is an optimization technique for speculative decoding based on pattern matching. It simultaneously retrieves repetitive sequences from both the prompt and the generated content, using frequency statistics to predict the most likely token continuations. Unlike traditional speculative decoding methods, Suffix Decoding runs entirely on the CPU, eliminating the need for additional GPU resources or draft models, which results in superior acceleration for repetitive tasks such as AI agents and code generation.

This document provides step-by-step guidance on how to deploy and benchmark the Suffix Decoding speculative inference technology supported by `vllm-ascend` on Atlas A2 hardware. The setup utilizes a single Atlas 800T A2 node with a 4-card deployment of the Qwen3-32B model instance. Benchmarking is conducted using authentic open-source datasets covering the following categories:

| **Dataset Category**           | **Dataset Name** |
| ------------------------------ | ---------------- |
| Code Generation                | HumanEval        |
| Common Sense Reasoning         | ARC              |
| Mathematical Reasoning         | gsm8k            |
| Natural Language Understanding | SuperGLUE_BoolQ  |
| Comprehensive Examination      | agieval          |
| Multi-turn Dialogue            | sharegpt         |

The benchmarking tool used in this tutorial is AISBench, which supports performance testing for all the datasets listed above. The final section of this tutorial presents a performance comparison between enabling and disabling Suffix Decoding under the condition of satisfying an SLO TPOT < 50ms across different datasets and concurrency levels. Validations demonstrate that the Qwen3-32B model achieves a throughput improvement of approximately 20% to 80% on various real-world datasets when Suffix Decoding is enabled.

## **Download vllm-ascend Image**

This tutorial uses the official image, version v0.13.0rc1. Use the following command to download:

```bash
docker pull quay.io/ascend/vllm-ascend:v0.13.0rc1
```

## **Run with Docker**

Container startup command:

```bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc1
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses four Atlas A2 NPU cards to create the container.
# Mount the hccn.conf file from the host node into the container.

docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:\
/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

## **Install arctic-inference**

Before enabling Suffix Decoding speculative inference on Ascend, the Arctic Inference plugin must be installed. Arctic Inference is an open-source plugin launched by Snowflake specifically to optimize LLM inference speed. For detailed technical principles, please refer to the following article: [Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/). Install it within the container using the following command:

```bash
pip install arctic-inference
```

## **vLLM Instance Deployment**

Use the following command to start the container service instance. Speculative inference is enabled via the `--speculative-config` parameter, where `method` is set to `suffix`. For this test, `num_speculative_tokens` is uniformly set to `3`.

```bash
# set the NPU device number:
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1
# Enable the AIVector core to directly schedule ROCE communication.
export HCCL_OP_EXPANSION_MODE="AIV"
# Enable MLP prefetch for better performance.
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
# Enable FlashComm_v1 optimization when tensor parallel is enabled.
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /data/Qwen3-32B \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --speculative-config '{"method": "suffix", "num_speculative_tokens": 3}' \
  --gpu-memory-utilization 0.9 \
  --additional-config '{"pa_shape_list":[48,64,72,80]}' \
  --port 8011
```

## **AISbench Benchmark Testing**

Performance for all open-source datasets is tested using AISbench. For specific instructions, refer to [Using AISBench for performance evaluation](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_ais_bench.html#execute-performance-evaluation).

**Model Configuration**:

```bash
# "ignore_eos" must be set to "False", and "max_out_len" should be set to a large value to allow the model to output completely and naturally.

from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="<path_to_your_model>/Qwen3-32B",
        model="qwen3",
        request_rate = 0,
        retry = 2,
        host_ip = "<your_server_ip>",
        host_port = 8011,
        max_out_len = 4000,
        batch_size= 16,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            ignore_eos = False
        )
    )
]
```

**Performance Benchmarking Commands**:

```bash
# Example command to test gsm8k dataset performance using the first 100 prompts. Commands for other datasets are similar.
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf --num-prompts 100
```

## **Test Results**

Below are the detailed test results of the six open-source datasets in this evaluation. Compared to the baseline performance, the improvement in TPOT and throughput performance at different concurrency levels after enabling Suffix Decoding varies across datasets. The extent of improvement after enabling Suffix Decoding differs among the datasets. Below is a summary of the results:

| **Dataset Category** | **Typical Representative** | **Throughput Improvement (BS=1-10)** | **SLO TPOT** |
| -------------------- | -------------------------- | ------------------------------------ | ------------ |
| **High Gain**        | AGIEval, GSM8K             | **> 50%**                            | < 50ms       |
| **Medium-Low Gain**  | ARC, ShareGPT              | **20% ~ 30%**                        | < 50ms       |

Below is the raw detailed test results:

| Concurrency         | Avg Input | Avg Output | Requests | Base TPOT(ms) | Base Throughput(TPS) | Suffix TPOT(ms) | Suffix Throughput(TPS) | Accept Rate | TPOT Gain | TPS Gain |
| ------------------- | --------- | ---------- | -------- | ------------- | -------------------- | --------------- | ---------------------- | ----------- | --------- | -------- |
| **Humaneval**       |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 150       | 2700       | 100      | 55.1          | 18.1                 | 37.9            | 26.3                   | 27.0%       | 45.2%     | 45.1%    |
| 15                  | 150       | 2700       | 100      | 61.6          | 233.8                | 45.8            | 318.2                  | 27.0%       | 34.6%     | 36.1%    |
| 26                  | 150       | 2700       | 100      | 64.7          | 403.8                | 50.9            | 519.2                  | 27.0%       | 27.2%     | 28.6%    |
| **ARC**             |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 76        | 960        | 100      | 52.8          | 18.9                 | 39.5            | 25.4                   | 23.9%       | 33.7%     | 34.6%    |
| 8                   | 76        | 960        | 100      | 59.1          | 125.4                | 47.0            | 163.1                  | 23.9%       | 25.7%     | 30.0%    |
| 15                  | 76        | 960        | 100      | 59.8          | 245.8                | 48.9            | 311.7                  | 23.9%       | 22.3%     | 26.8%    |
| **GSM8K**           |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 67        | 1570       | 100      | 55.5          | 18.0                 | 35.7            | 28.5                   | 31.1%       | 55.6%     | 58.4%    |
| 17                  | 67        | 1570       | 100      | 61.5          | 279.8                | 45.4            | 403.0                  | 31.1%       | 35.6%     | 44.0%    |
| 26                  | 67        | 1570       | 100      | 63.9          | 396.4                | 50.0            | 527.6                  | 31.1%       | 27.8%     | 33.1%    |
| **ShareGPT**        |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 666       | 231        | 327      | 54.1          | 18.3                 | 39.2            | 24.1                   | 23.9%       | 37.9%     | 31.5%    |
| 8                   | 666       | 231        | 327      | 58.8          | 125.0                | 46.2            | 153.2                  | 23.9%       | 27.1%     | 22.5%    |
| 14                  | 666       | 231        | 327      | 61.8          | 227.0                | 49.9            | 273.9                  | 23.9%       | 23.8%     | 20.7%    |
| **SuperGLUE_BoolQ** |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 207       | 314        | 100      | 54.1          | 18.4                 | 36.1            | 26.8                   | 33.4%       | 49.8%     | 45.6%    |
| 16                  | 207       | 314        | 100      | 60.0          | 229.7                | 43.5            | 303.9                  | 33.4%       | 38.0%     | 32.3%    |
| 32                  | 207       | 314        | 100      | 62.7          | 396.4                | 47.8            | 507.5                  | 33.4%       | 31.3%     | 28.0%    |
| **Agieval**         |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 735       | 1880       | 100      | 53.1          | 18.7                 | 31.8            | 34.1                   | 50.3%       | 66.8%     | 81.9%    |
| 24                  | 735       | 1880       | 100      | 64.0          | 381.2                | 43.3            | 629.0                  | 50.3%       | 47.8%     | 65.0%    |
| 34                  | 735       | 1880       | 100      | 70.0          | 494.6                | 50.2            | 768.4                  | 50.3%       | 39.4%     | 55.3%    |
