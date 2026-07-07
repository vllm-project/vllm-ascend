# Qwen3-Omni-30B-A3B-Thinking

## 1 Introduction

Qwen3-Omni is a native end-to-end multilingual omni-modal foundation model. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. We introduce several architectural upgrades to improve performance and efficiency. The Thinking model of Qwen3-Omni-30B-A3B, which contains the thinker component, is equipped with chain-of-thought reasoning and supports audio, video, and text input, with text output.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

The Qwen3-Omni-30B-A3B model is first supported in v0.12.0rc1. This document is validated and written based on vLLM-Ascend v0.22.1rc. All v0.22.1rc and later versions can run stably. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Please refer to [Supported Features List](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/support_matrix/supported_models/) to get the model's supported feature matrix.

Please refer to [Feature Guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/index.html) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory accessible to all nodes.

| Model                | Hardware Requirement                                                                             | Download                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Qwen3-Omni-30B-A3B (BF16) | Atlas 800I A3 (64G, 1\~2 cards)<br>Atlas 800I A2 (64G, 2\~4 cards) | [Download](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B)          |
| Qwen3-Omni-30B-A3B-W8A8   | Atlas 800I A3 (64G, 1\~2 cards)<br>Atlas 800I A2 (64G, 2\~4 cards)                               |  N/A|

The W8A8 quantized weights are not available for direct download, you can obtain them by quantizing the BF16 model using **msmodelslim**. Refer to the [Quantization Guide](../../user_guide/feature_guide/quantization.md) for details. All model paths in this document should be adjusted to your actual local paths.

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

!!! note

    Qwen3-Omni-30B-A3B-W8A8 adopts a hybrid quantization strategy (ordered by model structure):

    - **Embedding layer**: BF16 (no quantization)
    - **Q/K normalization** (q_norm, k_norm): BF16
    - **Attention projections** (q/k/v/o_proj): Static W8A8 with pre-computed per-tensor scales
    - **MoE routing gate** (mlp.gate): BF16
    - **MoE expert projections** (gate/up/down_proj): Dynamic W8A8 where input scales are computed on-the-fly during inference

It is recommended to download the model weight to a shared directory across multiple nodes.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3-Omni MoE models.

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

**Docker Run:**

=== "Atlas 800I A3"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run \
        --name vllm-ascend-env \
        --shm-size=128g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci8 \
        --device /dev/davinci9 \
        --device /dev/davinci10 \
        --device /dev/davinci11 \
        --device /dev/davinci12 \
        --device /dev/davinci13 \
        --device /dev/davinci14 \
        --device /dev/davinci15 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -it -d $IMAGE bash
    ```

    !!! note

        A3 has 8 NPUs with dual-die design (16 chips total: `/dev/davinci[0-15]`).
        If you are on a shared machine, map only the chips you need (e.g., `/dev/davinci[0-7]` for NPU 0-3).

=== "Atlas 800I A2"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run \
        --name vllm-ascend-env \
        --shm-size=128g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -it -d $IMAGE bash
    ```

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

**Installation Verification:**

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source. Install vLLM from source first:

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install the vLLM-Ascend repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

!!! note

    If deploying a multi-node environment, set up the environment on each node.

Please install system dependencies.

```bash
pip install qwen_omni_utils modelscope
# Used for audio processing.
apt-get update && apt-get install -y ffmpeg
# Check the installation.
ffmpeg -version
```

Required to avoid HcclAllreduce failures caused by the default FFTS+ mode's stream and shape limitations.

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
```

## 5 Online Service Deployment

PS:Because the model has fewer parameters, it doesn’t involve the PD separation scenario.

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios. For the Qwen3-Omni-30B-A3B MoE model, Expert Parallelism (EP) is required to distribute experts across NPUs.

> The following command is an example configuration. Adjust the parameters based on your actual scenario.

**Atlas 800I A2/A3:**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"  # not needed on A2
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-omni \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --quantization ascend \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --additional-config '{"enable_flashcomm1": true, "weight_nz_mode": 2}' \
    --port 8000 
```

!!! note

    - `ASCEND_RT_VISIBLE_DEVICES`: must be set to the NPU chip IDs allocated to your environment (e.g., `0,1,2,3` for 4 chips).
    - `--port`: adjust to avoid conflicts with other services running on the same machine.
    - `--no-enable-prefix-caching`: disabled by default as prefix caching effectiveness for this model on Ascend NPUs has not been fully characterized. You can try enabling it to evaluate the cache hit rate for your workload.
    - `--quantization ascend`: required for W8A8 quantized models. Remove this parameter when using BF16 weights.

!!! tip

    For parameter details, refer to:

    - [vLLM CLI documentation](https://docs.vllm.ai/en/stable/cli/) — standard serve parameters (`--host`, `--port`, `--max-model-len`, etc.)
    - [Environment Variables](../../user_guide/configuration/env_vars.md) — Ascend-specific environment variables (`HCCL_*`, etc.)
    - [Additional Configuration](../../user_guide/configuration/additional_config.md) — `--additional-config` format and options

**Service Verification:**

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

## 6 Functional Verification

Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8000/v1/chat/completions \
-X POST \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen3-omni",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
                    }
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
                    }
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
                    }

                },
                {
                    "type": "text",
                    "text":  "Analyze this audio, image, and video together."
                }
            ]
        }
    ]
}'
```

The output should resemble the following.

```bash
{
  "id": "chatcmpl-8f3a1c2e4b5d4f7a9c0e1d2b3a4f5e6d",
  "object": "chat.completion",
  "created": 1782714240,
  "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "Let me examine each modality. The image shows several cars parked in a row — looks like a street or lot, daytime. The audio is short and sharp, sounds like a human cough (a single dry cough). The video appears to show someone drawing/sketching on paper, a hand moving a pen. Now I need to connect them: there isn't a single obvious narrative linking cars + cough + drawing, so I should describe each and then note any plausible shared context (e.g., a person sketching cars while having a cold). I'll keep the synthesis honest rather than forcing a connection.",
        "content": "Here's an analysis of the three inputs together:\n\n**Image** — A row of parked cars photographed outdoors in daylight, likely on a street or in a parking area. The vehicles are of varying colors and models, shot at a slight angle so several are visible in perspective.\n\n**Audio** — A brief, dry human cough: a single forceful exhalation lasting under a second, with no speech or background music. It sounds like an isolated cough rather than a sustained coughing fit.\n\n**Video** — A close-up of a hand drawing/sketching with a pen on paper, the strokes building up a simple illustration over a few seconds.\n\n**Combined interpretation** — The three clips don't share an explicit storyline; they're distinct samples of vision, sound, and motion. If a connecting context is assumed, one plausible scene is a person sketching cars (the drawing in the video, the cars in the image) while momentarily coughing (the audio) — e.g., an artist working outdoors who has a cold. But strictly, each input stands on its own: a static photo of cars, a one-off cough sound, and a short hand-drawing clip.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 8423,
    "total_tokens": 8712,
    "completion_tokens": 289,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null
}

```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using EvalScope

As an example, take the `gsm8k` `omnibench` `bbh` dataset as a test dataset, and run accuracy evaluation of `Qwen3-Omni-30B-A3B-Thinking` in online mode.

1. Refer to [Using evalscope](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_evalscope/#2-install-evalscope-using-pip) for `evalscope`installation.
2. Run `evalscope` to execute the accuracy evaluation.

    ```bash
    evalscope eval \
        --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-Omni-30B-A3B-Thinking \
        --api-url http://localhost:8000/v1 \
        --api-key EMPTY \
        --eval-type server \
        --datasets omni_bench, gsm8k, bbh \
        --dataset-args '{"omni_bench": { "extra_params": { "use_image": true, "use_audio": false}}}' \
        --eval-batch-size 1 \
        --generation-config '{"max_completion_tokens": 10000, "temperature": 0.6}' \
        --limit 100
    ```

3. After execution, you can get the result, here is the result of `Qwen3-Omni-30B-A3B-Thinking` in vllm-ascend:0.13.0rc1 for reference only.

    ```bash
    +-----------------------------+------------+----------+----------+-------+---------+---------+
    | Model                       | Dataset    | Metric   | Subset   |   Num |   Score | Cat.0   |
    +=============================+============+==========+==========+=======+=========+=========+
    | Qwen3-Omni-30B-A3B-Thinking | omni_bench | mean_acc | default  |   100 |    0.44 | default |
    +-----------------------------+------------+----------+----------+-------+---------+---------+ 
    | Qwen3-Omni-30B-A3B-Thinking | gsm8k      | mean_acc | main     |   100 |    0.98 | default |
    +-----------------------------+-----------+----------+----------+-------+---------+---------+
    | Qwen3-Omni-30B-A3B-Thinking | bbh        | mean_acc | OVERALL  |   270 |  0.9148 |         |
    +-----------------------------+------------+----------+----------+-------+---------+---------+
    ```

## 8 Performance Evaluation

### Using vLLM Benchmark  

Run performance evaluation of `Qwen3-Omni-30B-A3B-Thinking` as an example.
Refer to vllm benchmark for more details.
Refer to [vLLM Benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```bash
export VLLM_USE_MODELSCOPE=True
export MODEL=Qwen/Qwen3-Omni-30B-A3B-Thinking
python3 -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 2 --swap-space 16 --disable-log-stats --disable-log-request --load-format dummy

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install -r vllm-ascend/benchmarks/requirements-bench.txt

vllm bench serve --model $MODEL --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After execution, you can get the result, here is the result of `Qwen3-Omni-30B-A3B-Thinking` in vllm-ascend:0.13.0rc1 for reference only.

```bash
============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Request rate configured (RPS):           1.00
Benchmark duration (s):                  211.90
Total input tokens:                      40000
Total generated tokens:                  25600
Request throughput (req/s):              0.94
Output token throughput (tok/s):         120.81
Peak output token throughput (tok/s):    216.00
Peak concurrent requests:                24.00
Total token throughput (tok/s):          309.58
---------------Time to First Token----------------
Mean TTFT (ms):                          215.50
Median TTFT (ms):                        211.51
P99 TTFT (ms):                           317.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          98.96
Median TPOT (ms):                        99.19
P99 TPOT (ms):                           101.52
---------------Inter-token Latency----------------
Mean ITL (ms):                           99.02
Median ITL (ms):                         96.10
P99 ITL (ms):                            176.02
==================================================
```

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario        | Deployment Mode   | *Total NPUs      | Weight Version | Key Considerations                                               |
| --------------- | ----------------- | ---------------- | -------------- | ---------------------------------------------------------------- |
| High Throughput | Single-Node (TP1) | 1 (A3)<br>2 (A2) | W8A8           | Single-card deployment maximizes concurrent request processing   |
| Low Latency     | Single-Node (TP4) | 2 (A3)<br>4 (A2) | W8A8           | Multi-card TP reduces per-token latency with expert parallelism  |
| Long Context    | Single-Node (TP4) | 2 (A3)<br>4 (A2) | W8A8           | Reduces concurrent sequences to accommodate longer max-model-len |

> `*Total NPUs` indicates the total number of NPUs used across all nodes. On Atlas 800I A3, each NPU contains two dies (chips), so TP4 requires 4 chips = 2 NPUs.

#### Table 2: Detailed Node Configuration

| Scenario        | NPUs   | TP  | max-model-len | max-num-seqs | FUSED_MC2 | EP  | hf-overrides |
| --------------- | ------ | --- | ------------- | ------------ | --------- | --- | ------------ |
| High Throughput | 1 (A3) | 1   | 37364         | 100          | Off       | Off | -            |
| Low Latency     | 2 (A3) | 4   | 37364         | 100          | Off       | On  | -            |
| Long Context    | 2 (A3) | 4   | 131072        | 14           | Off       | On  | YaRN         |

> For detailed parameter descriptions, please refer to the deployment examples in Section 5.

**Low Latency Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-omni \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true, "weight_nz_mode": 2}' \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}'
```

!!! tip

    Example AISBench settings for this configuration:

    - `request_rate`: 0
    - `batch_size`: 32
    - Input/Output length: 2048/2048 or 3500/1500

**High Throughput Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-omni \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 1 \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"weight_nz_mode": 2}' \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}'
```

!!! tip

    Example AISBench settings for this configuration:

    - `request_rate`: 0
    - `batch_size`: 32
    - Input/Output length: 2048/2048 or 3500/1500

**Long Context Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-omni \
    --trust-remote-code \
    --max-num-seqs 14 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true, "weight_nz_mode": 2}' \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}}'
```

!!! tip

    Example AISBench settings for this configuration:

    - `request_rate`: 0
    - `batch_size`: 32
    - Input/Output length: 65536/1024 or 131072/1024

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs/).
