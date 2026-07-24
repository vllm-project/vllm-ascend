# Qwen-VL-Dense(Qwen3-VL-2B/4B/8B/32B)

## 1 Introduction

The Qwen-VL(Vision-Language)series from Alibaba Cloud comprises a family of powerful Large Vision-Language Models (LVLMs) designed for comprehensive multimodal understanding. They accept images, text, and bounding boxes as input, and output text and detection boxes, enabling advanced functions like image detection, multi-modal dialogue, and multi-image reasoning.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, NPU deployment, accuracy and performance evaluation.

This tutorial uses the vLLM-Ascend `v0.11.0rc3-a3` version for demonstration, showcasing the `Qwen3-VL-8B-Instruct` model as an example for single NPU and multi-NPU deployment.

!!! note

    For **Atlas inference products**, Qwen3-VL Dense requires vLLM-Ascend `v0.18.0` or later. Do not use the demonstration version above on this hardware.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

Requires 1 card on Atlas 800I A2 (64G × 8), Atlas 800 A3 (64G × 16), or Atlas inference products:

- `Qwen3-VL-2B-Instruct`: [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)
- `Qwen3-VL-4B-Instruct`: [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)
- `Qwen3-VL-8B-Instruct`: [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)

Requires 2 cards on Atlas 800I A2 (64G × 8), Atlas 800 A3 (64G × 16), or Atlas inference products:

- `Qwen3-VL-32B-Instruct`: [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A2 / A3 series"

    ```bash
    # Update the vllm-ascend image
    # A2: quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    # A3: quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --device /dev/davinci0 \
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

=== "Atlas inference products"

    ```bash
    # Use the vllm-ascend image
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p

    docker run --rm \
    --name vllm-ascend \
    --shm-size=10g \
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
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash
    ```

**Installation Verification:**

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend
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

!!! note

    Atlas inference products do not support `triton` or `triton-ascend`. Source installation may pull them in automatically; uninstall them manually before running:

    ```bash
    pip uninstall -y triton-ascend triton
    ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

!!! note

    If deploying a multi-node environment, set up the environment on each node.

For more details, please refer to the [Installation Guide](../../installation.md).

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Run docker container to start the vLLM server on single-NPU:

=== "A2 / A3 series"

    ```bash
    vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --max_model_len 16384 \
    --max-num-batched-tokens 16384
    ```

=== "Atlas inference products"

    ```bash
    vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype float16 \
    --max_model_len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32]}' \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'
    ```

    !!! note

        On Atlas inference products:

        - Only `float16` dtype is supported.
        - Graph compilation (`--compilation-config`) requires **CANN version >= 9.0.0**. If your CANN version is lower, replace `--compilation-config` with `--enforce-eager`.
        - `--additional-config` with `"ascend_compilation_config": {"enable_npugraph_ex": false}` is required because `enable_npugraph_ex` is not supported on Atlas inference products.

Key Parameter Descriptions:

- Add `--max_model_len` option to avoid ValueError that the Qwen3-VL-8B-Instruct model's max seq len (256000) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series based on the on-chip memory size. Please modify the value according to a suitable value for your NPU series.

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustration?"}
    ]}
    ]
    }'
```

Expected Result:

The service returns HTTP 200 OK.

```bash
{"id":"chatcmpl-d3270d4a16cb4b98936f71ee3016451f","object":"chat.completion","created":1764924127,"model":"Qwen/Qwen3-VL-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is: **TONGYI Qwen**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":107,"total_tokens":123,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

## 7 Accuracy Evaluation

The accuracy of some models is already within our CI monitoring scope, including:

- `Qwen3-VL-8B-Instruct`

=== "A2 / A3 series"

    **Using Language Model Evaluation Harness**

    As an example, take the `mmmu_val` dataset as a test dataset, and run accuracy evaluation of `Qwen3-VL-8B-Instruct` in offline mode.

    1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for more details on `lm_eval` installation.

        ```shell
        pip install lm_eval
        ```

    2. Run `lm_eval` to execute the accuracy evaluation.

        ```shell
        lm_eval \
            --model vllm-vlm \
            --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_model_len=8192,gpu_memory_utilization=0.7 \
            --tasks mmmu_val \
            --batch_size 32 \
            --apply_chat_template \
            --trust_remote_code \
            --output_path ./results
        ```

    3. After execution, you can get the result, here is the result of `Qwen3-VL-8B-Instruct` in `vllm-ascend:0.11.0rc3` for reference only.

    | Tasks    | Value  | Stderr |
    | -------- | ------ | ------ |
    | mmmu_val | 0.5389 | 0.0159 |

=== "Atlas inference products"

    **Using AISBench**

    Take the `text_vqa` dataset as an example, and run accuracy evaluation of `Qwen3-VL-8B-Instruct`.

    1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for installation, dataset download, and configuration details.

    2. Run `ais_bench` to execute the accuracy evaluation.

        ```shell
        ais_bench --models vllm_api_general_chat --datasets textvqa_gen_base64 --mode all --debug
        ```

    3. After execution, you can get the result, here is the result of `Qwen3-VL-8B-Instruct` in `vllm-ascend:0.23.0rc1` for reference only.

    | dataset  | metric   | mode | vllm-api-general-chat |
    | -------- | -------- | ---- | --------------------- |
    | text_vqa | accuracy | gen  | 80.57                 |

## 8 Performance Evaluation

### Using vLLM Benchmark

Refer to [vLLM Benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

The performance evaluation must be conducted in an online mode. Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model Qwen/Qwen3-VL-8B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|High Throughput<br>(16K context)|Single-Node Mixed|1 (A3)|Qwen3-VL-8B-Instruct|Use tp2 for high-resolution text inputs|
|Long Context<br>(128K, no prefix cache)|Single-Node Mixed|1 (A3)|Qwen3-VL-8B-Instruct|tp2 for high-resolution text inputs|
|Long Context<br>(128K, with prefix cache)|Single-Node Mixed|1 (A3)|Qwen3-VL-8B-Instruct|tp2 for high-resolution text inputs|
|Multimodal<br>(1080P)|Single-Node Mixed|1 (A3)|Qwen3-VL-8B-Instruct|tp2 for high-resolution visual inputs|

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64G × 16 NPUs).

#### Table 2: Detailed Node Configuration

|Scenario|Configuration|NPUs|TP|DP|Max Model Len|MTP Speculation Num|Weight Version|
|--------|-------------|-----|--|--|-------------------|--------------------|---|
|High Throughput / Low Latency (16K)|Server / Single Machine|1|1|1|~16K|3|Qwen3-VL-8B-Instruct|
|Long Context (128K, no cache)|Server / Single Machine|1|1|1|128K|3|Qwen3-VL-8B-Instruct|
|Long Context (128K, with cache)|Server / Single Machine|1|1|1|128K|3|Qwen3-VL-8B-Instruct|
|Multimodal (1080P)|Server / Single Machine|1|1|1|~16K|3|Qwen3-VL-8B-Instruct|

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-online-service-deployment)** chapter.

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
