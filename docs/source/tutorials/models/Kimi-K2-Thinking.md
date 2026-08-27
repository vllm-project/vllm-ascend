# Kimi-K2-Thinking

## 1 Introduction

Kimi-K2-Thinking is a large-scale Mixture-of-Experts (MoE) model developed by Moonshot AI. It features a hybrid thinking architecture that excels in complex reasoning and problem-solving tasks.

This document will demonstrate the main verification steps and references of the model, including supported features, environment preparation, installation, online service deployment, functional verification, accuracy evaluation, performance evaluation, performance tuning, and FAQ.

This document is validated and written based on **vLLM-Ascend v0.9.0rc1**. The current model (Kimi-K2-Thinking) is first supported in this version, and **v0.9.0rc1 and later versions** can run stably. It is recommended to use the latest release candidate or stable version alongside this document.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `Kimi-K2-Thinking` (bfloat16): requires 1 Atlas 800 A3 (64GB × 16) node. [Download model weight](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

It is recommended to download the model weight to the shared directory, such as `/mnt/sfs_turbo/.cache/`.

After downloading the model weights, please edit the value of `"quantization_config.config_groups.group_0.targets"` from `["Linear"]` to `["MoE"]` in `config.json` of the original model to use the quantized model.

```json
{
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "targets": [
          "MoE"
        ]
      }
    }
  }
}
```

Your model files should look like:

```bash
.
|-- chat_template.jinja
|-- config.json
|-- configuration_deepseek.py
|-- configuration.json
|-- generation_config.json
|-- model-00001-of-000062.safetensors
|-- ...
|-- model-00062-of-000062.safetensors
|-- model.safetensors.index.json
|-- modeling_deepseek.py
|-- tiktoken.model
|-- tokenization_kimi.py
|-- tokenizer_config.json
```

## 4 Installation

### 4.1 Docker Image Installation

You can use the official Docker image to run `Kimi-K2-Thinking` directly.

Select an image based on your machine type and start the Docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

```bash
   # Update the vllm-ascend image according to your environment.
   export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
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
-v /mnt/sfs_turbo/.cache:/home/cache \
-it $IMAGE bash
```

**Parameter Descriptions:**

- `IMAGE`: specifies the `vllm-ascend` image. The `-a3` suffix selects the Atlas A3 image.
- `NAME`: specifies the container name.
- `--net=host`: uses host networking, so the vLLM service port is exposed on the host directly.
- `--shm-size=1g`: configures container shared memory.
- `--device /dev/davinci[0-15]`: exposes 16 Ascend NPU devices to the container.
- `--device /dev/davinci_manager`, `--device /dev/devmm_svm`, and `--device /dev/hisi_hdc`: expose required Ascend runtime device files.
- `-v /usr/local/dcmi:/usr/local/dcmi`: mounts DCMI tools for device management.
- `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi`: mounts the NPU monitoring command.
- `-v /usr/local/Ascend/driver/*`: mounts Ascend driver libraries and version files.
- `-v /etc/ascend_install.info:/etc/ascend_install.info`: mounts Ascend installation metadata.
- `-v /mnt/sfs_turbo/.cache:/home/cache`: mounts the shared model cache directory. Update it if you store model weights elsewhere.

After the container starts, run the following command on the host to verify the container status:

```bash
docker ps --filter name=vllm-ascend --format "table {% raw %}{{.Names}}\t{{.Status}}{% endraw %}"
```

Expected Status:

- The container name is `vllm-ascend`.
- The status is `Up ...`.
- The container does not exit immediately.

Run the following command in the container to verify that Ascend devices are visible:

```bash
npu-smi info
```

Expected Status:

- The command exits successfully.
- The output lists the expected NPU devices.
- Device health status is normal.

### 4.2 Source Code Installation

If you do not want to use the Docker image, you can also build from source:

```bash
# Install vLLM.
git clone --depth 1 --branch {{ vllm_version }} https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -e .
cd ..

# Install vLLM Ascend.
git clone --depth 1 --branch {{ vllm_ascend_version }} https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

To verify the source installation, run:

```bash
python -c "import vllm; import vllm_ascend; print('vllm and vllm_ascend import ok')"
```

Expected Status:

- The command exits successfully.
- `vllm and vllm_ascend import ok` is printed.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for online inference scenarios with moderate concurrency requirements.

For an Atlas 800 A3 (64GB × 16) node, `tensor-parallel-size` should be at least 16.

Run the following script to start the vLLM server:

```{model-code}
:block_name: kimi_k2_thinking_single_node
:converter_tag: single_node
:test_case_path: tests/e2e/nightly/single_node/models/configs/Kimi-K2-Thinking.yaml
```

**Parameter and Environment Variable Descriptions:**

The following table covers the generated `model`, all `envs`, and all `server_cmd` entries. Parameters are categorized by review priority: version-sensitive parameters, performance parameters, and Kimi-K2-Thinking-specific parameters.

| Parameter | Validated Value | Category | Description and Tuning Guidance |
| --- | --- | --- | --- |
| `moonshotai/Kimi-K2-Thinking` | model path | Model-specific | Specifies the model weight path passed to `vllm serve`. Because `--served-model-name` is not set in the script, API requests must use `moonshotai/Kimi-K2-Thinking` as the model name unless you add an explicit served-model-name override; see the FAQ in Chapter 10. |
| `HCCL_BUFFSIZE` | `1024` | Performance | Configures the HCCL communication buffer used by distributed NPU communication. This document validates `1024`; other values need separate throughput, TTFT, TPOT, and HCCL stability validation. |
| `TASK_QUEUE_ENABLE` | `1` | Version-sensitive / Performance | Enables task queue scheduling on Ascend. This document validates `1`; other values or version changes need startup and first-request validation. |
| `OMP_PROC_BIND` | `false` | Performance | Avoids overly strict OpenMP CPU binding. This document validates `false`; other values need separate CPU affinity, NPU health, and HCCL stability validation. |
| `HCCL_OP_EXPANSION_MODE` | `AIV` | Performance | Enables the AIV communication path. This document validates `AIV`; other values need separate throughput and latency validation. |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | Memory / Performance | Reduces NPU memory fragmentation. This document validates `expandable_segments:True`; other allocator settings need separate startup, memory, and runtime stability validation. |
| `SERVER_PORT` and `--port` | `8000` | Service | Sets the OpenAI-compatible service port. The documentation generator maps `DEFAULT_PORT` in the YAML to `8000`; update the curl examples if you change this value. |
| `--tensor-parallel-size` | `16` | Model-specific / Performance | Uses all 16 NPUs on one Atlas 800 A3 node. This document validates `tp16`; other topologies need separate memory, accuracy, and communication validation. |
| `--max-model-len` | `8192` | Performance | Sets the maximum input plus output tokens for one request and determines KV cache reservation. This document validates `8192`; larger values need separate NPU memory, accuracy, and performance validation. Keep it close to the real maximum input and output length for your workload. |
| `--max-num-batched-tokens` | `8192` | Performance | Limits tokens processed in one scheduler step. This document validates `8192`; other values need separate memory, TTFT, TPOT, and throughput validation. |
| `--max-num-seqs` | `12` | Performance | Limits active sequences scheduled at the same time. This document validates `12`; higher values need separate tail-latency and throughput validation. The reference sweep in Chapter 8 shows concurrency 16 causes severe TTFT growth; validate tail latency before raising it in production. |
| `--gpu-memory-utilization` | `0.9` | Memory / Performance | Controls the fraction of NPU HBM used by vLLM for KV cache planning. This document validates `0.9`; other values need separate startup, OOM, and runtime stability validation. |
| `--trust-remote-code` | enabled | Model-specific | Required because the model package contains model-specific configuration, modeling, tokenizer, and chat-template files. Disable it only after replacing the remote-code dependency with a validated native implementation. |
| `--enable-expert-parallel` | enabled | Model-specific / Performance | Enables expert parallelism for Kimi-K2-Thinking MoE layers so experts can be distributed across NPUs. This document validates it as enabled; disabling it is not validated in this tutorial. |
| `--no-enable-prefix-caching` | enabled | Performance | Disables prefix caching for the validated baseline and random-prompt benchmarks. Prefix caching is not validated in this tutorial. |

**Common Issues Tip:** For common environment, installation, and general parameter issues during deployment, please refer to the [Public FAQs](../../faqs.md). If the service runs under high concurrency, verify NPU health and HCCL status before increasing the request rate.

**Service Verification:**

After the service starts, you should see logs similar to:

```bash
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Expected Status:

- The server process starts successfully.
- No error logs related to HCCL or NPU initialization.
- The container does not exit immediately.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "moonshotai/Kimi-K2-Thinking",
  "messages": [
    {"role": "user", "content": "Who are you?"}
  ],
  "temperature": 1.0
}'
```

Expected Result:

- The HTTP status code is `200`.
- `choices[0].message.content` contains the generated assistant response.

## 7 Accuracy Evaluation

### Using AISBench

For details, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

### Using lm-eval

You can use [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate the model accuracy through the OpenAI-compatible API.

For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).

Run `lm_eval` to execute the accuracy evaluation:

```shell
lm_eval \
  --model local-completions \
  --model_args model=moonshotai/Kimi-K2-Thinking,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

Reference configuration: `gsm8k` (5-shot), `--apply_chat_template`, `--fewshot_as_multiturn`, greedy decoding (`temperature=0.0`, `top_p=1.0`), max 2048 output tokens, batch size 1.

Below are reference `gsm8k` results for `Kimi-K2-Thinking` powered by `vllm-ascend:v0.20.2rc1`, evaluated on one Atlas 800 A3 node (64GB × 16).

| task | version | filter | n-shot | metric | value | stderr |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `gsm8k` | 3 | `flexible-extract` | 5 | `exact_match` | 0.8992 | 0.0083 |
| `gsm8k` | 3 | `strict-match` | 5 | `exact_match` | 0.8453 | 0.0100 |

## 8 Performance Evaluation

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

**Test Command Example:**

```bash
vllm bench serve \
  --backend openai-chat \
  --model moonshotai/Kimi-K2-Thinking \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --num-prompts 10 \
  --request-rate 1
```

After the benchmark completes, you can get the performance result, including request throughput, output token throughput, TTFT, TPOT, and ITL.

The following reference results are obtained with `vllm-ascend:v0.20.2rc1` on one Atlas 800 A3 node (64GB × 16), using OpenAI chat serving, random input/output lengths, 10 prompts, and `--request-rate 1`:

| random input len | random output len | success | duration (s) | request throughput (req/s) | output throughput (tok/s) | total throughput (tok/s) | mean TTFT (ms) | mean TPOT (ms) | mean ITL (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 10 / 10 | 111.00 | 0.09 | 46.12 | 94.38 | 507.60 | 200.47 | 200.08 |
| 1024 | 1024 | 10 / 10 | 221.52 | 0.05 | 46.23 | 93.48 | 566.39 | 208.20 | 208.00 |
| 2048 | 2048 | 10 / 10 | 479.72 | 0.02 | 42.69 | 85.78 | 722.32 | 230.26 | 230.15 |

For a concurrency sweep, keep the input and output length fixed and vary `--max-concurrency`:

```bash
MODEL_NAME=moonshotai/Kimi-K2-Thinking
INPUT_LEN=1024
OUTPUT_LEN=1024

for CONCURRENCY in 1 2 4 8 16 32; do
  NUM_PROMPTS=$((CONCURRENCY * 10))
  vllm bench serve \
    --backend openai-chat \
    --model "$MODEL_NAME" \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate inf \
    --max-concurrency "$CONCURRENCY"
done
```

Reference results for 1024 input tokens and 1024 output tokens are:

| max concurrency | prompts | success | duration (s) | request throughput (req/s) | output throughput (tok/s) | total throughput (tok/s) | mean TTFT (ms) | P99 TTFT (ms) | mean TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 10 / 10 | 595.07 | 0.02 | 17.21 | 34.80 | 473.71 | 712.49 | 57.71 |
| 2 | 20 | 20 / 20 | 623.88 | 0.03 | 32.83 | 66.35 | 708.16 | 996.59 | 60.29 |
| 4 | 40 | 40 / 40 | 725.38 | 0.06 | 56.47 | 114.13 | 956.11 | 1137.55 | 69.97 |
| 8 | 80 | 80 / 80 | 907.44 | 0.09 | 90.28 | 182.43 | 1361.85 | 1900.15 | 87.37 |
| 16 | 160 | 160 / 160 | 3093.07 | 0.05 | 52.97 | 107.04 | 76766.84 | 251245.22 | 222.07 |

> **Note:** At concurrency levels of 16, the Mean TTFT increases significantly (76.7s), indicating severe queueing delay. For production deployment, it is recommended to limit concurrency based on your latency requirements or increase `--max-num-seqs` and `--max-num-batched-tokens` if NPU memory allows.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note:** The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | Total NPUs | Weight Version | Key Considerations |
|----------|----------------|------------|----------------|---------------------|
| Long Context | Single-node | 16 (A3) | bfloat16 | Keep `--max-model-len` close to the real maximum input and output length, and reduce `--max-num-seqs` first when memory pressure is high. The validated scope of this single-node baseline covers up to 2K input / 2K output in Chapter 8. |
| Low Latency | Single-node | 16 (A3) | bfloat16 | Reduce `--max-num-seqs` and `--max-num-batched-tokens` from the validated baseline (`12` and `8192`) to reduce queueing delay. In the Chapter 8 concurrency sweep, concurrency 1-4 kept mean TTFT below 1s; validate TTFT, TPOT, and tail latency against the target latency SLO. |
| High Throughput | Single-node | 16 (A3) | bfloat16 | Increase `--max-num-seqs` gradually and benchmark with a request rate close to the real workload. In the 1K/1K concurrency sweep in Chapter 8, concurrency 8 gave the best output throughput; validate tail latency before using higher concurrency in production. |

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

> For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

- **Q: API returns `{"error":"Model not found"}` or `404` when requesting with `model: "Kimi-K2-Thinking"`?**

  A: The server registers the model under its full path `moonshotai/Kimi-K2-Thinking` by default. When the request uses the short name `Kimi-K2-Thinking` without `--served-model-name` override, the server cannot resolve the model ID. Use `"model": "moonshotai/Kimi-K2-Thinking"` in requests, or start the server with `--served-model-name Kimi-K2-Thinking` to enable the short name.
