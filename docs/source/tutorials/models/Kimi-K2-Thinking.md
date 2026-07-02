# Kimi-K2-Thinking

## 1 Introduction

Kimi-K2-Thinking is a large-scale Mixture-of-Experts (MoE) model developed by Moonshot AI. It features a hybrid thinking architecture that excels in complex reasoning and problem-solving tasks.

This document will demonstrate the main verification steps of the model, including supported features, environment preparation, installation, online service deployment, functional verification, accuracy evaluation, performance evaluation, performance tuning, and FAQ.

This document is recommended to use the latest release candidate or official version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `Kimi-K2-Thinking` (bfloat16): requires 1 Atlas 800 A3 (64G x 16) node. [Download model weight](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

It is recommended to download the model weight to the shared directory, such as `/mnt/sfs_turbo/.cache/`.

After downloading the model weights, please edit the value of `"quantization_config.config_groups.group_0.targets"` from `["Linear"]` to `["MoE"]` in `config.json` of the original model to verify the quantized model.

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

```{code-block} bash
   :substitutions:
   # Update the vllm-ascend image according to your environment.
   export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3

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
docker ps --filter name=vllm-ascend --format "table {{.Names}}\t{{.Status}}"
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

```{code-block} bash
   :substitutions:
# Install vLLM.
git clone --depth 1 --branch |vllm_version| https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -e .
cd ..

# Install vLLM Ascend.
git clone --depth 1 --branch |vllm_ascend_version| https://github.com/vllm-project/vllm-ascend.git
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

If you want to deploy a multi-node environment, set up the same software environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for online inference scenarios with moderate concurrency requirements.

For an Atlas 800 A3 (64G x 16) node, `tensor-parallel-size` should be at least 16.

Run the following script to start the vLLM server:

```{model-code}
:block_name: kimi_k2_thinking_single_node
:converter_tag: single_node
:test_case_path: tests/e2e/nightly/single_node/models/configs/Kimi-K2-Thinking.yaml
```

**Parameter and Environment Variable Descriptions:**

- `HCCL_BUFFSIZE=1024`: configures the HCCL buffer size.
- `TASK_QUEUE_ENABLE=1`: enables task queue scheduling.
- `OMP_PROC_BIND=false`: avoids overly strict OpenMP CPU binding.
- `HCCL_OP_EXPANSION_MODE=AIV`: enables the AIV communication path.
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`: reduces NPU memory fragmentation.
- `SERVER_PORT`: sets the service port. The generated script maps `DEFAULT_PORT` to `8000`.
- `--tensor-parallel-size 16`: uses 16-way tensor parallelism on the A3 node.
- `--max-model-len 8192`: sets the maximum model context length.
- `--max-num-batched-tokens 8192`: sets the maximum number of batched tokens.
- `--max-num-seqs 12`: sets the maximum number of concurrent sequences.
- `--gpu-memory-utilization 0.9`: controls the memory ratio used by vLLM.
- `--trust-remote-code`: allows loading model-specific remote code.
- `--enable-expert-parallel`: enables expert parallelism for MoE layers.
- `--no-enable-prefix-caching`: disables prefix caching for a stable baseline.

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

Below are reference `gsm8k` results for `Kimi-K2-Thinking` powered by `vllm-ascend:v0.20.2rc1`, evaluated on one Atlas 800 A3 node (64G × 16).

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

The following reference results are obtained with `vllm-ascend:v0.20.2rc1` on one Atlas 800 A3 node (64G × 16), using OpenAI chat serving, random input/output lengths, 10 prompts, and `--request-rate 1`:

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
| Long Context | Single-node | 16 (A3) | bfloat16 | Keep `--max-model-len` close to the real maximum input and output length; reduce `--max-num-seqs` first when memory pressure is high. |
| Low Latency | Single-node | 16 (A3) | bfloat16 | Reduce `--max-num-seqs` and `--max-num-batched-tokens` to reduce queueing delay. |
| High Throughput | Single-node | 16 (A3) | bfloat16 | Increase `--max-num-seqs` gradually and benchmark with a request rate close to the real workload. |

#### Table 2: Detailed Recommendations

- **Long context:** use `tp16`, keep `--max-model-len` close to the real maximum input and output length, and reduce `--max-num-seqs` first when memory pressure is high.
- **Low latency:** reduce `--max-num-seqs` and `--max-num-batched-tokens` to reduce queueing delay.
- **High throughput:** increase `--max-num-seqs` gradually and benchmark with a request rate close to the real workload. For long-context throughput tests, evaluate `--decode-context-parallel-size` as an optional tuning knob.
- For 1024 input tokens and 1024 output tokens in the reference concurrency sweep, `--max-concurrency 8` had the best output throughput. Higher concurrency increased TTFT significantly, so validate tail latency before using it in production.

> **Note:**
>
> - `--max-model-len` and `--max-num-seqs` need to be set according to the actual usage scenario.
> - If the service runs under high concurrency, verify NPU health and HCCL status before increasing request rate.

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

> For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html); this chapter only covers model-specific issues.

- **Q: API returns `{"error":"Model not found"}` or `404` when requesting with `model: "Kimi-K2-Thinking"`?**

  A: The server registers the model under its full path `moonshotai/Kimi-K2-Thinking` by default. When the request uses the short name `Kimi-K2-Thinking` without `--served-model-name` override, the server cannot resolve the model ID. Use `"model": "moonshotai/Kimi-K2-Thinking"` in requests, or start the server with `--served-model-name Kimi-K2-Thinking` to enable the short name.
