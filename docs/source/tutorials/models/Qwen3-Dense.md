# Qwen3-Dense (Qwen3-0.6B/1.7B/4B/8B/14B/32B)

## 1 Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support. The Dense variants covered in this document include Qwen3-0.6B, 1.7B, 4B, 8B, 14B, and 32B, along with their quantized versions (W8A8, W4A4, and W8A8SC-310) optimized for Ascend NPU deployment.

This document will demonstrate the main validation steps for Qwen3 Dense models in the vLLM-Ascend environment, including supported features, environment preparation, model quantization, single-node and multi-node deployment, as well as accuracy and performance evaluation. By tailoring service-level configurations to fit different use cases, you can ensure optimal performance across various scenarios.

The Qwen3 Dense models are first supported in v0.8.4rc2. W8A8 quantization was first supported in v0.8.4rc2, and W4A4 is supported since v0.11.0rc1. Atlas 300I DUO uses the W8A8SC-310 quantized weights listed in this tutorial. This document is validated and written based on **vLLM-Ascend v0.21.0**. All **v0.21.0 and later versions** can run stably. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory accessible to all nodes.

**BF16 Versions:**

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-0.6B | 1 Atlas A3 inference products (64GB × 16), 1 Atlas A2 inference products (64GB × 8) | [Modelers](https://modelers.cn/models/Modelers_Park/Qwen3-0.6B) |
| Qwen3-1.7B | 1 Atlas A3 inference products (64GB × 16), 1 Atlas A2 inference products (64GB × 8) | [Modelers](https://modelers.cn/models/Modelers_Park/Qwen3-1.7B) |
| Qwen3-4B | 1 Atlas A3 inference products (64GB × 16), 1 Atlas A2 inference products (64GB × 8) | [Modelers](https://modelers.cn/models/Modelers_Park/Qwen3-4B) |
| Qwen3-8B | 1 Atlas A3 inference products (64GB × 16), 1 Atlas A2 inference products (64GB × 8) | [Modelers](https://modelers.cn/models/Modelers_Park/Qwen3-8B) |
| Qwen3-14B | 1 Atlas A3 inference products (64GB × 16), 1 Atlas A2 inference products (64GB × 8) | [Modelers](https://modelers.cn/models/Modelers_Park/Qwen3-14B) |
| Qwen3-32B | 1 Atlas A3 inference products (64GB × 16), 1 Atlas A2 inference products (64GB × 8) | [Modelers](https://modelers.cn/models/Modelers_Park/Qwen3-32B) |

**Quantized Versions for Atlas A2/A3 inference products:**

| Model | Quantization | Hardware Requirement | Download |
|-------|-------------|---------------------|----------|
| Qwen3-32B-W4A4 | W4A4 | 1 Atlas A3 inference products (64GB × 16) or 1 Atlas A2 inference products (64GB × 8) | [ModelScope](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4) |
| Qwen3-32B-W8A8 | W8A8 | 1 Atlas A3 inference products (64GB × 16) or 1 Atlas A2 inference products (64GB × 8) | [ModelScope](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W8A8) |

**Quantized Versions for Atlas 300I DUO:**

| Model | Quantization | Hardware Requirement | Download |
|-------|-------------|---------------------|----------|
| Qwen3-8B-W8A8SC | W8A8SC | Atlas 300I DUO (TP1) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3-8B-w8a8sc-310-vllm) |
| Qwen3-14B-W8A8SC | W8A8SC | Atlas 300I DUO (TP1) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3-14B-w8a8sc-310-vllm) |
| Qwen3-32B-W8A8SC | W8A8SC | Atlas 300I DUO (TP4) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3-32B-w8a8sc-310-vllm) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 Dense models.

**Docker Pull:**

```bash

docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

**Docker Run:**

Start the docker image on each node.

=== "Atlas A3 inference products"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

    docker run --rm \
        --name vllm-ascend-env \
        --shm-size=1g \
        --net=host \
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
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

    !!! note

        Atlas A3 inference products have 8 NPUs with dual-die design (16 chips total: `/dev/davinci[0-15]`).
        If you are on a shared machine, map only the chips you need (e.g., `/dev/davinci[0-7]` for NPU 0-3).

=== "Atlas A2 inference products"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm \
        --name vllm-ascend-env \
        --shm-size=1g \
        --net=host \
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
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "Atlas 300I DUO"

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
    -p 8080:8080 \
    -it $IMAGE bash
    ```

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

Installation Verification:
After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status Up. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../getting_started/installation.md).

!!! note

    For Atlas 300I DUO, source installation may pull in `triton` and `triton-ascend`. Uninstall them before running vLLM-Ascend on Atlas 300I DUO:

    ```bash
    pip uninstall -y triton-ascend triton
    ```

To verify the source installation:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, confirming a successful installation.

!!! note

    If deploying a multi-node environment, set up the environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios.

**Start the server:**
> The following command is an example configuration. Adjust the parameters based on your actual scenario.

=== "Atlas A2 inference products / Atlas A3 inference products"

    Qwen3-32B-W8A8:

    ```bash
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    # Ensure the model path matches the directory recorded during download
    vllm serve your_model_path \
        --served-model-name qwen3 \
        --trust-remote-code \
        --quantization ascend \
        --distributed-executor-backend "mp" \
        --tensor-parallel-size 4 \
        --max-model-len 5500 \
        --max-num-batched-tokens 40960 \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --port <port> \
        --gpu-memory-utilization 0.9
    ```

    Qwen3-32B-W4A4:

    ```bash
    export HCCL_BUFFSIZE=1024
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    # Ensure the model path matches the directory recorded during download
    vllm serve your_model_path \
        --port 8004 \
        --data-parallel-size 1 \
        --tensor-parallel-size 2 \
        --served-model-name qwen3 \
        --distributed-executor-backend "mp" \
        --max-model-len 40960 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 64 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --quantization ascend \
        --compilation-config '{"cudagraph_capture_sizes": [64]}' \
        --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'
    ```

=== "Atlas 300I DUO"

    Qwen3-8B-W8A8SC:

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0

    # Ensure the model path matches the directory recorded during download
    vllm serve Eco-Tech/Qwen3-8B-w8a8sc-310-vllm/TP1/Qwen3-8B-w8a8sc-310-vllm-tp1 \
        --host 127.0.0.1 \
        --port 8080 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 32 \
        --served-model-name qwen3 \
        --dtype float16 \
        --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32]}' \
        --quantization ascend \
        --max-model-len 20480 \
        --no-enable-prefix-caching \
        --load-format sharded_state
    ```

    Qwen3-14B-W8A8SC:

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0

    # Ensure the model path matches the directory recorded during download
    vllm serve Eco-Tech/Qwen3-14B-w8a8sc-310-vllm/TP1/Qwen3-14B-w8a8sc-310-vllm-tp1 \
        --host 127.0.0.1 \
        --port 8080 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 16 \
        --served-model-name qwen3 \
        --dtype float16 \
        --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16]}' \
        --quantization ascend \
        --max-model-len 20480 \
        --no-enable-prefix-caching \
        --load-format sharded_state
    ```

    Qwen3-32B-W8A8SC:

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

    # Ensure the model path matches the directory recorded during download
    vllm serve Eco-Tech/Qwen3-32B-w8a8sc-310-vllm/TP4/Qwen3-32B-w8a8sc-310-vllm-tp4 \
        --host 127.0.0.1 \
        --port 8080 \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 32 \
        --served-model-name qwen3 \
        --dtype float16 \
        --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [16,32]}' \
        --quantization ascend \
        --max-model-len 20480 \
        --no-enable-prefix-caching \
        --load-format sharded_state
    ```

!!! note

    - [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/cli/serve/#arguments) — Additional parameter details for vLLM serve commands.
    - [Environment Variables](../../user_guide/configuration/env_vars.md) — Ascend-specific environment variables (`HCCL_*`, etc.).

**Service Verification:**

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```bash
curl http://localhost:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3",
        "messages": [
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_completion_tokens": 4096
    }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### LLM-Compressor Qwen3-32B-W8A8 on 310P

The `compressed-tensors` path supports dense symmetric INT8 per-channel
weights with dynamic per-token INT8 activations and FP16 model execution.
Do not pass `--quantization ascend` for an LLM-Compressor checkpoint.
ModelSlim continues to use its existing quantization configuration and NZ
weight layout; compressed-tensors uses a separate contiguous NK scheme.

The following real-weight configuration was tested on Ascend310P3 with
**two devices**, CANN/NNAL 9.2.1, driver 25.5.0, Python 3.12.16,
PyTorch 2.10.0, torch-npu 2.10.0.post4, Transformers 5.19.0 and
compressed-tensors 0.17.0. The source was vLLM Ascend main snapshot
`24a76e8` plus this change, with native extensions built, and the verified
vLLM main commit `544f182d9f0c9bfe8bd7645525951d5e3692a542`.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve ramblingpolymath/Qwen3-32B-W8A8 \
    --revision c97ec599f6d17967bb1049180d81d8a3e959fc97 \
    --served-model-name qwen3-w8a8 \
    --dtype float16 --tensor-parallel-size 2 \
    --max-model-len 2048 --max-num-seqs 1 \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.8 \
    --enforce-eager --port 8000
```

Default Model Runner V2 offline generation returned `56` and `Paris`; input
and generated token IDs matched the independent CPU reference exactly.
Both eager and decode-graph serving passed `/v1/models`, four HTTP 200 chat
requests and three completed serial streaming requests. The model checkpoint
revision above was fixed, with all seven safetensors LFS SHA256 hashes verified.

| Evaluation | Configuration | Result |
| --- | --- | --- |
| MMLU computer_security | Full 100-question test split, 0-shot, raw prompt, eager TP2/FP16, Model Runner V1, max sequences 8 | 86/100 |
| Backend selection audit | Both TP workers, actual loaded model | 256 dedicated compressed-tensors 310P schemes per rank; contiguous INT8 weights |
| Eager streaming | Model Runner V2, three serial requests, 93 output tokens each | TPOT 0.5999–0.6007 s/token; 1.63–1.65 tokens/s |
| Eager memory snapshot | After the chat smoke requests; device 0/device 1 totals | 27827/28073 MB |

For decode-graph execution, set `TORCHINDUCTOR_COMPILE_THREADS=1` and replace
`--enforce-eager` with
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'`.
All other model, batching and concurrency settings remain the same.

| Model Runner V2 mode | Streaming TPOT | Output throughput | Device 0/device 1 memory snapshot |
| --- | --- | --- | --- |
| Eager | 0.5999–0.6007 s/token | 1.63–1.65 tokens/s | 27827/28073 MB |
| Decode graph `[1]` | 0.5985–0.5992 s/token | 1.63–1.65 tokens/s | 27995/28242 MB |

These are three serial streaming requests per mode on the same board and
software versions in separate runs. They establish graph compatibility;
the small latency difference does not establish a reliable performance gain.
Graph capture reported 4 seconds and 0.27 GiB. Other capture sizes were not
tested. TTFT is measured to the first non-empty streaming delta; the eager
samples were 0.6528/1.2411/1.2306 seconds and graph samples were
0.6450/1.2361/1.2270 seconds. TPOT is estimated from the first and last
non-empty deltas divided by completion tokens minus one; generated token
counts include template delimiters. HTTP, scheduling and generation overhead
are included in these request-level measurements.

The MMLU score is uncorrected. **Use Model Runner V1 for scoring** by setting
`VLLM_USE_V2_MODEL_RUNNER=0`: the current 310P V2 sampler explicitly rejects
logprobs and logits processors, including allowed-token constraints. This
is a sampler capability limit; ordinary V2 generation and serving were tested
separately. The validation driver verified that all four answer continuations
are single tokens and preserve the prompt tokenization boundary, so restricted
greedy selection has the same choice argmax as continuation log-likelihood.
The test parquet SHA256 was
`a6ad5774acf5133ac88cf1bb066c60db8668abee2d6a724cf2f9dfa7cb4f103d`.
The opt-in model config is
`tests/e2e/models/configs/Qwen3-32B-W8A8-CompressedTensors-310P.yaml`.
For the full repository evaluation flow, select it with
`VLLM_USE_V2_MODEL_RUNNER=0 pytest -sv tests/e2e/models/test_lm_eval_correctness.py --config tests/e2e/models/configs/Qwen3-32B-W8A8-CompressedTensors-310P.yaml --tp-size 2`.
The reported score used the independent equivalent-choice driver, rather than
this full pytest flow.

An independent Transformers 4.57.6 / compressed-tensors 0.20.1 CPU FP16
fake-quant reference loaded the same checkpoint with a 12 GiB CPU memory
budget and disk offload. Both smoke outputs matched exactly. On the fixed
MMLU subset at indices 0, 10, ..., 90, all 10 answer choices matched the NPU
run, including two questions both implementations answered incorrectly.
This is a ten-question alignment check, not a full 100-question reference
score or an unquantized-model accuracy comparison. An earlier CANN 9.1 / vLLM
0.23.0 / Ascend 0.23.0.post1 feature backport also scored 86/100; all 100
input token sequences and predictions matched the main-stack V1 run.

The memory values are snapshots, not peak allocation measurements. The
configured context limit is 2048 tokens, with MMLU prompts of 44–221 tokens;
these runs do not establish maximum-length inference. The reported KV-cache
allocation capacity does not establish 40960-token inference or eight concurrent
maximum-length requests. Static W8A8, W4A8, FP8/MXFP, BF16 activation and
quantized MoE are outside this compressed-tensors support scope. See the
[quantization guide](../../user_guide/feature_guide/quantization.md) for the
capability matrix.

### 7.1 Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

The following is an example configuration for the accuracy evaluation config file:

**Accuracy Evaluation Config File:**

```bash
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="your_model_path",
        model="qwen3",
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1",
        host_port=2001,
        max_out_len=32768,
        batch_size=32,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

**Run the accuracy evaluation using the aime2025 dataset as an example:**

```bash
ais_bench --models vllm_api_general_chat --datasets aime2025_gen_0_shot_chat_prompt --debug
```

> The --models parameter value corresponds to the abbr field in the configuration file above. Adjust max_out_len, batch_size, and dataset tasks based on your scenario.

## 8 Performance Evaluation

### 8.1 Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

The following is an example configuration for the accuracy evaluation config file:

```bash
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="your_model_path",
        model="qwen3",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="127.0.0.1",
        host_port=8004,
        max_out_len=1500,
        batch_size=90,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True
        ),
    )
]
```

**Run the performance evaluation using the GSM8K dataset as an example:**

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts 360
```

### 8.2 Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take `serve` as an example:

```shell
vllm bench serve \
    --model your_model_path \
    --served-model-name qwen3 \
    --port <port> \
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

After several minutes, you will get the performance evaluation result.

## 9 Performance Tuning

Please refer to the [vLLM Features](https://docs.vllm.ai/en/stable/features), [vLLM Ascend Additional Configuration](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/additional_config.html), [vLLM Ascend Feature Matrix](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/feature_matrix.html) and [vLLM Ascend Feature Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide) for detailed key parameter descriptions.

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput | Single-Node (TP4) | 4 (Atlas A3 inference products) | W8A8 | 4-card TP maximizes concurrent request processing |
| Long Context | Single-Node (TP4) | 4 (Atlas A3 inference products) | W8A8 | 4-card TP extends context window for long sequences |
| Low Latency | Single-Node (TP8) | 8 (Atlas A3 inference products) | W8A8 | 8-card TP reduces per-token latency for interactive responses |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | FUSED_MC2 | EP Switch | Async Scheduling |
|----------|---------------|-------|----|----|-------------|--------------|--------------|
| High Throughput | Single-Node | 4 | 4 | 1 | Off | Off | On |
| Long Context | Single-Node | 4 | 4 | 1 | Off | Off | On |
| Low Latency | Single-Node | 8 | 8 | 1 | Off | Off | On |

>For detailed parameter descriptions, please refer to the deployment examples in [Section 5.1](#51-single-node-online-deployment).

<u>High Throughput Configuration:</u>

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3 \
    --trust-remote-code \
    --distributed-executor-backend "mp" \
    --tensor-parallel-size 4 \
    --max-model-len 5500 \
    --max-num-batched-tokens 40960 \
    --no-enable-prefix-caching \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,64,72,76,80,96,100,120,140,144,160,192,216,240,252,288,320,336,360,384,400,408,416,420,432,480,540,576,600]}' \
    --additional-config '{"weight_prefetch_config":{"enabled":true}, "pa_shape_list":[32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256]}' \
    --host <host_ip> \
    --port <port> \
    --block-size 128 \
    --gpu-memory-utilization 0.9
```

<u>Long Context Configuration:</u>

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --host <host_ip> \
    --port <port> \
    --served-model-name qwen3 \
    --trust-remote-code \
    --seed 1024 \
    --max-model-len 135000 \
    --max-num-batched-tokens 40960 \
    --tensor-parallel-size 4 \
    --distributed-executor-backend "mp" \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "eagle3", "model":"your_eagle3_model_path", "num_speculative_tokens": 3}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":131072}}' \
    --gpu-memory-utilization 0.9 \
    --quantization ascend
```

<u>Low Latency Configuration:</u>

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3 \
    --trust-remote-code \
    --distributed-executor-backend "mp" \
    --tensor-parallel-size 8 \
    --max-model-len 5500 \
    --max-num-batched-tokens 40960 \
    --no-enable-prefix-caching \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,72,76,80,96,100,120,140,144,160,192,216,240,252,288,320,336,360,384,400,408,416,420,432,480,540,576,600]}' \
    --speculative-config '{"method": "eagle3", "model":"your_eagle3_model_path", "enforce_eager": true, "num_speculative_tokens": 3}' \
    --port <port> \
    --block-size 128 \
    --gpu-memory-utilization 0.9
```

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md). This section only covers issues specific to Qwen3 Dense models.
