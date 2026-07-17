# Qwen3.5-Dense (Qwen3.5-2B/4B/9B)

## 1 Introduction

Qwen3.5-2B, Qwen3.5-4B, and Qwen3.5-9B are dense hybrid Mamba-Transformer language models in the Qwen3.5 family. They share the same hybrid attention design (GDN + full attention) and are suitable for general-purpose text generation tasks such as dialogue, content creation, and code generation.

This document describes deployment and verification of these models on **Atlas inference products** and **Atlas 200I Pro**, including environment preparation, Docker installation, single-node online deployment, functional verification, and tuning notes.

It is **strongly recommended to use the latest release candidate (rc) version or the latest official version** of `vllm-ascend`. Support for Qwen3.5-2B/4B/9B on Atlas inference products and Atlas 200I Pro starts from `vllm-ascend:v0.23.0rc1`.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

| Model | Version | Hardware Requirement | Download |
|-------|---------|----------------------|----------|
| Qwen3.5-2B | FP16 | Atlas inference products or Atlas 200I Pro | [Download](https://modelscope.cn/models/Qwen/Qwen3.5-2B) |
| Qwen3.5-4B | FP16 | Atlas inference products or Atlas 200I Pro | [Download](https://modelscope.cn/models/Qwen/Qwen3.5-4B) |
| Qwen3.5-9B | FP16 | Atlas inference products or Atlas 200I Pro | [Download](https://modelscope.cn/models/Qwen/Qwen3.5-9B) |

It is recommended to download the model weight to a local directory such as `/root/.cache/` or `/home/data/`.

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

It is **recommended to use the latest release candidate (rc) version or the latest official version** of the `vllm-ascend` image. As a minimum-version requirement, use `vllm-ascend:v0.23.0rc1-310p` (or a later `-310p`) image. For Atlas 200I Pro on openEuler, use the matching `-310p-openeuler` image.

:::::::{tab-set}

::::::{tab-item} Atlas inference products

Start the docker image on your each node.

```{code-block} bash
  :substitutions:
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p
docker run --rm \
    --name vllm-ascend \
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
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8080:8080 \
    -it $IMAGE bash
```

::::::

::::::{tab-item} Atlas 200I Pro

Start the docker image on your each node. Adjust `--device=/dev/davinci0` according to the NPU ID you want to use.

:::::{tab-set}

::::{tab-item} Ubuntu 24.04
:selected:

```{code-block} bash
  :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p

docker run --rm \
--privileged \
--name vllm-ascend \
--shm-size=10g \
--device=/dev/davinci0:/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/ascend_manager \
--device=/dev/user_config \
-v /etc/sys_version.conf:/etc/sys_version.conf \
-v /etc/ld.so.conf.d/mind_so.conf:/etc/ld.so.conf.d/mind_so.conf \
-v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg \
-v /var/dmp_daemon:/var/dmp_daemon \
-v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
-v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
-v /usr/lib/aarch64-linux-gnu/libyaml-0.so.2:/usr/lib64/libyaml-0.so.2 \
-v /etc/slog.conf:/etc/slog.conf \
-v /var/slogd:/var/slogd \
-v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
-v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so \
-v /root/.cache:/root/.cache \
-p 8080:8080 \
-it $IMAGE bash
```

::::

::::{tab-item} openEuler 24.03

```{code-block} bash
  :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p-openeuler

docker run --rm \
--privileged \
--name vllm-ascend \
--shm-size=10g \
--device=/dev/davinci0:/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/ascend_manager \
--device=/dev/user_config \
-v /etc/sys_version.conf:/etc/sys_version.conf \
-v /etc/ld.so.conf.d/mind_so.conf:/etc/ld.so.conf.d/mind_so.conf \
-v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg \
-v /var/dmp_daemon:/var/dmp_daemon \
-v /usr/lib64/libsemanage.so.2:/usr/lib64/libsemanage.so.2 \
-v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
-v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
-v /usr/lib64/libyaml-0.so.2.0.9:/usr/lib64/libyaml-0.so.2 \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
-v /etc/slog.conf:/etc/slog.conf \
-v /var/slogd:/var/slogd \
-v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
-v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so \
-v /root/.cache:/root/.cache \
-p 8080:8080 \
-it $IMAGE bash
```

::::
:::::

::::::
:::::::

After a successful docker run, you can verify the running container service by executing the `docker ps` command. The expected result is that the container `vllm-ascend` is listed with status `Up`, confirming the docker installation is successful.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

1. Clone the repository and install `vllm-ascend` from source:

    ```bash
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    pip install -e .
    ```

    For the complete installation steps, refer to [installation](../../installation.md).

````{note}
On Atlas inference products and Atlas 200I Pro, you may need to uninstall `triton-ascend` and `triton` to avoid dependency conflicts:

```bash
pip uninstall -y triton-ascend triton
```
````

To verify the source code installation, run the following command and confirm the displayed version matches the one you installed:

```bash
pip show vllm-ascend
```

Expected result: The version information of `vllm-ascend` is displayed, confirming a successful installation.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. `Qwen3.5-2B`, `Qwen3.5-4B`, and `Qwen3.5-9B` can be deployed on Atlas inference products or Atlas 200I Pro.

> **Parallelism note**: These platforms currently support the **TP** scenario. Choose **TP=1** or **TP=2** according to the available devices. On Atlas 200I Pro with a single visible NPU, use **TP=1**.

The following examples use FP16 weights from ModelScope. Replace `MODEL_PATH` with your local directory if needed.

:::::{tab-set}

::::{tab-item} Qwen3.5-2B

Startup Command:

```bash
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Model weight path; can be a ModelScope model id or a local directory path
export MODEL_PATH=Qwen/Qwen3.5-2B

vllm serve $MODEL_PATH \
--host 127.0.0.1 \
--port 1025 \
--tensor-parallel-size 1 \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 16384 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--mamba-ssm-cache-dtype float16 \
--dtype float16 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
```

::::

::::{tab-item} Qwen3.5-4B

Startup Command:

```bash
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Model weight path; can be a ModelScope model id or a local directory path
export MODEL_PATH=Qwen/Qwen3.5-4B

vllm serve $MODEL_PATH \
--host 127.0.0.1 \
--port 1025 \
--tensor-parallel-size 1 \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 16384 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--mamba-ssm-cache-dtype float16 \
--dtype float16 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
```

::::

::::{tab-item} Qwen3.5-9B

Startup Command:

```bash
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Model weight path; can be a ModelScope model id or a local directory path
export MODEL_PATH=Qwen/Qwen3.5-9B

vllm serve $MODEL_PATH \
--host 127.0.0.1 \
--port 1025 \
--tensor-parallel-size 1 \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 16384 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--mamba-ssm-cache-dtype float16 \
--dtype float16 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
```

::::
:::::

Key Parameter Descriptions:

- `--tensor-parallel-size` sets the tensor parallel size. Prefer **TP=1** on Atlas 200I Pro. On Atlas inference products, **TP=1** and **TP=2** are both supported; choose according to the available devices.
- `--max-model-len` represents the context length (input plus output for a single request). On Atlas inference products and Atlas 200I Pro, configure this value according to the actual device memory; setting it too high may cause OOM.
- `--max-num-seqs` indicates the maximum number of requests that can be processed concurrently. On Atlas inference products and Atlas 200I Pro, configure this value according to the actual device memory; setting it too high may cause OOM.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. On Atlas inference products and Atlas 200I Pro, configure this value according to the actual device memory; setting it too high may cause OOM. The default value is `0.9`.
- `--dtype float16` must be set on Atlas inference products and Atlas 200I Pro. These devices only support the FP16 data type.
- `--mamba-ssm-cache-dtype` sets the data type of the Mamba SSM cache. On Atlas inference products and Atlas 200I Pro, only `float16` is supported.
- `--speculative-config` uses `qwen3_5_mtp` for Qwen3.5 Dense models that include an MTP head.
- `--compilation-config` contains configurations related to the aclgraph graph mode:
    - `"cudagraph_mode"`: `"FULL_DECODE_ONLY"` is recommended.
    - `"cudagraph_capture_sizes"`: when tensor parallelism (TP) is enabled, hardware event-id constraints allow at most two capture sizes (for example, `[1, 8]`).
- `--additional-config` with `"ascend_compilation_config": {"enable_npugraph_ex": false}` is required because `enable_npugraph_ex` is not supported on these platforms.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. Two API interfaces are supported: `completions` and `chat.completions`. Use the `--served-model-name` you configured (for example, `qwen3.5`). If you used `--port 1025` or `-p 8080:8080`, adjust the URL accordingly.

**Completions API:**

```bash
curl http://127.0.0.1:1025/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

**Chat Completions API:**

```bash
curl http://127.0.0.1:1025/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95
    }'
```

Expected Result: The service returns HTTP 200 OK. The JSON response contains the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. Here are the accuracy results of `Qwen3.5-2B`, `Qwen3.5-4B`, and `Qwen3.5-9B` on Atlas inference products for reference only.

**Accuracy Evaluation Config File:**

```python
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="your_model_path",
        model="qwen3.5",
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1",
        host_port=1025,
        max_out_len=4096,
        batch_size=16,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.0,
            ignore_eos=False,
            chat_template_kwargs = {"enable_thinking": False},
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

| Model | dataset | version | metric | mode | vllm-api-general-chat |
|-------|---------|---------|--------|------|------------------------|
| Qwen3.5-2B | gsm8k | - | accuracy | gen | 77.71 |
| Qwen3.5-2B | textvqa | - | accuracy | gen | 76.09 |
| Qwen3.5-4B | gsm8k | - | accuracy | gen | 93.18 |
| Qwen3.5-4B | textvqa | - | accuracy | gen | 79.08 |
| Qwen3.5-9B | gsm8k | - | accuracy | gen | 95.30 |
| Qwen3.5-9B | textvqa | - | accuracy | gen | 82.33 |

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are for reference only. The optimal configuration depends on model size, maximum input/output length, and actual device memory.
>
> **Atlas inference products / Atlas 200I Pro**: Currently only the TP scenario is supported. Prefer **TP=1** on Atlas 200I Pro. On Atlas inference products, **TP=1** and **TP=2** are both supported; choose according to the available devices. Configure `--max-model-len`, `--max-num-seqs`, and `--gpu-memory-utilization` based on the actual device memory; setting them too high may cause OOM.

### 9.2 Tuning Guidelines

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
