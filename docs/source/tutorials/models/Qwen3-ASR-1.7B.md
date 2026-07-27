# Qwen3-ASR-1.7B

## 1 Introduction

Qwen3-ASR-1.7B is a 1.7B-parameter automatic speech recognition (ASR) model from the Qwen team. It supports Chinese and English speech, Chinese dialects, multilingual speech, and singing voice transcription, and provides long-audio and streaming inference capabilities.

This document describes the supported features, environment preparation, single-node deployment, functional verification, and evaluation workflow for Qwen3-ASR-1.7B on Ascend NPUs.

Qwen3-ASR-1.7B was introduced with upstream vLLM v0.19.0. Use a vLLM-Ascend image that matches your vLLM version, and refer to the support matrix for the current release status.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The BF16 model can be deployed with one Ascend 910B 64 GB NPU or one Ascend Atlas inference products 48 GB NPU. Download the model weights from [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B).

Download the weights to a directory that is accessible from the deployment environment. For multi-node deployments, use a shared directory; for example, `/root/.cache/`.

## 4 Installation

### 4.1 Docker Image Installation

Use the vLLM-Ascend Docker image that corresponds to your hardware. Replace the model-weight mount with the path used in your environment.

:::::{tab-set}
::::{tab-item} Atlas A2 inference products
:sync: A2

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net host \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it -d $IMAGE bash
```

::::

::::{tab-item} Atlas inference products

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p

docker run --rm \
        --name vllm-ascend \
        --shm-size=10g \
        --net host \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it -d $IMAGE bash
```

::::
:::::

Verify that the container is running and that the installed package version matches the image tag:

```bash
docker ps --filter name=vllm-ascend
pip show vllm vllm-ascend
```

Expected result: `docker ps` lists the container with status `Up`, and `pip show` displays version information for both packages.

### 4.2 Source Code Installation

f you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../installation.md).

:::{note}

    For Atlas inference products, source installation may pull in `triton` and `triton-ascend`. Uninstall them before running vLLM-Ascend on Atlas inference products:

    ```bash
    pip uninstall -y triton-ascend triton
    ```

To verify the source installation:

```bash
pip show vllm-ascend
```

:::

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment runs both audio prefill and decoding on one NPU, making it suitable for development, testing, and small-scale ASR services. Replace `your_model_path` with the local model directory, or use `Qwen/Qwen3-ASR-1.7B` to download the model through the configured model hub.

:::::{tab-set}
::::{tab-item} Atlas A2 inference products

```shell
vllm serve your_model_path \
    --served-model-name qwen3-asr \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --port 8000
```

::::
::::{tab-item} Atlas inference products

```{code-block} bash
   :substitutions:

vllm serve your_model_path \
  --served-model-name qwen3-asr \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --dtype float16 \
  --max-model-len 4096 \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false,"enable_npu_graph_ex":false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,4]}' \
  --port 8000
```

::::

:::::

Key parameters Descriptions:

- `--tensor-parallel-size 1` uses one NPU. Increase it only after confirming that the hardware and deployment topology support the chosen parallel configuration.
- `--max-model-len 4096` limits the maximum sequence length. On Atlas 300I DUO, always specify a conservative value explicitly; automatic detection can allocate an oversized attention mask and cause an out-of-memory error.
- `--gpu-memory-utilization 0.9` sets the fraction of device memory available to the vLLM executor. Lower this value if other workloads share the NPU.
- `--enforce-eager` disables graph execution. It is used in the Atlas 300I A2 2UP example for compatibility.

When the service starts successfully, the log contains `Application startup complete`. If startup fails, see the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```{code-block} bash
   :substitutions:

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-asr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                        }
                    }
                ]
            }
        ]
    }'
```

Replace `localhost`, `8000`, and `qwen3-asr` with the address, port, and `--served-model-name` used by your deployment. Expected result: HTTP 200 and a JSON response containing the transcription in the `choices` field.

## 7 Accuracy Evaluation

Evaluate transcription quality with Word Error Rate (WER) for word-level recognition and Character Error Rate (CER) for character-level recognition.

## 8 Performance Evaluation

Measure ASR serving performance with audio samples that represent the production workload. Record at least the audio duration, request concurrency, end-to-end latency, real-time factor, and throughput. This ensures that audio preprocessing, request construction, API communication, inference, and response parsing are included in the result.

Actual performance varies with hardware, audio duration, concurrency, and deployment configuration. Evaluate short audio, long audio, and concurrent requests separately before selecting a production configuration.

## 9 Performance Tuning

The following settings are starting points rather than globally optimal configurations. Tune them according to audio duration, concurrency, latency requirements, and available NPU memory.

| Scenario | Recommended Starting Point | Key Considerations |
| --- | --- | --- |
| Low latency | `--tensor-parallel-size 1`, `--max-model-len 4096` | Use short audio inputs and avoid sharing the NPU with other workloads. |
| High throughput | Increase request concurrency after establishing the latency baseline | Monitor NPU memory and end-to-end latency; do not use synthetic text-only requests as a proxy for ASR traffic. |
| Long audio | Increase `--max-model-len` only as required | On Atlas inference products, keep the value conservative because attention-mask memory grows with the configured maximum length. |

For general parameter tuning, refer to the [Performance Tuning Guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

## 10 FAQ

For common environment, installation, and general parameter issues, see the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This section covers model- and hardware-specific guidance.

### Atlas inference products runs out of memory during startup

**Symptom:** The server fails with an out-of-memory error while initializing attention.

**Cause:** On Atlas inference products, an automatically detected large context length can create a full causal attention mask whose memory consumption grows quadratically with `max_model_len`.

**Solution:** Always set `--max-model-len` explicitly to a conservative value, such as `4096`, and increase it only after verifying available NPU memory.
