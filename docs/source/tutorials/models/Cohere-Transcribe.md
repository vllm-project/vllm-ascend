# Cohere Transcribe

## 1 Introduction

Cohere Transcribe is a family of automatic speech recognition (ASR) models from Cohere. The model covered in this document is `CohereLabs/cohere-transcribe-arabic-07-2026`, a 2B-parameter Conformer encoder-decoder ASR model that supports 14 languages and is specialized for Arabic speech recognition.

This document describes the supported features, environment preparation, single-node deployment, functional verification, and evaluation workflow for Cohere Transcribe on Ascend NPUs.

Cohere Transcribe is supported by upstream vLLM (via the `cohere_asr` model implementation). Use a vLLM-Ascend image that matches your vLLM version, and refer to the support matrix for the current release status.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The BF16 model can be deployed with one Ascend 910B 64 GB NPU. Download the model weights from [Hugging Face](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) (or your configured model hub).

Download the weights to a directory that is accessible from the deployment environment. For multi-node deployments, use a shared directory; for example, `/root/.cache/`.

## 4 Installation

### 4.1 Docker Image Installation

Use the vLLM-Ascend Docker image that corresponds to your hardware. Replace the model-weight mount with the path used in your environment.

=== "Atlas A2 inference products"

    ```bash
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

Verify that the container is running and that the installed package version matches the image tag:

```bash
docker ps --filter name=vllm-ascend
pip show vllm vllm-ascend
```

Expected result: `docker ps` lists the container with status `Up`, and `pip show` displays version information for both packages.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../installation.md).

To verify the source installation:

```bash
pip show vllm-ascend
```

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs both audio prefill and decoding on one NPU, making it suitable for development, testing, and small-scale ASR services. Replace `your_model_path` with the local model directory.

=== "Atlas A2 inference products"

    ```shell
    vllm serve your_model_path \
      --served-model-name cohere-transcribe \
      --trust-remote-code \
      --tensor-parallel-size 1 \
      --dtype bfloat16 \
      --max-model-len 4096 \
      --block-size 128 \
      --enforce-eager \
      --port 8000
    ```

    !!! note

        - `--trust-remote-code` is required because the model repository ships custom modeling code.
        - `--tensor-parallel-size 1` uses one NPU. Increase it only after confirming that the hardware and deployment topology support the chosen parallel configuration.
        - `--dtype bfloat16` matches the BF16 deployment validated on Ascend 910B.
        - `--max-model-len 4096` limits the maximum sequence length. Always specify a conservative value explicitly for ASR workloads; automatic detection can allocate an oversized attention mask and cause an out-of-memory error.

When the service starts successfully, the log contains `Application startup complete`. If startup fails, see the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).

## 6 Functional Verification

After the service is started, the model can be invoked by sending an audio prompt.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "cohere-transcribe",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://example.com/your_audio.wav"
                        }
                    }
                ]
            }
        ]
    }'
```

Replace `localhost`, `8000`, and `cohere-transcribe` with the address, port, and `--served-model-name` used by your deployment. Expected result: HTTP 200 and a JSON response containing the transcription in the `choices` field.

## 7 Accuracy Evaluation

Evaluate transcription quality with Word Error Rate (WER) for word-level recognition.

On the Common Voice 18 (CV18) Arabic test set (10,471 samples), the model deployed on Ascend 910B achieves a WER of 5.69%, close to the official result (5.82%). The WER distribution is as follows:

| Metric | Result |
| --- | --- |
| WER = 0 (perfect recognition) | 2,931 / 10,471 (28.0%) |
| WER < 5% | 6,990 / 10,471 (66.8%) |
| WER < 10% | 8,116 / 10,471 (77.5%) |
| WER < 20% | 9,145 / 10,471 (87.3%) |

## 8 Performance Evaluation

Measure ASR serving performance with audio samples that represent the production workload. Record at least the audio duration, request concurrency, end-to-end latency, real-time factor, and throughput. This ensures that audio preprocessing, request construction, API communication, inference, and response parsing are included in the result.

On Ascend 910B, the model achieves a real-time factor (RTF) of 0.100, i.e. an RTFx of 10.0: 1 second of audio is transcribed in about 0.1 seconds, so the service can process audio roughly 10x faster than real time.

Actual performance varies with hardware, audio duration, concurrency, and deployment configuration. Evaluate short audio, long audio, and concurrent requests separately before selecting a production configuration.

## 9 Performance Tuning

The following settings are starting points rather than globally optimal configurations. Tune them according to audio duration, concurrency, latency requirements, and available NPU memory.

| Scenario | Recommended Starting Point | Key Considerations |
| --- | --- | --- |
| Low latency | `--tensor-parallel-size 1`, `--max-model-len 4096` | Use short audio inputs and avoid sharing the NPU with other workloads. |
| High throughput | Increase request concurrency after establishing the latency baseline | Monitor NPU memory and end-to-end latency; do not use synthetic text-only requests as a proxy for ASR traffic. |
| Long audio | Increase `--max-model-len` only as required | Keep the value conservative because attention-mask memory grows with the configured maximum length. |

For general parameter tuning, refer to the [Performance Tuning Guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

## 10 FAQ

For common environment, installation, and general parameter issues, see the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This section covers model- and hardware-specific guidance.

### The server fails to start with an out-of-memory error

**Symptom:** The server fails with an out-of-memory error while initializing attention.

**Cause:** An automatically detected large context length can create a full causal attention mask whose memory consumption grows quadratically with `max_model_len`.

**Solution:** Always set `--max-model-len` explicitly to a conservative value, such as `4096`, and increase it only after verifying available NPU memory.
