# Cohere Transcribe

## 1 Introduction 

Cohere Transcribe is a family of automatic speech recognition (ASR) models from Cohere, based on a 2B-parameter Conformer encoder-decoder architecture that supports 14 languages (English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Chinese (Mandarin), Japanese, Korean, Vietnamese, and Arabic).

This document covers two model versions verified on Ascend NPUs:

| Version | Description | Verified by |
| --- | --- | --- |
| `CohereLabs/cohere-transcribe-03-2026` | Base multilingual model (14 languages) | Colleague verification, results recorded in the internal evaluation report |
| `CohereLabs/cohere-transcribe-arabic-07-2026` | Arabic fine-tuned model | Internal evaluation report (WER / RTFx measured on Atlas A2 products) |

This document describes the supported features, environment preparation, single-node deployment, functional verification, and evaluation workflow for Cohere Transcribe on Ascend NPUs.

Cohere Transcribe is supported by upstream vLLM (via the `cohere_asr` model implementation). Use a vLLM-Ascend image that matches your vLLM version, and refer to the support matrix for the current release status.

The current release is adapted for Atlas A2 inference products. Support for the Ascend 950DT series is planned for the next phase.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The BF16 model can be deployed with one Atlas A2 64 GB NPU. Download the model weights from any of the following sources:

- GitCode mirror (03-2026): [weixin_62994174/CohereLabs_cohere-transcribe-03-2026](https://ai.gitcode.com/weixin_62994174/CohereLabs_cohere-transcribe-03-2026)
- Hugging Face: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) or [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
- ModelScope: [CohereLabs/cohere-transcribe-03-2026](https://modelscope.cn/models/CohereLabs/cohere-transcribe-03-2026)

Note that the model repository ships custom modeling code, so `--trust-remote-code` is required when serving.

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

!!! note

    If you build a custom image (for example, based on a vLLM-Ascend image matching your vLLM version), make sure `librosa` is installed in the container before serving Cohere Transcribe:

    ```bash
    pip install librosa
    ```

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../installation.md).

To verify the source installation:

```bash
pip show vllm-ascend
```

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs both audio prefill and decoding on one NPU, making it suitable for development, testing, and small-scale ASR services. Both model versions can be deployed on one Atlas A2 NPU with the same parameters; only the model weight path changes.

=== "Atlas A2 inference products"

    The following examples are for Atlas A2 inference products.

    === "cohere-transcribe-03-2026"

        ```shell
        export ASCEND_RT_VISIBLE_DEVICES=0
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        vllm serve /data/llm-workspace/cohere-transcribe-03-2026 \
          --served-model-name cohere-transcribe \
          --trust-remote-code \
          --tensor-parallel-size 1 \
          --dtype bfloat16 \
          --enforce-eager \
          --block-size 128 \
          --host 0.0.0.0 \
          --port 8000
        ```

    === "cohere-transcribe-arabic-07-2026"

        ```shell
        export ASCEND_RT_VISIBLE_DEVICES=0
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        vllm serve /data/llm-workspace/cohere-transcribe-arabic-07-2026 \
          --served-model-name cohere-transcribe \
          --trust-remote-code \
          --tensor-parallel-size 1 \
          --dtype bfloat16 \
          --enforce-eager \
          --block-size 128 \
          --host 0.0.0.0 \
          --port 8000
        ```

    !!! note

        - `--trust-remote-code` is required because the model repository ships custom modeling code.
        - `--block-size 128` is required: the model must be served with a block size of at least 128.
        - `--tensor-parallel-size 1` uses one NPU. Increase it only after confirming that the hardware and deployment topology support the chosen parallel configuration.
        - `--dtype bfloat16` matches the BF16 deployment validated on Atlas A2 products.
        - `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are recommended when the model weights are downloaded to a local directory in advance.

When the service starts successfully, the log contains `Application startup complete`. If startup fails, see the [Public FAQs](../../faqs.md).

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

Evaluate transcription quality with Word Error Rate (WER) for word-level recognition and Character Error Rate (CER) for character-level recognition.

### 7.1 cohere-transcribe-03-2026 (base multilingual model)

Verified by a colleague on the same benchmark as 07-2026 (see Section 7.2). Results are recorded in the internal evaluation report, which compares the two versions side by side.

| Metric | Result (100 Common Voice Arabic samples) |
| --- | --- |
| WER | 9.06% |
| CER | 2.96% |

### 7.2 cohere-transcribe-arabic-07-2026 (Arabic fine-tuned model)

Measured in the internal evaluation report on Atlas A2 products.

On the Common Voice 18 (CV18) Arabic test set (10,471 samples), the model achieves a WER of 5.69%, close to the official result (5.82%). The WER distribution is as follows:

| Metric | Result |
| --- | --- |
| WER = 0 (perfect recognition) | 2,931 / 10,471 (28.0%) |
| WER < 5% | 6,990 / 10,471 (66.8%) |
| WER < 10% | 8,116 / 10,471 (77.5%) |
| WER < 20% | 9,145 / 10,471 (87.3%) |

On the same 100 Common Voice Arabic samples used for 03-2026, the 07-2026 model achieves:

| Metric | Result (100 Common Voice Arabic samples) |
| --- | --- |
| WER | 5.28% |
| CER | 1.27% |

The 07-2026 Arabic fine-tuned model significantly outperforms the 03-2026 base model on Arabic (WER 5.28% vs 9.06%, CER 1.27% vs 2.96%), confirming that the fine-tuned version is recommended for Arabic production use.

## 8 Performance Evaluation

Measure ASR serving performance with audio samples that represent the production workload. Record at least the audio duration, request concurrency, end-to-end latency, real-time factor, and throughput. This ensures that audio preprocessing, request construction, API communication, inference, and response parsing are included in the result.

Both models were evaluated on the same 100 Common Voice Arabic samples (440 seconds total audio) with the same inference stack (vLLM 0.23.0 + vllm-ascend). Results from the internal evaluation report:

| Metric | 07-2026 Arabic | 03-2026 (base) |
| --- | --- | --- |
| RTF | 0.100 | 0.107 |
| RTFx | 10.0 | 9.4 |
| Inference time for 100 samples | 44 s | 47 s |

The 07-2026 Arabic model is slightly faster (RTF 0.100 vs 0.107, about 6.5% faster) while providing significantly better Arabic accuracy.

Actual performance varies with hardware, audio duration, concurrency, and deployment configuration. Evaluate short audio, long audio, and concurrent requests separately before selecting a production configuration.

## 9 Performance Tuning

The following settings are starting points rather than globally optimal configurations. Tune them according to audio duration, concurrency, latency requirements, and available NPU memory.

| Scenario | Recommended Starting Point | Key Considerations |
| --- | --- | --- |
| Low latency | `--tensor-parallel-size 1`, `--block-size 128` | Use short audio inputs and avoid sharing the NPU with other workloads. |
| High throughput | Increase request concurrency after establishing the latency baseline | Monitor NPU memory and end-to-end latency; do not use synthetic text-only requests as a proxy for ASR traffic. |
| Long audio | Increase `--max-model-len` only as required | Keep the value conservative because attention-mask memory grows with the configured maximum length. |

For general parameter tuning, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

## 10 FAQ

For common environment, installation, and general parameter issues, see the [Public FAQs](../../faqs.md). This section covers model- and hardware-specific guidance.

### The server fails to start with an out-of-memory error

**Symptom:** The server fails with an out-of-memory error while initializing attention.

**Cause:** An automatically detected large context length can create a full causal attention mask whose memory consumption grows quadratically with `max_model_len`.

**Solution:** If an out-of-memory error occurs, set `--max-model-len` explicitly to a conservative value (for example `4096`) and increase it only after verifying available NPU memory. Also confirm that `--block-size 128` is used as required by the model.
