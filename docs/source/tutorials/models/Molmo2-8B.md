# Molmo2-8B (Molmo2ForConditionalGeneration)

## 1 Introduction

[Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) is an open vision–language model from the Allen Institute for AI (Ai2). It uses a **Qwen3-8B** language backbone and **SigLIP 2** as the vision encoder, and supports **image** and **video** inputs with pointing and dense captioning-style outputs. In upstream vLLM it is registered as `Molmo2ForConditionalGeneration`; see the [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html) list.

This document will demonstrate the primary validation steps for the model, including environment preparation, deployment, functional verification, and accuracy evaluation.

This document is validated and written based on **vLLM-Ascend v0.13.0**. The current model (Molmo2-8B) is fully supported in this version, and all **v0.13.0 and later versions** can run stably.

## 2 Feature Matrix

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Environment Preparation

### 3.1 Model Weight

| Model Version | Hardware Requirements | Download Link |
| ---------- | ---------- | ---------- |
| Molmo2-8B (BF16) | 1×Atlas 800 A2 (64G) | [Hugging Face](https://huggingface.co/allenai/Molmo2-8B) |

Optional mirror: set `export VLLM_USE_MODELSCOPE=true` if you pull weights via ModelScope equivalents.

Weights are large; use a shared cache (for example `/root/.cache/huggingface`) on the host and mount it into the container when using Docker.

### 3.2 Verify Multi-node Communication (Optional)

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

Use the same Ascend driver/CANN stack and vLLM Ascend image as in [Installation](../../installation.md) and [Quickstart](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html).

Example (single NPU, adapt image tag to your release):

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

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

Recommended environment variables:

```bash
export VLLM_USE_MODELSCOPE=true   # optional, for faster downloads in CN regions
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for most inference scenarios.

Startup Command:

```{test} bash
:sync-yaml: tests/e2e/models/configs/Molmo2-8B.yaml
:sync-target: model_name doc_serve_options
:sync-class: cmd

vllm serve allenai/Molmo2-8B \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-num-batched-tokens 36864 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 1
```

Key parameters:

- `--trust-remote-code`: Required for loading model with custom code
- `--dtype bfloat16`: Use BF16 precision for inference
- `--max-num-batched-tokens 36864`: Maximum number of tokens in a batch for long visual sequences
- `--limit-mm-per-prompt.image 1`: Limit images per prompt
- `--limit-mm-per-prompt.video 1`: Limit videos per prompt

:::{note}
On a single **64G** NPU, if you hit OOM, lower `max_model_len`, reduce `max_num_batched_tokens`, or serve with `tensor_parallel_size` across more NPUs.
:::

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt:

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "allenai/Molmo2-8B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                    {"type": "text", "text": "What is in this image?"}
                ]
            }
        ],
        "max_tokens": 128,
        "temperature": 0
    }'
```

## 7 Accuracy Evaluation

Nightly accuracy for this model is configured in `tests/e2e/models/configs/Molmo2-8B.yaml` using the `mmmu_val` task. The reference `acc,none` target is aligned with the public **~53% MMMU** figure for Molmo2-8B.

### Using AISBench

For details, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

### Using Language Model Evaluation Harness

For details, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).

## 8 Best Practices

Provide recommended configurations for different scenarios:

- **Long sequence**: Set `max_num_batched_tokens` to a generous value (e.g., 36864) when processing long visual sequences.
- **Low latency**: Use single-NPU deployment with `trust_remote_code=True` and `dtype=bfloat16`.
- **High throughput**: Deploy across multiple NPUs with `tensor_parallel_size` for better throughput.

## 9 FAQ

### Problem: Download timeout during model weight fetching

**Phenomenon**: `Error while downloading ... model-00005-of-00008.safetensors ... Read timed out.`

**Cause**: Large model shard download from Hugging Face is unstable.

**Solution**: Pre-download model weights locally and use local path for deployment.

### Problem: OOM on single NPU

**Phenomenon**: Out of memory error when serving the model.

**Cause**: Model requires more memory than available on single NPU.

**Solution**: Lower `max_model_len`, reduce `max_num_batched_tokens`, or use `tensor_parallel_size` across more NPUs.

## References

- [Molmo2 announcement (Ai2)](https://allenai.org/blog/molmo2)
- [vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Molmo2-8B model card](https://huggingface.co/allenai/Molmo2-8B)
