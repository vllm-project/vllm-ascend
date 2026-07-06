# Llama-3.2-3B-Instruct

## Introduction

`Llama-3.2-3B-Instruct` is a lightweight, instruction-tuned large language model released by Meta. It uses the standard decoder-only (`LlamaForCausalLM`) architecture with GQA (Grouped-Query Attention), RoPE, SwiGLU and RMSNorm, has about 3.2B parameters, and is well suited for general-purpose chat, reasoning and text generation.

This document describes the main verification steps of the model on `vllm-ascend`, including supported features, environment preparation, single-node deployment, functional verification, and accuracy evaluation on the GSM8K benchmark.

This document is verified and written based on **vLLM-Ascend v0.11.0** (vLLM v0.11.0). `Llama-3.2` is a standard architecture natively supported by upstream vLLM, requiring no custom modeling code, patches or operators, and runs stably on v0.11.0 and later.

## Supported Features

Please refer to [Supported Models](../../user_guide/support_matrix/supported_models.md) for the feature support matrix. `Llama2/3/3.1/3.2` is already marked as supported (Atlas A2/A3). Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

| Model | Supported | BF16 | Hardware | Tensor Parallel | Chunked Prefill | Automatic Prefix Caching | LoRA | Full ACL Graph | Max Model Len |
| ----- | --------- | ---- | -------- | --------------- | --------------- | ------------------------ | ---- | -------------- | ------------- |
| Llama-3.2-3B-Instruct | ✅ | ✅ | Atlas A2/A3: min 1 card | ✅ | ✅ | ✅ | ✅ | ✅ | 128k |

## Environment Preparation

### Model Weight

`Llama-3.2-3B-Instruct` (BF16 version) requires only 1 Atlas 800I A2 (64G × 1) card.

Recommended download sources:

- ModelScope (no license gate, recommended): [LLM-Research/Llama-3.2-3B-Instruct](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct)
- Hugging Face (requires accepting the Llama 3.2 Community License): [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

It is recommended to place the model weight in a shared cache directory such as `/root/.cache/`.

### Installation

You can use the official docker image to deploy `Llama-3.2-3B-Instruct`:

```{code-block} bash
   :substitutions:

# For Atlas A2 machines (for Atlas A3, append the -a3 suffix to the image tag)
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

If you do not want to use the docker image, you can also build `vllm-ascend` from source, refer to [installation](../../installation.md).

To speed up weight download from ModelScope:

```bash
export VLLM_USE_MODELSCOPE=True
```

## Deployment

Start the online serving service with the following command. `Llama-3.2-3B-Instruct` runs on a single Atlas A2 (64G) card with `tensor-parallel-size 1`:

```bash
vllm serve "LLM-Research/Llama-3.2-3B-Instruct" \
    --served-model-name llama-3.2-3b-instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --port 8000
```

Key parameters:

- `--tensor-parallel-size 1`: number of cards for tensor parallelism; 1 for single card.
- `--max-model-len 4096`: maximum sequence length (input + output) per request; adjustable up to the model's native context length.
- `--gpu-memory-utilization 0.8`: NPU memory utilization, range (0, 1].

A successful startup prints `Application startup complete.` in the logs.

> Tip: for common environment, installation or general parameter questions, refer to the [public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).

## Functional Verification

Once your server is started, you can query the model with a chat prompt:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-3.2-3b-instruct",
        "messages": [
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0.7,
        "max_completion_tokens": 200
    }'
```

Expected result: HTTP 200, with a JSON response body containing a `choices` field. Measured sample output on Atlas A2 (910B, excerpt):

```text
**Introduction to Large Language Models**

Large language models (LLMs) are a type of artificial intelligence (AI) designed to
process and understand human language. These models have revolutionized the field of
natural language processing (NLP) and have numerous applications ... including chatbots,
language translation, text summarization, and more.
```

### Offline Inference

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=64)
llm = LLM(
    model="LLM-Research/Llama-3.2-3B-Instruct",
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

Measured sample output on Atlas A2 (910B):

```text
Prompt: 'Hello, my name is', Generated text: ' [Your Name] and I am a digital nomad, living in [Your Current Location]. ...'
Prompt: 'The future of AI is', Generated text: ' a topic of much debate. While some experts predict that AI will continue to advance at an exponential rate, others argue that its development should be slowed ...'
```

## Accuracy Evaluation

The GSM8K dataset is used to evaluate the reasoning capability of `Llama-3.2-3B-Instruct` via the Language Model Evaluation Harness (`lm_eval`).

1. For `lm_eval` installation, refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. The accuracy regression config `tests/e2e/models/configs/Llama-3.2-3B-Instruct.yaml` is already provided and can be run directly:

```bash
export VLLM_USE_MODELSCOPE=True
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
    --config ./tests/e2e/models/configs/Llama-3.2-3B-Instruct.yaml \
    --tp-size 1 \
    --report-dir ./benchmarks/accuracy
```

Evaluation setting:

- Dataset: `gsm8k`, split: `test` (1319 samples)
- Few-shot: `5-shot`
- `apply_chat_template`: `True`
- `fewshot_as_multiturn`: `True`

Measured accuracy on Atlas A2 (910B2C, 64G), with vLLM-Ascend v0.11.0 / CANN 8.3.RC2 / torch_npu 2.7.1:

| Category | Dataset | Metric | Threshold | Measured | Result |
|----------|---------|--------|-----------|----------|--------|
| Accuracy | gsm8k / test | exact_match,strict-match | 0.71 | 0.7316 | ✅ |
| Accuracy | gsm8k / test | exact_match,flexible-extract | 0.76 | 0.768 | ✅ |

> Decision rule: measured values are compared with thresholds using relative tolerance `RTOL=0.05`; both metrics pass within tolerance.

### Remarks on Metrics

- **exact_match,strict-match**: Only predictions that strictly match the expected final-answer extraction format are counted as correct.
- **exact_match,flexible-extract**: Predictions are evaluated with a more flexible answer extraction rule, tolerating minor formatting differences as long as the final numeric answer is correct.

## Performance

`Llama-3.2-3B-Instruct` can be deployed through `vllm-ascend` for online inference and benchmark evaluation. Actual throughput and latency depend on hardware resources, prompt length, output length, concurrency, and runtime configuration. Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for further benchmarking, covering request latency, throughput under concurrency, long-context inference, memory utilization, and stability under continuous serving workloads.
