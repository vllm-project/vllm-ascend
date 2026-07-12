# Phi-4-mini-instruct

## Introduction

`Phi-4-mini-instruct` is a compact, instruction-tuned large language model developed by Microsoft. With approximately 3.8 billion parameters, it delivers strong reasoning and coding capabilities while being efficient enough for single-NPU deployment. The model uses the `Phi3ForCausalLM` architecture with LongRoPE scaling, enabling context lengths up to 131,072 tokens.

This document describes the main verification steps of the model on Ascend NPU, including supported features, environment preparation, single-node deployment, functional verification, accuracy evaluation on the GSM8K benchmark, and performance results.

## Environment Preparation

### Model Weight

`Phi-4-mini-instruct` (BF16 version): requires 1 Ascend 910B (with 1 x 64G NPUs). [Download model weight](https://huggingface.co/microsoft/Phi-4-mini-instruct) or [ModelScope mirror](https://modelscope.cn/models/AI-ModelScope/Phi-4-mini-instruct)

It is recommended to place the model weight in a shared cache directory, such as `/root/.cache/` or a local model path like `/data/models/Phi-4-mini-instruct`.

### Installation

`Phi-4-mini-instruct` can be deployed with `vllm-ascend` in a compatible runtime environment.

You can use the official docker image for deployment:

```bash
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
  -v /data/models:/data/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

If you do not want to use the docker image, you can also build from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

### Online Serving

Start the online serving service with the following command:

```bash
vllm serve "microsoft/Phi-4-mini-instruct" \
  --served-model-name phi-4-mini-instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --port 8000
```

```{note}
`--enforce-eager` is recommended for NPU deployment stability. The model uses the built-in `Phi3ForCausalLM` architecture in vLLM, so `--trust-remote-code` is not required.
```

### Offline Inference

You can also use the Python API for offline inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="microsoft/Phi-4-mini-instruct",
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)

sampling_params = SamplingParams(max_tokens=256, temperature=0.7)
outputs = llm.chat(
    [[{"role": "user", "content": "What is the capital of France?"}]],
    sampling_params,
)
for output in outputs:
    print(output.outputs[0].text)
```

## Functional Verification

Once your server is started, you can query the model with a simple prompt:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4-mini-instruct",
    "prompt": "Question: If a train travels 60 miles in 2 hours, what is its average speed in miles per hour?\nAnswer:",
    "max_tokens": 64,
    "temperature": 1.0
  }'
```

You can also test with the chat endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4-mini-instruct",
    "messages": [
      {"role": "user", "content": "Write a Python function to compute fibonacci numbers."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

A valid response indicates that the model is deployed correctly and can generate text outputs.

## Accuracy Evaluation

The GSM8K dataset was used to evaluate the reasoning capability of `Phi-4-mini-instruct`.

The current evaluation setting is:

- Dataset: `gsm8k`
- Split: `test`
- Number of samples: `1319` (full test set)
- Few-shot setting: `5-shot`
- `apply_chat_template`: `True`
- `fewshot_as_multiturn`: `True`

The current evaluation results are:

| Category | Dataset      | Metric                       | Result |
| -------- | ------------ | ---------------------------- | ------ |
| Accuracy | gsm8k / test | Total Samples                | 1319   |
| Accuracy | gsm8k / test | exact_match,strict-match     | 0.810  |
| Accuracy | gsm8k / test | exact_match,flexible-extract | 0.812  |

### Evaluation Command

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=microsoft/Phi-4-mini-instruct,tensor_parallel_size=1,dtype=auto,trust_remote_code=False,max_model_len=4096,gpu_memory_utilization=0.8,enforce_eager=True" \
  --tasks gsm8k \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 5 \
  --batch_size auto
```

### Remarks on Metrics

- **exact_match,strict-match**: Only predictions that strictly match the expected final-answer extraction format are counted as correct.
- **exact_match,flexible-extract**: Predictions are evaluated with a more flexible answer extraction rule, which tolerates minor formatting differences as long as the final numeric answer is correct.

## Performance

### Baseline Result

`Phi-4-mini-instruct` can be deployed through `vllm-ascend` for online inference and benchmark evaluation.

The following benchmark was obtained with `vllm bench serve` on a single Ascend 910B NPU:

- `max_model_len`: 4096
- `tensor_parallel_size`: 1
- `enforce_eager`: true
- Dataset: ShareGPT
- Number of prompts: 50

| Metric                               | Value   |
| ------------------------------------ | ------- |
| Successful requests                  | 50      |
| Failed requests                      | 0       |
| Benchmark duration (s)               | 26.50   |
| Request throughput (req/s)           | 1.89    |
| Output token throughput (tok/s)      | 382.81  |
| Peak output token throughput (tok/s) | 1190.00 |
| Mean TTFT (ms)                       | 1319.22 |
| Median TTFT (ms)                     | 1230.32 |
| Mean TPOT (ms)                       | 51.06   |
| Median TPOT (ms)                     | 37.49   |
| Mean ITL (ms)                        | 36.80   |
| Median ITL (ms)                      | 33.90   |

```{note}
This benchmark was run in eager mode (`--enforce-eager`). Enabling graph mode may improve throughput. Actual performance depends on hardware, prompt/output length, and concurrency.
```

### Remarks

This document focuses on functional verification and benchmark accuracy on GSM8K.
Further benchmarking is recommended for:

- request latency
- throughput under concurrency
- long-context inference (up to 131K tokens with LongRoPE)
- memory utilization
- stability under continuous serving workloads
