# e5-mistral-7b-instruct

## Introduction

`e5-mistral-7b-instruct` is a text embedding model released by `intfloat` and built on `Mistral-7B`.
It is designed for retrieval tasks where queries are formatted with a task instruction and documents are
embedded as plain text. This guide describes how to run online and offline embedding inference with
vLLM Ascend.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) for the full model support
matrix. In vLLM Ascend, this model uses the pooling runner with `embed` conversion. The SentenceTransformers
checkpoint provides last-token pooling and normalized 4096-dimensional embeddings.

## Environment Preparation

### Model Weight

- `e5-mistral-7b-instruct` FP16 checkpoint:
  [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)

The model can run on a single 64 GB Ascend NPU for functional validation. Download the model weights to
a shared cache or local path, for example `/data/models/e5-mistral-7b-instruct`.

### Installation

You can use the official docker image or build `vllm-ascend` from source. For docker setup, refer to
[using docker](../../installation.md#set-up-using-docker). For source installation, refer to
[installation](../../installation.md).

## Deployment

### Online Inference

Start an OpenAI-compatible embeddings service:

```bash
vllm serve intfloat/e5-mistral-7b-instruct \
  --runner pooling \
  --served-model-name e5-mistral-7b-instruct \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
```

For a local checkpoint, replace `intfloat/e5-mistral-7b-instruct` with the local model path.

## Functional Verification

The query side should include the retrieval instruction. Documents do not need the instruction prefix.

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "e5-mistral-7b-instruct",
    "input": [
      "Instruct: Retrieve passages that answer the query\nQuery: What is the capital of China?",
      "The capital of China is Beijing."
    ]
  }'
```

A valid response returns one 4096-dimensional embedding for each input.

### Offline Inference

```python
import torch
from vllm import LLM


def format_query(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


model = "intfloat/e5-mistral-7b-instruct"
task = "Given a web search query, retrieve relevant passages that answer the query"

queries = [
    format_query(task, "What is the capital of China?"),
    format_query(task, "Explain gravity"),
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other and gives weight to physical objects.",
]

llm = LLM(
    model=model,
    runner="pooling",
    max_model_len=512,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)

outputs = llm.embed(queries + documents)
embeddings = torch.tensor([output.outputs.embedding for output in outputs])
scores = embeddings[:2] @ embeddings[2:].T
print(scores.tolist())
```

The relevant document should receive the highest score for each query.

## Accuracy Evaluation

The model was verified with a retrieval smoke test that checks embedding dimension, L2 normalization,
and top-1 ranking on two query-document pairs.

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Functional | e5_mistral_retrieval_smoke | embedding_dimension | 4096 |
| Functional | e5_mistral_retrieval_smoke | mean_l2_norm | 1.0 |
| Functional | e5_mistral_retrieval_smoke | top1_accuracy | 1.0 |
| Functional | e5_mistral_retrieval_smoke | min_score_margin | 0.273 |

Run the e2e test with a local checkpoint:

```bash
export E5_MISTRAL_MODEL_PATH=/data/models/e5-mistral-7b-instruct
python -m pytest tests/e2e/models/test_embedding_eval_correctness.py \
  --config tests/e2e/models/configs/e5-mistral-7b-instruct.yaml
```

## Performance

Use the OpenAI embeddings benchmark after starting the service:

```bash
vllm bench serve \
  --model e5-mistral-7b-instruct \
  --backend openai-embeddings \
  --dataset-name random \
  --endpoint /v1/embeddings \
  --host 127.0.0.1 \
  --port 8000 \
  --random-input 200 \
  --save-result \
  --result-dir ./
```

Throughput and latency depend on prompt length, batch size, `max_model_len`, and available NPU memory.
