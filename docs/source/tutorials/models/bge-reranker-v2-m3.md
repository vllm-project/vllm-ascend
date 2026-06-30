# BGE-Reranker-V2-M3

## Introduction

BGE-Reranker-V2-M3 is a cross-encoder reranker model developed by BAAI, based on the RoBERTa architecture. Unlike decoder-only LLM-based rerankers, it is a lightweight (278M parameters) encoder-only model that directly computes relevance scores for (query, document) pairs. It supports over 100 languages and is widely used in RAG (Retrieval-Augmented Generation) pipelines. This guide describes how to run the model with vLLM Ascend.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## Environment Preparation

### Model Weight

- `BAAI/bge-reranker-v2-m3` [Download model weight](https://huggingface.co/BAAI/bge-reranker-v2-m3)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

You can use our official docker image to run `BAAI/bge-reranker-v2-m3`.

- Start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

Using the BAAI/bge-reranker-v2-m3 model as an example, first run the docker container with the following command:

### Online Inference

Unlike LLM-based rerankers, bge-reranker-v2-m3 is a standard cross-encoder model that does not require `hf_overrides` or special prompt templates. It can be served directly:

```bash
vllm serve BAAI/bge-reranker-v2-m3 --host 127.0.0.1 --port 8888
```

Once your server is started, you can send requests with the following examples.

### requests demo

```python
import requests

url = "http://127.0.0.1:8888/v1/rerank"

query = "What is the capital of China?"

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other.",
]

response = requests.post(url,
                         json={
                             "query": query,
                             "documents": documents,
                         }).json()

print(response)
```

If you run this script successfully, you will see a list of scores printed to the console, similar to this:

```bash
{'id': 'rerank-xxxxxxxx', 'model': 'BAAI/bge-reranker-v2-m3', 'results': [{'index': 0, 'relevance_score': 0.99}, {'index': 1, 'relevance_score': 0.01}]}
```

### Offline Inference

bge-reranker-v2-m3 is a standard cross-encoder and works with vLLM's `score()` API out of the box. No `hf_overrides` or architecture patches are needed:

```python
from vllm import LLM

model = LLM(
    model="BAAI/bge-reranker-v2-m3",
    runner="pooling",
)

if __name__ == "__main__":
    query = "What is the capital of China?"

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]

    # 1-to-N scoring: one query against multiple documents
    outputs = model.score(query, documents)

    print([output.outputs.score for output in outputs])

    # N-to-N scoring: pairwise scoring
    queries = [
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]
    docs = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
    ]
    outputs = model.score(queries, docs)

    print([output.outputs.score for output in outputs])
```

If you run this script successfully, you will see scores printed to the console, similar to:

```bash
[0.9995, 1.2e-06]
[0.9998, 0.9997]
```

## Performance

Performance data for `BAAI/bge-reranker-v2-m3` on Atlas A2 Series.

Take the `serve` as an example. Run:

```bash
vllm bench serve --model BAAI/bge-reranker-v2-m3 --backend vllm-rerank --dataset-name random-rerank --host 127.0.0.1 --port 8888 --endpoint /v1/rerank --tokenizer /root/.cache/bge-reranker-v2-m3 --random-input 200 --save-result --result-dir ./
```

```bash
============ Serving Benchmark Result ============
Successful requests:                     TBD
Failed requests:                         TBD
Benchmark duration (s):                  TBD
Request throughput (req/s):              TBD
==================================================
```
