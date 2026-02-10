# Qwen3-VL-Embedding

## Introduction

The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities. This guide describes how to run the model with vLLM Ascend.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## Environment Preparation

### Model Weight

- `Qwen3-VL-Embedding-8B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-8B)
- `Qwen3-VL-Embedding-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

You can use our official docker image to run `Qwen3-VL-Embedding` series models.

- Start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

Using the Qwen3-VL-Embedding-8B model as an example, first run the docker container with the following command:

### Online Inference

```bash
vllm serve Qwen/Qwen3-VL-Embedding-8B --runner pooling
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://127.0.0.1:8000/v1/embeddings -H "Content-Type: application/json" -d '{
  "input": [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
}'
```

### Offline Inference

```python
import torch
from vllm import LLM

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


if __name__=="__main__":
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents

    model = LLM(model="Qwen/Qwen3-VL-Embedding-8B",
                runner="pooling",
                distributed_executor_backend="mp")

    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
```

If you run this script successfully, you can see the info shown below:

```bash
Adding requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 192.47it/s]
Processed prompts:   0%|                                            | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](EngineCore_DP0 pid=2425173) (Worker pid=2425180) INFO 01-09 00:44:40 [acl_graph.py:194] Replaying aclgraph
(EngineCore_DP0 pid=2425173) (Worker pid=2425180) ('Warning: torch.save with "_use_new_zipfile_serialization = False" is not recommended for npu tensor, which may bring unexpected errors and hopefully set "_use_new_zipfile_serialization = True"', 'if it is necessary to use this, please convert the npu tensor to cpu tensor for saving')
Processed prompts: 100%|████████████████████████████████████| 4/4 [00:00<00:00, 21.34it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
[[0.9279120564460754, 0.32747742533683777], [0.4124627113342285, 0.7425257563591003]]
```

For more examples, refer to the vLLM official examples:

- [Offline Vision Embedding Example](https://github.com/vllm-project/vllm/blob/main/examples/pooling/embed/vision_embedding_offline.py)
- [Online Vision Embedding Example](https://github.com/vllm-project/vllm/blob/main/examples/pooling/embed/vision_embedding_online.py)

## Performance

Run performance of `Qwen3-VL-Embedding-8B` as an example.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli/) for more details.

Take the `serve` as an example. Run the code as follows.

```bash
vllm bench serve --model Qwen/Qwen3-VL-Embedding-8B --backend openai-embeddings --dataset-name random --endpoint /v1/embeddings --random-input 200 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result. With this tutorial, the performance result is:

```bash
============ Serving Benchmark Result ============
Successful requests:                     1000
Failed requests:                         0
Benchmark duration (s):                  19.53
Total input tokens:                      200000
Request throughput (req/s):              51.20
Total token throughput (tok/s):          10240.42
----------------End-to-end Latency----------------
Mean E2EL (ms):                          10360.53
Median E2EL (ms):                        10354.37
P99 E2EL (ms):                           19423.21
==================================================
```
