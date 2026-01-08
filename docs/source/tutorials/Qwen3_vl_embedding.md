# Qwen3-Embedding

## Introduction
The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities. This guide describes how to run the model with vLLM Ascend.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## Environment Preparation

### Model Weight

- `Qwen3-VL-Embedding-8B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-8B)
- `Qwen3-VL-Embedding-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`
### Installation
You can use our official docker image to run `Qwen3-VL-Embedding` series models.
- Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

if you don't want to use the docker image as above, you can also build all from source:
- Install `vllm-ascend` from source, refer to [installation](../installation.md).

## Deployment

Using the Qwen3-VL-Embedding-8B model as an example, first run the docker container with the following command:

### Online Inference

```bash
vllm serve Qwen/Qwen3-VL-Embedding-8B --runner pooling --host 127.0.0.1 --port 8888
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://127.0.0.1:8888/v1/embeddings -H "Content-Type: application/json" -d '{
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
    return f'Instruct: {task_description}\nQuery:{query}'


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

    model = LLM(model="Qwen/Qwen3-VL-Embedding-2B",
                runner="pooling",
                distributed_executor_backend="mp")

    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
```

If you run this script successfully, you can see the info shown below:

```bash
INFO 01-08 16:02:30 [llm.py:344] Supported tasks: ['token_embed', 'embed']
Adding requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 202.21it/s]
Processed prompts:   0%|                                            | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](EngineCore_DP0 pid=2271981) (Worker pid=2271988) INFO 01-08 16:02:30 [acl_graph.py:194] Replaying aclgraph
(EngineCore_DP0 pid=2271981) (Worker pid=2271988) ('Warning: torch.save with "_use_new_zipfile_serialization = False" is not recommended for npu tensor, which may bring unexpected errors and hopefully set "_use_new_zipfile_serialization = True"', 'if it is necessary to use this, please convert the npu tensor to cpu tensor for saving')
Processed prompts: 100%|████████████████████████████████████| 4/4 [00:00<00:00, 25.07it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
[[0.9131501913070679, 0.28830885887145996], [0.2932716906070709, 0.7744175791740417]]
```

## Performance

Run performance of `Qwen3-VL-Reranker-8B` as an example.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/) for more details.

Take the `serve` as an example. Run the code as follows.

```bash
vllm bench serve --model Qwen3-Embedding-8B --backend openai-embeddings --dataset-name random --host 127.0.0.1 --port 8888 --endpoint /v1/embeddings --tokenizer /root/.cache/Qwen3-Embedding-8B --random-input 200 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result. With this tutorial, the performance result is:

```bash
TODO
```
