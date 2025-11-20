# Qwen3-Embedding

## Introduction
The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings and reranking models in various sizes (0.6B, 4B, and 8B). This guide describes how to run the model with vLLM Ascend. Note that only 0.9.2rc1 and higher versions of vLLM Ascend support the model.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## Environment Preparation

### Model Weight

- `Qwen3-Embedding-8B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B)
- `Qwen3-Embedding-4B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B)
- `Qwen3-Embedding-0.6B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`
### Installation
You can use our official docker image to run `Qwen3-Embedding` series models.
- Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).


if you don't want to use the docker image as above, you can also build all from source:
- Install `vllm-ascend` from source, refer to [installation](../installation.md).

## Deployment
*only support single node single npu deployment*

Using the Qwen3-Embedding-8B model as an example, first run the docker container with the following command:

### Online Inference

```bash
vllm serve Qwen/Qwen3-Embedding-8B --task embed
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://<ip>:<port>/v1/embeddings -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-Embedding-8B",
  "input": [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
}'
```

### Offline Inference

```python
import torch
import vllm
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

    model = LLM(model="Qwen/Qwen3-Embedding-8B",
                task="embed",
                distributed_executor_backend="mp")

    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
```

If you run this script successfully, you can see the info shown below:

```bash
Adding requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 282.22it/s]
Processed prompts:   0%|                                                                                                                                    | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](VllmWorker rank=0 pid=4074750) ('Warning: torch.save with "_use_new_zipfile_serialization = False" is not recommended for npu tensor, which may bring unexpected errors and hopefully set "_use_new_zipfile_serialization = True"', 'if it is necessary to use this, please convert the npu tensor to cpu tensor for saving')
Processed prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 31.95it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
[[0.7477798461914062, 0.07548339664936066], [0.0886271521449089, 0.6311039924621582]]
```
## Accuracy Evaluation
will be provided later...
## Performance
will be provided later...