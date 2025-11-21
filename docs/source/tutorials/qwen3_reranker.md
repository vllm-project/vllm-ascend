# Qwen3-Reranker

## Introduction
The Qwen3 Reranker model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings and reranking models in various sizes (0.6B, 4B, and 8B). This guide describes how to run the model with vLLM Ascend. Note that only 0.9.2rc1 and higher versions of vLLM Ascend support the model.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## Environment Preparation

### Model Weight

- `Qwen3-Reranker-8B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-8B)
- `Qwen3-Reranker-4B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-4B)
- `Qwen3-Reranker-0.6B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation
You can use our official docker image to run `Qwen3-Reranker` series models.
- Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

if you don't want to use the docker image as above, you can also build all from source:
- Install `vllm-ascend` from source, refer to [installation](../installation.md).

## Deployment
*only support single node single npu deployment*

Using the Qwen3-Reranker-8B model as an example, first run the docker container with the following command:

### Online Inference

```bash
vllm serve Qwen/Qwen3-Reranker-8B --task score --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
```

Once your server is started, you can send request with follow examples.

### requests demo + formatting query & document

```python
import requests

url = "http://<ip>:<port>/v1/rerank"
MODEL_NAME = "Qwen/Qwen3-Reranker-8B"

# Please use the query_template and document_template to format the query and
# document for better reranker results.

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

instruction = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

query = "What is the capital of China?"

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

documents = [
    document_template.format(doc=doc, suffix=suffix) for doc in documents
]

response = requests.post(url,
                         json={
                             "query": query_template.format(prefix=prefix, instruction=instruction, query=query),
                             "documents": documents,
                         }).json()

print(response)
```

If you run this script successfully, you will see a list of scores printed to the console, similar to this:

```bash
{'id': 'rerank-e856a17c954047a3a40f73d5ec43dbc6', 'model': 'Qwen/Qwen3-Reranker-8B', 'usage': {'total_tokens': 193}, 'results': [{'index': 0, 'document': {'text': '<Document>: The capital of China is Beijing.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', 'multi_modal': None}, 'relevance_score': 0.9944348335266113}, {'index': 1, 'document': {'text': '<Document>: Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', 'multi_modal': None}, 'relevance_score': 6.700084327349032e-07}]}
```

### Offline Inference

```python
from vllm import LLM

model_name = "Qwen/Qwen3-Reranker-8B"

# What is the difference between the official original version and one
# that has been converted into a sequence classification model?
# Qwen3-Reranker is a language model that doing reranker by using the
# logits of "no" and "yes" tokens.
# It needs to computing 151669 tokens logits, making this method extremely
# inefficient, not to mention incompatible with the vllm score API.
# A method for converting the original model into a sequence classification
# model was proposed. See：https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
# Models converted offline using this method can not only be more efficient
# and support the vllm score API, but also make the init parameters more
# concise, for example.
# model = LLM(model="Qwen/Qwen3-Reranker-8B", task="score")

# If you want to load the official original version, the init parameters are
# as follows.

model = LLM(
    model=model_name,
    task="score",
    hf_overrides={
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    },
)

# Why do we need hf_overrides for the official original version:
# vllm converts it to Qwen3ForSequenceClassification when loaded for
# better performance.
# - Firstly, we need using `"architectures": ["Qwen3ForSequenceClassification"],`
# to manually route to Qwen3ForSequenceClassification.
# - Then, we will extract the vector corresponding to classifier_from_token
# from lm_head using `"classifier_from_token": ["no", "yes"]`.
# - Third, we will convert these two vectors into one vector.  The use of
# conversion logic is controlled by `using "is_original_qwen3_reranker": True`.

# Please use the query_template and document_template to format the query and
# document for better reranker results.

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

if __name__ == "__main__":
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    query = "What is the capital of China?"

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

    outputs = model.score(query_template.format(prefix=prefix, instruction=instruction, query=query), documents)

    print([output.outputs[0].score for output in outputs])
```

If you run this script successfully, you will see a list of scores printed to the console, similar to this:

```bash
[0.9943699240684509, 6.876250040477316e-07]
```

## Accuracy Evaluation
will be provided later...

## Performance
will be provided later...
