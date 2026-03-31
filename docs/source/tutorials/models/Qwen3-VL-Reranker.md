# Qwen3-VL-Reranker

## Introduction

The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities. This guide describes how to run the model with vLLM Ascend.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## Environment Preparation

### Model Weight

- `Qwen3-VL-Reranker-8B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-8B)
- `Qwen3-VL-Reranker-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

You can use our official docker image to run `Qwen3-VL-Reranker` series models.

- Start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

Using the Qwen3-VL-Reranker-8B model as an example:

### Chat Template

The Qwen3-VL-Reranker model requires a specific chat template for proper formatting. Create a file named `qwen3_vl_reranker.jinja` with the following content:

```jinja
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{
    messages
    | selectattr("role", "eq", "system")
    | map(attribute="content")
    | first
    | default("Given a search query, retrieve relevant candidates that answer the query.")
}}<Query>:{{
    messages
    | selectattr("role", "eq", "query")
    | map(attribute="content")
    | first
}}
<Document>:{{
    messages
    | selectattr("role", "eq", "document")
    | map(attribute="content")
    | first
}}<|im_end|>
<|im_start|>assistant

```

Save this file to a location of your choice (e.g., `./qwen3_vl_reranker.jinja`).

### Online Inference

Start the server with the following command:

```bash
vllm serve Qwen/Qwen3-VL-Reranker-8B \
    --runner pooling \
    --max-model-len 4096 \
    --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
    --chat-template ./qwen3_vl_reranker.jinja
```

Once your server is started, you can send request with follow examples.

```python
import requests

url = "http://127.0.0.1:8000/v1/rerank"

# Please use the query_template and document_template to format the query and
# document for better reranker results.

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

instruction = (
    "Given a search query, retrieve relevant candidates that answer the query."
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
{'id': 'rerank-ac3495afa8e12404', 'model': 'Qwen/Qwen3-VL-Reranker-8B', 'usage': {'prompt_tokens': 315, 'total_tokens': 315}, 'results': [{'index': 0, 'document': {'text': '<Document>: The capital of China is Beijing.<|im_end|>\n<|im_start|>assistant\n', 'multi_modal': None}, 'relevance_score': 0.6368980407714844}, {'index': 1, 'document': {'text': '<Document>: Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.<|im_end|>\n<|im_start|>assistant\n', 'multi_modal': None}, 'relevance_score': 0.20816077291965485}]}
```

### Offline Inference

```python
from vllm import LLM

model_name = "Qwen/Qwen3-VL-Reranker-8B"

# What is the difference between the official original version and one
# that has been converted into a sequence classification model?
# Qwen3-Reranker is a language model that doing reranker by using the
# logits of "no" and "yes" tokens.
# It needs to computing 151669 tokens logits, making this method extremely
# inefficient, not to mention incompatible with the vllm score API.
# A method for converting the original model into a sequence classification
# model was proposed. See: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
# Models converted offline using this method can not only be more efficient
# and support the vllm score API, but also make the init parameters more
# concise, for example.
# model = LLM(model="Qwen/Qwen3-VL-Reranker-8B", runner="pooling")

# If you want to load the official original version, the init parameters are
# as follows.

model = LLM(
    model=model_name,
    runner="pooling",
    hf_overrides={
        # Manually route to sequence classification architecture
        # This tells vLLM to use Qwen3VLForSequenceClassification instead of
        # the default Qwen3VLForConditionalGeneration
        "architectures": ["Qwen3VLForSequenceClassification"],
        # Specify which token logits to extract from the language model head
        # The original reranker uses "no" and "yes" token logits for scoring
        "classifier_from_token": ["no", "yes"],
        # Enable special handling for original Qwen3-Reranker models
        # This flag triggers conversion logic that transforms the two token
        # vectors into a single classification vector
        "is_original_qwen3_reranker": True,
    },
)

# Why do we need hf_overrides for the official original version:
# vllm converts it to Qwen3VLForSequenceClassification when loaded for
# better performance.
# - Firstly, we need using `"architectures": ["Qwen3VLForSequenceClassification"],`
# to manually route to Qwen3VLForSequenceClassification.
# - Then, we will extract the vector corresponding to classifier_from_token
# from lm_head using `"classifier_from_token": ["no", "yes"]`.
# - Third, we will convert these two vectors into one vector.  The use of
# conversion logic is controlled by `using "is_original_qwen3_reranker": True`.

# Please use the query_template and document_template to format the query and
# document for better reranker results.

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

if __name__ == "__main__":
    instruction = (
        "Given a search query, retrieve relevant candidates that answer the query."
    )

    query = "What is the capital of China?"

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

    outputs = model.score(query_template.format(prefix=prefix, instruction=instruction, query=query), documents)

    print([output.outputs.score for output in outputs])
```

If you run this script successfully, you will see a list of scores printed to the console, similar to this:

```bash
Adding requests: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 2409.83it/s]
Processed prompts:   0%|                                            | 0/2 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](EngineCore_DP0 pid=682882) INFO 01-20 04:38:46 [acl_graph.py:188] Replaying aclgraph
Processed prompts: 100%|████████████████████████████████████| 2/2 [00:00<00:00,  9.44it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
[0.7235596776008606, 0.0002742875076364726]
```

For more examples, refer to the vLLM official examples:

- [Offline Vision Embedding Example](https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/vision_reranker_offline.py)
- [Online Vision Embedding Example](https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/vision_reranker_online.py)

## Performance

Run performance of `Qwen3-VL-Reranker-8B` as an example.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli/) for more details.

Take the `serve` as an example. Run the code as follows.

```bash
vllm bench serve --model Qwen/Qwen3-VL-Reranker-8B --backend vllm-rerank --dataset-name random-rerank --endpoint /v1/rerank --random-input 200  --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result. With this tutorial, the performance result is:

```bash
============ Serving Benchmark Result ============
Successful requests:                     1000
Failed requests:                         0
Benchmark duration (s):                  13.70
Total input tokens:                      265122
Request throughput (req/s):              72.99
Total token throughput (tok/s):          19351.23
----------------End-to-end Latency----------------
Mean E2EL (ms):                          7474.64
Median E2EL (ms):                        7528.72
P99 E2EL (ms):                           13523.32
==================================================
```
