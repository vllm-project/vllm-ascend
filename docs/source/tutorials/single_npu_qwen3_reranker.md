# Single NPU (Qwen3-Reranker-8B)

The Qwen3 Reranker model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings and reranking models in various sizes (0.6B, 4B, and 8B). This guide describes how to run the model with vLLM Ascend. Note that only 0.9.2rc1 and higher versions of vLLM Ascend support the model.

## Run Docker Container

Using the Qwen3-Reranker-8B model as an example, first run the docker container with the following command:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
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

Set up environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

### Online Inference

```bash
vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
```

Once your server is started, you can send request with follow examples.


### requests demo + formating query & document:
```python
import requests

url = "http://127.0.0.1:8000/score"
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

queries = [
    "What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

queries = [
    query_template.format(prefix=prefix, instruction=instruction, query=query)
    for query in queries
]
documents = [
    document_template.format(doc=doc, suffix=suffix) for doc in documents
]

response = requests.post(url,
                         json={
                             "model": MODEL_NAME,
                             "text_1": queries,
                             "text_2": documents,
                             "truncate_prompt_tokens": -1,
                         }).json()

print(response)
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
# model = LLM(model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", task="score")

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

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    queries = [
        query_template.format(prefix=prefix, instruction=instruction, query=query)
        for query in queries
    ]
    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

    outputs = model.score(queries, documents)

    print([output.outputs.score for output in outputs])
```

If you run this script successfully, you can see the info shown below:

```bash
{'id': 'score-14f698f021b9434482ec3d94a5757e11', 'object': 'list', 'created': 1749786173, 'model': 'tomaarsen/Qwen3-Reranker-0.6B-seq-cls', 'data': [{'index': 0, 'object': 'score', 'score': 0.99951171875}, {'index': 1, 'object': 'score', 'score': 0.99951171875}], 'usage': {'prompt_tokens': 189, 'total_tokens': 189, 'completion_tokens': 0, 'prompt_tokens_details': None}}
```

### reference by [this](https://github.com/vllm-project/vllm/pull/19260)