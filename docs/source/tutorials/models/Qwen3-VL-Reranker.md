# Qwen3-VL-Reranker

## 1 Introduction

The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities. This guide describes how to run the model with vLLM Ascend.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3-VL-Reranker-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-8B)
- `Qwen3-VL-Reranker-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `Qwen3-VL-Reranker` model directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Start the docker image on your each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "A2 series"

    Start the docker image on your each node.

    ```shell
      export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "Atlas inference products"

    Start the docker image on your each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment

### 5.1 Chat Template

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

=== "A3/A2 series"

    ```shell
    #!/bin/sh
    vllm serve Qwen/Qwen3-VL-Reranker-2B \
        --served-model-name Qwen/Qwen3-VL-Reranker-2B \
        --runner pooling \
        --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
        --chat-template ./qwen3_vl_reranker.jinja \
        --port 8000 \
        --max-model-len 1024
    ```

=== "Atlas inference products"

    Start the docker image on your each node.

    ```shell
    #!/bin/sh
    vllm serve Qwen/Qwen3-VL-Reranker-2B \
        --served-model-name Qwen/Qwen3-VL-Reranker-2B \
        --runner pooling \
        --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
        --chat-template ./qwen3_vl_reranker.jinja \
        --compilation-config '{"cudagraph_capture_sizes": [1024,512]}' \
        --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
        --dtype float16 \
        --port 8000 \
        --max-model-len 1024
    ```
    Required  Parameter Descriptions:

    `--compilation-config` For Atlas inference products, due to limited hardware streams, the size of cudagraph_capture_sizes is restricted.

Key Parameter Descriptions:

- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request. For Atlas inference products if automatic parsing resolves to a large context length, allocating this mask (O(max_model_len^2)) may exceed NPU memory and trigger OOM. Be sure to set an explicit and conservative value, such as --max-model-len 1024.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

## 6 Functional Verification

Once your server is started, you can verify by follow command:

Service Verification:

```bash
curl  http://localhost:8000/v1/rerank \
    -X POST \
    -d '{"query":"What is the capital of China?", "documents": ["The capital of China is Beijing.", "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."]}' \
    -H 'Content-Type: application/json'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `relevance_score` field. Example output:

```json
{
    "id": "score-xxxxx",
    "model": "Qwen/Qwen3-VL-Reranker-2B",
    "usage": {
        "prompt_tokens": 179,
        "total_tokens": 179
    },
    "results": [
        {
            "index": 0,
            "document": {
                "text": "The capital of China is Beijing.",
                "multi_modal": null
            },
            "relevance_score": 0.7209711670875549
        },
        {
            "index": 1,
            "document": {
                "text": "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
                "multi_modal": null
            },
            "relevance_score": 0.18871910870075226
        }
    ]
}
```

For more usage examples, please reference the [examples](https://github.com/vllm-project/vllm/tree/main/examples/pooling/score)

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using MTEB

1. Refer to [MTEB](https://docs.mteb.org/) for details.

2. Run follow code to execute the accuracy evaluation.

    ```python
  
    import os
    
    from mteb.models.vllm_wrapper import VllmCrossEncoderWrapper
    
    if __name__ == "__main__":
        import mteb
    
        data_path = "/home/data/mteb_data"
        os.environ["HF_DATASETS_CACHE"] = data_path
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
        model = VllmCrossEncoderWrapper(f"/home/data/Qwen3-VL-Reranker-2B",
                                    revision="norm",
                                    dtype="float16",
                                    enforce_eager=True,
                                    max_model_len=10240,
                                    hf_overrides={"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": True})
    
        cache = mteb.ResultCache("/home/data/mteb_data")
        tasks = mteb.get_tasks(
            task_types=["Reranking"],
            languages=["zho"]
        )
        tasks = mteb.get_tasks(tasks=["MultiLongDocReranking"])
        results = mteb.evaluate(model, tasks=tasks, cache=cache, overwrite_strategy="always")
        print(results)

    ```

3. After execution, you can get the result.

## 8 Performance Evaluation

### Using vLLM Benchmark

Run performance of `Qwen3-VL-Reranker-2B` as an example.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli/) for more details.

Take the `serve` as an example. Run the code as follows.

```bash
vllm bench serve --model Qwen/Qwen3-VL-Reranker-2B --backend vllm-rerank --port 8000 --dataset-name random-rerank --endpoint /v1/rerank --random-input 200  --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
