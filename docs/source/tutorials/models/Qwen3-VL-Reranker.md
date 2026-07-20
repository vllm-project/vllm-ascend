# Qwen3-VL-Reranker

## 1 Introduction

The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities. This guide describes how to run the model with vLLM Ascend.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3-VL-Reranker-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B)
- `Qwen3-VL-Reranker-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`


## 4 Installation

### 4.1 Docker Image Installation
You can use our official docker image to run `Qwen3-VL-Reranker` model directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "Atlas A3 inference products"

    Start the docker image on your each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
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

=== "Atlas A2 inference products"

    Start the docker image on your each node.

    ```shell
      export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p
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

=== "Atlas A3 inference products"

    Start the docker image on your each node.

     ```shell
    #!/bin/sh
    vllm serve Qwen/Qwen3-VL-Reranker-2B \
        --runner pooling \
        --max-model-len 4096 \
        --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
        --chat-template ./qwen3_vl_reranker.jinja \
        --port 8000 \
        --max-model-len 10240
    ```

=== "Atlas A2 inference products"

    Start the docker image on your each node.

    ```shell
    #!/bin/sh
    vllm serve Qwen/Qwen3-VL-Reranker-2B \
        --runner pooling \
        --max-model-len 4096 \
        --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
        --chat-template ./qwen3_vl_reranker.jinja \
        --port 8000 \
        --max-model-len 10240
    ```

=== "Atlas inference products"

    Start the docker image on your each node.

    ```shell
    #!/bin/sh
    vllm serve Qwen/Qwen3-VL-Reranker-2B \
        --runner pooling \
        --max-model-len 4096 \
        --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
        --chat-template ./qwen3_vl_reranker.jinja \
        --compilation-config '{"cudagraph_capture_sizes": [1024,512]}' \
        --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
        --dtype float16 \
        --port 8000 \
        --max-model-len 10240
    ```

    The `--max-model-len` option is added to prevent errors when generating the attention operator mask on the Atlas inference products.


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

The service returns HTTP 200 OK with a JSON response containing the `embedding` field. Example output:

```json
{
    "id": "score-xxxxx",
    "model": "/home/data/Qwen3-VL-Reranker-2B",
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

For more usage examples, please check the [link](https://github.com/vllm-project/vllm/tree/main/examples/pooling/score)

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
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    
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

Run performance of `Qwen3-VL-Embedding-2B` as an example.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli/) for more details.

Take the `serve` as an example. Run the code as follows.

```bash
vllm bench serve --model Qwen/Qwen3-VL-Reranker-2B --backend vllm-rerank --dataset-name random-rerank --endpoint /v1/rerank --random-input 200  --save-result --result-dir ./
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

## 9 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
