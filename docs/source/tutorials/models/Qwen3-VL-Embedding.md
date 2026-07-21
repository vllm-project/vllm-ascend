# Qwen3-VL-Embedding

## 1 Introduction

The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities. This guide describes how to run the model with vLLM Ascend.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3-VL-Embedding-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-8B)
- `Qwen3-VL-Embedding-2B` [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `Qwen3-VL-Embedding` model directly.

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

=== "A3/A2 series"

    Start the docker image on your each node.

    ```shell
    #!/bin/sh
    vllm serve Qwen/Qwen3-VL-Embedding-2B  \
      --served-model-name Qwen/Qwen3-VL-Embedding-2B  \
      --runner pooling \
      --port 8000 \
      --max-model-len 1024
    ```

=== "Atlas inference products"

    Start the docker image on your each node.

    ```shell
    #!/bin/sh
    vllm serve Qwen/Qwen3-VL-Embedding-2B  \
      --served-model-name Qwen/Qwen3-VL-Embedding-2B  \
      --compilation-config '{"cudagraph_capture_sizes": [1024,512]}' \
      --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
      --runner pooling \
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
curl -X POST http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
  "input": [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
}'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `embedding` field. Example output:

```json
{
  "id": "embd-8136155c01e8411d",
  "object": "list",
  "created": 1784538286,
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [
        -0.028474265709519386,
        -0.02678542211651802
      ]
    },
    {
      "index": 1,
      "object": "embedding",
      "embedding": [
        -0.016785264015197754,
        -0.003787524998188019
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 39,
    "total_tokens": 39,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

For more usage examples, please reference the [examples](https://github.com/vllm-project/vllm/tree/main/examples/pooling/embed)

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using MTEB

1. Refer to [MTEB](https://docs.mteb.org/) for details.

2. Run follow code to execute the accuracy evaluation.

    ```python
  
    import os
    import mteb
    
    from mteb.models.vllm_wrapper import VllmEncoderWrapper
    
    if __name__ == "__main__":
    
        data_path = "/home/data/mteb_data"
        os.environ["HF_DATASETS_CACHE"] = data_path
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
        model = VllmEncoderWrapper(f"/root/.cache/Qwen3-VL-Embedding-2B",
                                    revision="norm",
                                    dtype="float16",
                                    max_model_len=10240,
                                   )
    
        cache = mteb.ResultCache("/home/data/mteb_data")
        tasks = mteb.get_tasks(tasks=["LeCaRDv2"])
        results = mteb.evaluate(model, tasks=tasks, cache=cache, encode_kwargs={"batch_size": 2}, overwrite_strategy="always")
        df = results.to_dataframe()
        print(df)
    ```

3. After execution, you can get the result.

## 8 Performance Evaluation

### Using vLLM Benchmark

Run performance of `Qwen3-VL-Embedding-2B` as an example.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli/) for more details.

Take the `serve` as an example. Run the code as follows.

```bash
vllm bench serve --model Qwen/Qwen3-VL-Embedding-2B --backend openai-embeddings --port 8000 --dataset-name random --endpoint /v1/embeddings --random-input 200 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
