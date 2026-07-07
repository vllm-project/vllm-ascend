# PaddleOCR-VL

## 1 Introduction

PaddleOCR-VL is a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition.

This document provides a detailed workflow for the complete deployment and verification of the model, including supported features, environment preparation, single-node deployment, and functional verification. It is designed to help users quickly complete model deployment and verification.

This document is validated and written based on **vLLM-Ascend v0.21.0rc1**. The current model (PaddleOCR-VL) is supported in this version. It is recommended to use this version or another updated official version for deployment.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `PaddleOCR-VL-0.9B`: [PaddleOCR-VL-0.9B](https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL)

It is recommended to download the model weights to the cache directory and set `VLLM_USE_MODELSCOPE=True` to load the model automatically. If you have downloaded the weights to a local directory, update the `MODEL_PATH` variable in the deployment script accordingly.

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `PaddleOCR-VL` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A2 series"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
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

=== "Atlas 300 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
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

!!! note

    If you are using Atlas 300 inference products, you may need to uninstall `triton-ascend` to avoid dependency conflicts.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

PaddleOCR-VL supports single-node single-card deployment on the A2 series and Atlas 300 inference products platform. Single-node deployment completes both Prefill and Decode within the same node.

Follow these steps to start the inference service:

1. Prepare model weights: Ensure the model weights are accessible. With `VLLM_USE_MODELSCOPE=True`, the model will be loaded automatically from ModelScope.
2. Set the `MODEL_PATH` environment variable to point to your model directory.
3. Create and execute the deployment script (save as `deploy.sh`).

Startup Command:

=== "A2 series"

    ```bash
    #!/bin/sh
    export VLLM_USE_MODELSCOPE=True
    export MODEL_PATH="PaddlePaddle/PaddleOCR-VL"
    export TASK_QUEUE_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

    vllm serve ${MODEL_PATH} \
              --max-num-batched-tokens 16384 \
              --served-model-name PaddleOCR-VL-0.9B \
              --trust-remote-code \
              --no-enable-prefix-caching \
              --mm-processor-cache-gb 0 \
              --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
              --additional_config '{"enable_cpu_binding":true}' \
              --port 8000
    ```

=== "Atlas 300 inference products"

    ```bash
    #!/bin/sh
    export VLLM_USE_MODELSCOPE=True
    export MODEL_PATH="PaddlePaddle/PaddleOCR-VL"
    export TASK_QUEUE_ENABLE=1
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

    vllm serve ${MODEL_PATH} \
              --max_model_len 16384 \
              --served-model-name PaddleOCR-VL-0.9B \
              --trust-remote-code \
              --no-enable-prefix-caching \
              --mm-processor-cache-gb 0 \
              --dtype float16 \
              --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
              --additional_config '{"ascend_compilation_config": {"fuse_norm_quant": false}, "enable_cpu_binding":true}' \
              --port 8000
    ```

!!! note

    On Atlas 300 inference products:

    - Only `float16` dtype is supported.
    - The `--max_model_len` option is added to prevent errors when generating the attention operator mask.
    - Graph compilation (`--compilation-config`) requires **CANN version >= 9.0.0**. If your CANN version is lower, please revert to eager mode by replacing the `--compilation-config` argument with `--enforce-eager`.
    - The `fuse_norm_quant` option in `--additional_config` is disabled (`false`) because it is not supported by the graph compilation on this hardware. Keep this setting unchanged.

Key Parameter Descriptions:

- `--max-num-batched-tokens` specifies the maximum number of tokens batched in a single forward pass. Adjust this parameter for throughput optimization.
- `--max_model_len` specifies the maximum context length — that is, the sum of input and output tokens for a single request.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--mm-processor-cache-gb` sets the size of the multimodal processor cache (in GB). A value of `0` disables caching.
- `--dtype float16` specifies the model dtype. On Atlas 300 inference products, only `float16` is supported.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full decode graph compilation for improved performance. On Atlas 300 inference products, `fuse_norm_quant` in graph compilation is disabled by default in `--additional_config`.
- `--additional_config '{"enable_cpu_binding":true}'` enables CPU binding to improve performance.

Common Issues Tip: If you encounter startup issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

### 5.2 Multi-Node PD Separation Deployment

Not supported yet.

### 5.3 Special Deployment Modes

#### 5.3.1 Offline Inference with vLLM and PP-DocLayoutV2

In the above example, we demonstrated how to use vLLM to infer the PaddleOCR-VL-0.9B model. Typically, we also need to integrate the PP-DocLayoutV2 model to fully unleash the capabilities of the PaddleOCR-VL model, making it more consistent with the examples provided by the official PaddlePaddle documentation.

!!! note

    Use separate virtual environments for VLLM and PP-DocLayoutV2 to prevent dependency conflicts.

=== "A2 series"

    The A2 series device supports inference using the PaddlePaddle framework.

    1. Pull the PaddlePaddle-compatible CANN image

        ```bash
        docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-aarch64-gcc84
        ```

        Start the container using the following command:

        ```bash
        docker run -it --name paddle-npu-dev -v $(pwd):/work \
            --privileged --network=host --shm-size=128G -w=/work \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
            ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-$(uname -m)-gcc84 /bin/bash
        ```

    2. Install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

        ```bash
        python -m pip install paddlepaddle==3.2.0
        wget https://paddle-whl.bj.bcebos.com/stable/npu/paddle-custom-npu/paddle_custom_npu-3.2.0-cp310-cp310-linux_aarch64.whl
        pip  install  paddle_custom_npu-3.2.0-cp310-cp310-linux_aarch64.whl
        python -m pip install -U "paddleocr[doc-parser]"
        pip install safetensors
        ```

        !!! note

                The OpenCV component may be missing:

                ```bash
                apt-get update
                apt-get install -y libgl1 libglib2.0-0
                ```

                CANN-8.0.0 does not support some versions of NumPy and OpenCV. It is recommended to install the specified versions.

                ```bash
                python -m pip install numpy==1.26.4
                python -m pip install opencv-python==3.4.18.65
                ```

=== "Atlas 300 inference products"

    The Atlas 300 inference products support only the OM model inference. For details about the process, see the guide provided in [ModelZoo](https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/ocr/PP-DocLayoutV2).

##### 5.3.2 Using vLLM as the backend, combined with PP-DocLayoutV2 for offline inference

```python
from paddleocr import PaddleOCRVL

doclayout_model_path = "/path/to/your/PP-DocLayoutV2/"

pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", 
                       vl_rec_server_url="http://localhost:8000/v1", 
                       layout_detection_model_name="PP-DocLayoutV2",  
                       layout_detection_model_dir=doclayout_model_path,
                       device="npu")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")

for i, res in enumerate(output):
    res.save_to_json(save_path=f"output_{i}.json")
    res.save_to_markdown(save_path=f"output_{i}.md")
```

## 6 Functional Verification

If your service starts successfully, you can see the info shown below:

```bash
INFO:     Started server process [87471]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can use the OpenAI API client to make queries.

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

# Task-specific base prompts
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": TASKS["ocr"]
            }
        ]
    }
]

response = client.chat.completions.create(
    model="PaddleOCR-VL-0.9B",
    messages=messages,
    temperature=0.0,
)
print(f"Generated text: {response.choices[0].message.content}")
```

Expected Result:

If you query the server successfully, you can see the info shown below (client):

```bash
Generated text: CINNAMON SUGAR
1 x 17,000
17,000
SUB TOTAL
17,000
GRAND TOTAL
17,000
CASH IDR
20,000
CHANGE DUE
3,000
```

## 7 Accuracy Evaluation

For the accuracy evaluation of PaddleOCR-VL, please refer to the official [ModelZoo](https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/ocr/PP-DocLayoutV2) for the evaluation process and results.

## 8 Performance Evaluation

For the performance evaluation of PaddleOCR-VL, please refer to the official [ModelZoo](https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/ocr/PP-DocLayoutV2) for the benchmark methodology and results.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, precision requirements, and actual hardware specifications. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

PaddleOCR-VL is a lightweight model that runs on a single NPU. The key tuning parameters differ between hardware platforms.

#### Table 1: Scenario Overview

| Scenario | Hardware | *Total NPUs | Weight Version | Key Considerations |
|----------|----------|------------|---------------|-------------------|
| High Throughput | A2 series | 1 | PaddleOCR-VL-0.9B | - |
| High Throughput | Atlas 300 inference products | 1 | PaddleOCR-VL-0.9B | Graph compilation requires **CANN >= 9.0.0** |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Model Len | Max Num Batched Tokens | Graph Compilation | dtype |
|----------|-------------|------|----|----|---------------|------------------------|--------------------|-------|
| High Throughput | A2 series / Single Machine | 1 | — | — | — | 16384 | FULL_DECODE_ONLY | bfloat16 (default) |
| High Throughput | Atlas 300 inference products / Single Machine | 1 | — | — | 16384 | — | FULL_DECODE_ONLY; otherwise enforce-eager | float16 |

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Section 5.1](#51-single-node-online-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

For performance tuning, please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods, including OS optimization (jemalloc, tcmalloc), `torch_npu` optimization (memory and scheduling), and CANN optimization.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html); this chapter only covers model-specific issues.

- **Q: What are the deployment requirements for Atlas 300 inference products?**

  A: On Atlas 300 inference products, only `float16` dtype is supported. Graph compilation (`--compilation-config`) requires **CANN version >= 9.0.0**; if your CANN version is lower, use `--enforce-eager` instead.

- **Q: What should I do if I encounter dependency conflicts during installation on Atlas 300 inference products?**

  A: You may need to uninstall `triton-ascend` to avoid dependency conflicts. See [Section 4.2](#42-source-code-installation) for details.
