# PaddleOCR-VL

## Introduction

PaddleOCR-VL 是一个专为文档解析设计的 SOTA 和资源高效模型。其核心组件是 PaddleOCR-VL-0.9B，一个紧凑但强大的视觉语言模型（VLM），集成了 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型，实现了精准的元素识别。

本文档详细介绍了模型的完整部署和验证工作流程，包括支持的功能、环境准备、单节点部署、功能验证、准确性和性能评估。它旨在帮助用户快速完成模型部署和验证。

## Supported Features

Refer to [supported features](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html) to get the model's supported feature matrix.

Refer to [feature guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/index.html) to get the feature's configuration.

## Environment Preparation

### Model Weight

* `PaddleOCR-VL-0.9B`：require 1 910B4 cards(32G × 1). [PaddleOCR-VL-0.9B](https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL)

It is recommended to download the model weights to a local directory (e.g., `./PaddleOCR-VL`) for quick access during deployment.

### Installation

You can using our official docker image to run `PaddleOCR-VL` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

```{code-block}
:substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc1
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

## Deployment

### Single-node Deployment

#### Single NPU (PaddleOCR-VL)

:::{note}
PaddleOCR-VL supports single-node single-card deployment on the 910B4 platform. Follow these steps to start the inference service:

1. Prepare model weights: Ensure the downloaded model weights are stored in the `PaddleOCR-VL` directory.

```bash
export VLLM_USE_MODELSCOPE=true
export MODEL_PATH="PaddlePaddle/PaddleOCR-VL"

vllm serve ${MODEL_PATH} \
          --max-num-batched-tokens 16384 \
          --served-model-name PaddleOCR-VL-0.9B \
          --trust-remote-code \
          --no-enable-prefix-caching \
		  --mm-processor-cache-gb 0 \
		  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

#### Multiple NPU (Qwen2.5-Omni-7B)

Single-node deployment is recommended.

### Prefill-Decode Disaggregation

Not supported yet

## Functional Verification

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [87471]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can使用 OpenAI API 客户端查询

```bash
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

## 结合 vLLM 与 PP-DocLayoutV2 进行离线推理

在上述示例中，我们演示了利用vLLM推断PaddleOCR-VL的方法。通常，我们还需要整合PP-DocLayoutV2模型，充分释放PaddleOCR-VL模型的能力，使其更符合PaddlePaddle官方提供的示例。

:::{note}
Use separate virtual environments for and to prevent dependency conflicts.

### 拉取paddle配套cann镜像

```
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-aarch64-gcc84
```

启动容器，参考如下命令

```
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-$(uname -m)-gcc84 /bin/bash
```

### Install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

```bash
python -m pip install paddlepaddle==3.2.0
wget https://paddle-whl.bj.bcebos.com/stable/npu/paddle-custom-npu/paddle_custom_npu-3.2.0-cp310-cp310-linux_aarch64.whl
pip  install  paddle_custom_npu-3.2.0-cp310-cp310-linux_aarch64.whl
python -m pip install -U "paddleocr[doc-parser]"
pip install safetensors
```

:::{note}
可能缺少opencv组件：

```
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

CANN-8.0.0 对 numpy 和 opencv 部分版本不支持，建议安装指定版本

```
python -m pip install numpy==1.26.4
python -m pip install opencv-python==3.4.18.65
```

## Accuracy Evaluation

Qwen2.5-Omni on vllm-ascend has been test on AISBench.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.
2. After execution, you can get the result, here is the result of `Qwen2.5-Omni-7B` with `vllm-ascend:0.11.0rc0` for reference only.

| dataset | platform | metric | mode | vllm-api-stream-chat |
|----- | ----- | ----- | ----- | -----|
| textVQA | A2 | accuracy | gen_base64 | 83.47 |
| textVQA | A3 | accuracy | gen_base64 | 84.04 |

## Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen2.5-Omni-7B` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model Qwen/Qwen2.5-Omni-7B --dataset-name random --random-input 1024 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

