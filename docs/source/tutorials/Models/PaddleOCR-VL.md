# PaddleOCR-VL

## Introduction

PaddleOCR-VL is a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition.

This document provides a detailed workflow for the complete deployment and verification of the model, including supported features, environment preparation, single-node deployment, and functional verification. It is designed to help users quickly complete model deployment and verification.

## Supported Features

Refer to [supported features](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html) to get the model's supported feature matrix.

Refer to [feature guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/index.html) to get the feature's configuration.

## Environment Preparation

### Model Weight

* `PaddleOCR-VL-0.9B`: [PaddleOCR-VL-0.9B](https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL)

It is recommended to download the model weights to a local directory (e.g., `./PaddleOCR-VL`) for quick access during deployment.

### Installation

You can using our official docker image to run `PaddleOCR-VL` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

```{code-block} bash
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

PaddleOCR-VL supports single-node single-card deployment on the 910B4 platform. Follow these steps to start the inference service:

1. Prepare model weights: Ensure the downloaded model weights are stored in the `PaddleOCR-VL` directory.
2. Create and execute the deployment script (save as `deploy.sh`):

```shell
#!/bin/sh
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

#### Multiple NPU (PaddleOCR-VL)

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

## Offline Inference with vLLM and PP-DocLayoutV2

In the above example, we demonstrated how to use vLLM to infer the PaddleOCR-VL-0.9B model. Typically, we also need to integrate the PP-DocLayoutV2 model to fully unleash the capabilities of the PaddleOCR-VL model, making it more consistent with the examples provided by the official PaddlePaddle documentation.

:::{note}
Use separate virtual environments for VLLM and PPdoclayoutV2 to prevent dependency conflicts.
:::

### Pull the PaddlePaddle-compatible CANN image

Obtaining Ascend Images from PaddlePaddle:

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

### Install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

```bash
python -m pip install paddlepaddle==3.2.0
wget https://paddle-whl.bj.bcebos.com/stable/npu/paddle-custom-npu/paddle_custom_npu-3.2.0-cp310-cp310-linux_aarch64.whl
pip  install  paddle_custom_npu-3.2.0-cp310-cp310-linux_aarch64.whl
python -m pip install -U "paddleocr[doc-parser]"
pip install safetensors
```

:::{note}
The OpenCV component may be missing：

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

CANN-8.0.0 does not support some versions of NumPy and OpenCV. It is recommended to install the specified versions.

```bash
python -m pip install numpy==1.26.4
python -m pip install opencv-python==3.4.18.65
```

:::

### Using vLLM as the backend, combined with PP-DocLayoutV2 for offline inference

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
