# InternVL2-1B

## Introduction

InternVL2-1B is a multimodal large language model developed by OpenGVLab (Shanghai AI Lab), and is the smallest model in the InternVL 2.0 series. It consists of InternViT-300M-448px, an MLP projector, and Qwen2-0.5B-Instruct, with approximately 0.9 billion parameters in total. The model supports image, video, and text inputs, enabling capabilities such as visual question answering, image captioning, document understanding, and multimodal reasoning.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, NPU deployment, accuracy and performance evaluation.

This tutorial uses the vLLM-Ascend `v0.11.0rc3` version for demonstration, showcasing the `InternVL2-1B` model for single-NPU deployment.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

Require 1 Atlas 800I A2 (64G × 8) node or 1 Atlas 800 A3 (64G × 16) node:

- `InternVL2-1B`: [Download model weight](https://huggingface.co/OpenGVLab/InternVL2-1B)

It is recommended to download the model weight to the shared directory, such as `/root/.cache/`.

### Installation

Run docker container:

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

Setup environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

:::{note}
`max_split_size_mb` prevents the native allocator from splitting blocks larger than this size (in MB). This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory. You can find more details [<u>here</u>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/envref/envref_07_0061.html).
:::

## Deployment

### Offline Inference

Run the following script to execute offline inference on single-NPU:

```python
from vllm import LLM, SamplingParams

MODEL_PATH = "OpenGVLab/InternVL2-1B"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=32768,
    trust_remote_code=True,
    dtype="bfloat16",
    limit_mm_per_prompt={"image": 10},
)

sampling_params = SamplingParams(
    max_tokens=512
)

image_url = "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"

outputs = llm.chat(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Please provide a detailed description of this image"},
            ],
        },
    ],
    sampling_params=sampling_params,
)

print(outputs[0].outputs[0].text)
```

### Online Serving

Run docker container to start the vLLM server on single-NPU:

```{code-block} bash
   :substitutions:
vllm serve OpenGVLab/InternVL2-1B \
--trust-remote-code \
--dtype bfloat16 \
--max-model-len 32768 \
--enforce-eager
```

:::{note}
Add `--max-model-len` option to limit the context length. InternVL2-1B is trained with an 32K context window. Adjust this value according to available HBM on your NPU series.
:::

If your service starts successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "OpenGVLab/InternVL2-1B",
    "messages": [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```

## Accuracy Evaluation

### Using Language Model Evaluation Harness

As an example, take the `mmmu_val` dataset as a test dataset, and run accuracy evaluation of `InternVL2-1B` in offline mode.

1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for more details on `lm_eval` installation.

   ```shell
   pip install lm_eval
   ```

2. An evaluation configuration can be created for InternVL2-1B.

   ```shell
   pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
       --config tests/e2e/models/configs/InternVL2-1B.yaml
   ```

3. After execution, you can get the result. Here is the result of `InternVL2-1B` for reference:

   | tasks | version | Filter | metric | mode | result |
   | ----- | ----- | ----- | ----- | ----- | ----- |
   | mmmu_val | - | none | accuracy | gen | 0.3089 |

## Performance

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

The performance evaluation must be conducted in an online mode. Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model OpenGVLab/InternVL2-1B --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
