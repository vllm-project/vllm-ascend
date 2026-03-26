# DeepSeek-OCR-2

## Introduction

DeepSeek-OCR-2 is a recent vision-language model developed by DeepSeek for advanced optical character recognition and document understanding. Compared to traditional OCR systems, it not only extracts text but also understands document structure, such as layouts, tables, and formulas.

The model introduces an improved visual encoding approach that better captures reading order and semantic relationships, enabling more accurate and consistent interpretation of complex documents.

With a relatively compact design, DeepSeek-OCR-2 achieves strong performance while remaining efficient, making it suitable for real-world deployment in document analysis and processing tasks.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `DeepSeek-OCR-2` (BF16 version): [Download model weight](https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR-2)

It is recommended to download the model weights to a local directory (e.g., `./DeepSeek-OCR-2/`) for quick access. Since this model uses a custom vision encoder architecture, **`trust_remote_code=True` needs to be enabled** during execution.

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/installation.md#verify-multi-node-communication).

### Installation

You can use our official docker image to run DeepSeek-OCR-2 directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/installation.md#set-up-using-docker).

```python
   :substitutions:
# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-it $IMAGE bash
```

## Deployment

Single Node Deployment is enough for DeepSeek-OCR-2.

### ⚠️ Critical Environment Configuration for Ascend

DeepSeek-OCR-2 requires dynamic loading of remote code (`trust_remote_code=True`). During the initial code loading phase, PyTorch's underlying initialization is triggered. In the Ascend/ARM architecture, if vLLM subsequently spawns multi-process engines using the default `fork` method, it will cause a severe conflict with the polluted OpenMP thread pool (resulting in an `Invalid thread pool` crash).

To ensure stable model startup, **you must set the following environment variables** to enforce isolated process contexts before running any inference scripts or API servers:

```bash
# Force vLLM to use 'spawn' instead of 'fork'
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Prevent OpenMP thread spinning deadlocks
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### Quick Test Script

You can run the following script to quickly verify the model's functionality.

```python
# test_deepseek_ocr_2.py
from vllm import LLM, SamplingParams
from PIL import Image
import gc

def main():

    llm = LLM(
        model="/path/to/DeepSeek-OCR-2",
        trust_remote_code=False,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16384,
    )

    image_url = "Your image URL here"

    try:
        image = Image.open(image_url).convert("RGB")
    except Exception as e:
        print(f"Image loading failed: {e}")
        return

    inputs = {
        "prompt": "<image>\nWhat's the content of the image?",
        "multi_modal_data": {
            "image": image
        }
    }

    outputs = llm.generate(inputs, sampling_params)

    print("\n--- Model Recognition Results ---")
    print(outputs[0].outputs[0].text)

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
```

> **Note:** Since DeepSeek-OCR-2 requires dynamic loading of remote code (with trust_remote_code=True), this triggers the underlying initialization of PyTorch in the main process in advance. On the Ascend/ARM architecture, this causes vLLM to encounter conflicts with the OpenMP thread pool when subsequently launching a multi-process engine through fork (resulting in an error message "Invalid thread pool"). 
To ensure that the model can start stably, before running any inference script or starting the API service, the following environment variables must be set in the terminal to forcibly isolate the multi-process context:

```bash
python test_deepseek_ocr_2.py
```

### Start API Server

After configuring the environment variables mentioned above, you can start an OpenAI-compatible API server. 

*Note: The vision encoder consumes significant memory. It is highly recommended to limit `max-model-len` and reduce `gpu-memory-utilization` (e.g., `0.7`) to prevent Out-Of-Memory (OOM) errors during the KV Cache initialization on 32GB NPU cards.*

```bash
vllm serve deepseek-ai/DeepSeek-OCR-2 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

### Test with cURL

Once the server is up and running, you can test the multimodal capabilities by sending an image to the chat completions endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Please extract all the text from this image."},
          {
            "type": "image_url",
            "image_url": {
              "url": "/path/to/your/image.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.0
  }'
```
*Note: You can also replace the URL with a base64 encoded string (`"url": "data:image/jpeg;base64,<BASE64_STRING>"`) to test local images.*

## Accuracy Evaluation

### Using Language Model Evaluation Harness (lm_eval)

Currently, the `lm_eval` framework lacks suitable full-text pure OCR transcription datasets. The `doc_vqa` (Document Visual Question Answering) dataset is used here **solely for smoke-testing the end-to-end execution pipeline** to prevent hardware regression. 

Since DeepSeek-OCR-2 is a base model without a predefined Chat Template, using a few-shot prompt (`--num_fewshot 3`) is mandatory to regularize its output format for this specific evaluation task.

First, disable the ModelScope hijacking to ensure `lm_eval` can successfully download standard datasets from HuggingFace:

```bash
export VLLM_USE_MODELSCOPE=False
export HF_ENDPOINT=https://hf-mirror.com
```

Then, run the evaluation against your running local API server:

```bash
lm_eval --model local-chat-completions \
   --model_args model=deepseek-ai/DeepSeek-OCR-2,base_url=http://localhost:8000/v1/chat/completions \
   --tasks doc_vqa \
   --num_fewshot 3
```

| dataset | version | metric | mode | note |
|----- | ----- | ----- | ----- | ----- |
| doc_vqa | 0 | anls | gen | Atlas A2 (32G × 1) |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for detailed guidelines.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three vllm bench subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.
- 
Take the serve as an example. Run the code as follows.

```bash
vllm bench serve --model deepseek-ai/DeepSeek-OCR-2  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1
```

After about several minutes, you can get the performance evaluation result.

**Results:**

On a single Atlas A2 (32G × 1) card, the performance metrics for DeepSeek-OCR-2 are as follows:

| Category | Metric | Value |
|----------|--------|-------|
| Throughput | Request Throughput | 0.99 req/s |
| Throughput | Output Token Throughput | 126.75 tok/s |
| Latency | TTFT (Mean) | 48.93 ms |
| Latency | TPOT (Mean) | 14.49 ms/token |
| Concurrency | Peak Concurrent Requests | 9 |
| Reliability | Failed Requests | 0 |