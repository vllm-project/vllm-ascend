# Qwen3-Omni-30B-A3B-Thinking

## Introduction

Qwen3-Omni is the natively end-to-end multilingual omni-modal foundation models. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. We introduce several architectural upgrades to improve performance and efficiency. The Thinking model of Qwen3-Omni-30B-A3B, containing the thinker component, equipped with chain-of-thought reasoning, supporting audio, video, and text input, with text output.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## Supported Features
Refer to [supported features](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/support_matrix/supported_models.html) to get the model's supported feature matrix.

Refer to [feature guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/index.html) to get the feature's configuration.

## Environment Preparation
### Model Weight

- `Qwen3-Omni-30B-A3B-Thinking` require 2 NPU Card(64G × 2).[Download model weight](https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Thinking)
It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

:::::{tab-set}
::::{tab-item} Use docker image

You can using our official docker image to run Qwen3-Omni-30B-A3B-Thinking directly

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

```{code-block} bash
  :substitutions:
# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
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

::::
::::{tab-item} Build from source

You can build all from source.

- Install `vllm-ascend`, refer to [set up using python](../installation.md#set-up-using-python).

::::
:::::

Please install system dependencies

```bash
pip install qwen_omni_utils modelscope
# Used for audio processing.
apt-get update && apt-get install ffmpeg -y
# Check the installation.
ffmpeg -version
```

## Deployment
### Single-node Deployment
#### Offline Inference on Multi-NPU

Run the following script to execute offline inference on multi-NPU:

```python
import gc
import torch
import os
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel
)
from modelscope import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

os.environ["HCCL_BUFFSIZE"] = "1024"

def clean_up():
    """Clean up distributed resources and NPU memory"""
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()  # Garbage collection to free up memory
    torch.npu.empty_cache()


def main():
    MODEL_PATH = "Qwen3/Qwen3-Omni-30B-A3B-Thinking"
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        limit_mm_per_prompt={'image': 5, 'video': 2, 'audio': 3},
        max_model_len=32768,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"},
                {"type": "text", "text": "What can you see and hear? Answer in one sentence."}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # 'use_audio_in_video = True' requires equal number of audio and video items, including audio from the video. 
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": True}
    }
    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()


if __name__ == "__main__":
    main()
```

#### Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:
For an Atlas A2 with 64 GB of NPU card memory, tensor-parallel-size should be at least 1, and for 32 GB of memory, tensor-parallel-size should be at least 2.

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --tensor-parallel-size 2 --enable_expert_parallel
```

## Functional Verification
Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8000/v1/chat/completions \
-X POST \
-H "Content-Type: application/json" \
-d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
                    }
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
                    }
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
                    }

                },
                {
                    "type": "text",
                    "text":  "Analyze this audio, image, and video together."
                }
            ]
        }
    ]
}'
```

## Accuracy Evaluation

Here are accuracy evaluation methods.

### Using EvalScope

As an example, take the `gsm8k` `omnibench` `bbh` dataset as a test dataset, and run accuracy evaluation of `Qwen3-Omni-30B-A3B-Thinking` in online mode.
1. Refer to Using evalscope(https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_evalscope.html#install-evalscope-using-pip) for `evalscope`installation.
2. Run `evalscope` to execute the accuracy evaluation.

```bash
evalscope eval \
    --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --api-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --eval-type server \
    --datasets omni_bench, gsm8k, bbh \
    --dataset-args '{"omni_bench": { "extra_params": { "use_image": true, "use_audio": false}}}' \
    --eval-batch-size 1 \
    --generation-config '{"max_tokens": 10000, "temperature": 0.6}' \
    --limit 100
```

3. After execution, you can get the result, here is the result of `Qwen3-Omni-30B-A3B-Thinking` in vllm-ascend:0.13.0rc1 for reference only.

```bash
 +-----------------------------+------------+----------+----------+-------+---------+---------+
| Model                       | Dataset    | Metric   | Subset   |   Num |   Score | Cat.0   |
+=============================+============+==========+==========+=======+=========+=========+
| Qwen3-Omni-30B-A3B-Thinking | omni_bench | mean_acc | default  |   100 |    0.44 | default |
+-----------------------------+------------+----------+----------+-------+---------+---------+ 
| Qwen3-Omni-30B-A3B-Thinking | gsm8k      | mean_acc | main     |   100 |    0.98 | default |
+-----------------------------+-----------+----------+----------+-------+---------+---------+
| Qwen3-Omni-30B-A3B-Thinking | bbh        | mean_acc | OVERALL  |   270 |  0.9148 |         |
+-----------------------------+------------+----------+----------+-------+---------+---------+
```

## Performance

### Using vLLM Benchmark  
Run performance evaluation of `Qwen3-Omni-30B-A3B-Thinking` as an example.
Refer to vllm benchmark for more details.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```bash
VLLM_USE_MODELSCOPE=True 
export MODEL=Qwen/Qwen3-Omni-30B-A3B-Thinking
python3 -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 2 --swap-space 16 --disable-log-stats --disable-log-request --load-format dummy

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install -r vllm-ascend/benchmarks/requirements-bench.txt

vllm bench serve --model $MODEL --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After execution, you can get the result, here is the result of `Qwen3-Omni-30B-A3B-Thinking` in vllm-ascend:0.13.0rc1 for reference only.

```bash
============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Request rate configured (RPS):           1.00
Benchmark duration (s):                  211.90
Total input tokens:                      40000
Total generated tokens:                  25600
Request throughput (req/s):              0.94
Output token throughput (tok/s):         120.81
Peak output token throughput (tok/s):    216.00
Peak concurrent requests:                24.00
Total token throughput (tok/s):          309.58
---------------Time to First Token----------------
Mean TTFT (ms):                          215.50
Median TTFT (ms):                        211.51
P99 TTFT (ms):                           317.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          98.96
Median TPOT (ms):                        99.19
P99 TPOT (ms):                           101.52
---------------Inter-token Latency----------------
Mean ITL (ms):                           99.02
Median ITL (ms):                         96.10
P99 ITL (ms):                            176.02
==================================================
```
