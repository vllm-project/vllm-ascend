# Qwen3-Omni-30B-A3B-Thinking

## 1 Introduction

Qwen3-Omni is a native end-to-end multilingual omni-modal foundation model. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. We introduce several architectural upgrades to improve performance and efficiency. The Thinking model of Qwen3-Omni-30B-A3B, which contains the thinker component, is equipped with chain-of-thought reasoning and supports audio, video, and text input, with text output.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

The Qwen3-Omni-30B-A3B model is first supported in v0.12.0rc1. This document is validated and written based on vLLM-Ascend v0.22.1rc. All v0.22.1rc and later versions can run stably. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Refer to [supported features](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/support_matrix/supported_models.html) to get the model's supported feature matrix.

Refer to [feature guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/index.html) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory accessible to all nodes.

| Model                | Hardware Requirement                                                                             | Download                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Qwen3-Omni-30B-A3B (BF16) | Atlas 800I A3 (64G, 1\~2 cards)<br>Atlas 800I A2 (64G, 2\~4 cards) | [Download](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B)          |
| Qwen3-Omni-30B-A3B-W8A8   | Atlas 800I A3 (64G, 1\~2 cards)<br>Atlas 800I A2 (64G, 2\~4 cards)                               |  N/A|

the W8A8 quantized weights are not available for direct download, you can obtain them by quantizing the BF16 model using **msmodelslim**. Refer to the [Quantization Guide](../../user_guide/feature_guide/quantization.md) for details. All model paths in this document should be adjusted to your actual local paths.

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

:::{note}
Qwen3-30B-A3B-W8A8 adopts a hybrid quantization strategy (ordered by model structure):

- **Embedding layer**: BF16 (no quantization)
- **Q/K normalization** (q_norm, k_norm): BF16
- **Attention projections** (q/k/v/o_proj): Static W8A8 with pre-computed per-tensor scales
- **MoE routing gate** (mlp.gate): BF16
- **MoE expert projections** (gate/up/down_proj): Dynamic W8A8 where input scales are computed on-the-fly during inference
:::

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 MoE models.

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

**Docker Run:**

:::::{tab-set}
:sync-group: hardware

::::{tab-item} Atlas 800I A3
:sync: a3

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run \
    --name vllm-ascend-env \
    --shm-size=128g \
    --net=host \
    --privileged=true \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /data:/data \
    -v /tmp:/tmp \
    -v /mnt:/mnt \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -v /root:/host_root \
    -it -d $IMAGE bash
```

:::{note}
A3 has 8 NPUs with dual-die design (16 chips total: `/dev/davinci[0-15]`).
If you are on a shared machine, map only the chips you need (e.g., `/dev/davinci[0-7]` for NPU 0-3).
:::

::::

::::{tab-item} Atlas 800I A2
:sync: a2

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run \
    --name vllm-ascend-env \
    --shm-size=128g \
    --net=host \
    --privileged=true \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /data:/data \
    -v /tmp:/tmp \
    -v /mnt:/mnt \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -v /root:/host_root \
    -it -d $IMAGE bash
```

::::

:::::

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

**Installation Verification:**

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source. Install vLLM from source first:

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install the vLLM-Ascend repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

:::{note}
If deploying a multi-node environment, set up the environment on each node.
:::

Please install system dependencies

```bash
pip install qwen_omni_utils modelscope
# Used for audio processing.
apt-get update && apt-get install -y ffmpeg
# Check the installation.
ffmpeg -version
```

Required to avoid HcclAllreduce failures caused by the default FFTS+ mode's stream and shape limitations.

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
```

## 5 Deployment

### 5.1 Single-node Deployment

#### 5.1.1 Offline Inference on Multi-NPU

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
    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
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

#### 5.1.2 Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:
For an Atlas A2 with 64 GB of NPU card memory, tensor-parallel-size should be at least 1, and for 32 GB of memory, tensor-parallel-size should be at least 2.

```bash
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
```

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --tensor-parallel-size 2 --enable_expert_parallel
```

## 6 Functional Verification

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
The output should resemble the following.
```
{
  "id": "chatcmpl-8f3a1c2e4b5d4f7a9c0e1d2b3a4f5e6d",
  "object": "chat.completion",
  "created": 1782714240,
  "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "Let me examine each modality. The image shows several cars parked in a row — looks like a street or lot, daytime. The audio is short and sharp, sounds like a human cough (a single dry cough). The video appears to show someone drawing/sketching on paper, a hand moving a pen. Now I need to connect them: there isn't a single obvious narrative linking cars + cough + drawing, so I should describe each and then note any plausible shared context (e.g., a person sketching cars while having a cold). I'll keep the synthesis honest rather than forcing a connection.",
        "content": "Here's an analysis of the three inputs together:\n\n**Image** — A row of parked cars photographed outdoors in daylight, likely on a street or in a parking area. The vehicles are of varying colors and models, shot at a slight angle so several are visible in perspective.\n\n**Audio** — A brief, dry human cough: a single forceful exhalation lasting under a second, with no speech or background music. It sounds like an isolated cough rather than a sustained coughing fit.\n\n**Video** — A close-up of a hand drawing/sketching with a pen on paper, the strokes building up a simple illustration over a few seconds.\n\n**Combined interpretation** — The three clips don't share an explicit storyline; they're distinct samples of vision, sound, and motion. If a connecting context is assumed, one plausible scene is a person sketching cars (the drawing in the video, the cars in the image) while momentarily coughing (the audio) — e.g., an artist working outdoors who has a cold. But strictly, each input stands on its own: a static photo of cars, a one-off cough sound, and a short hand-drawing clip.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 8423,
    "total_tokens": 8712,
    "completion_tokens": 289,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null
}

```

## 7 Accuracy Evaluation

Here are accuracy evaluation methods.

### 7.1 Using EvalScope

As an example, take the `gsm8k` `omnibench` `bbh` dataset as a test dataset, and run accuracy evaluation of `Qwen3-Omni-30B-A3B-Thinking` in online mode.

1. Refer to Using evalscope(<https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_evalscope.html#install-evalscope-using-pip>) for `evalscope`installation.
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
        --generation-config '{"max_completion_tokens": 10000, "temperature": 0.6}' \
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

## 8 Performance

### 8.1 Using vLLM Benchmark  

Run performance evaluation of `Qwen3-Omni-30B-A3B-Thinking` as an example.
Refer to vllm benchmark for more details.
Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```bash
export VLLM_USE_MODELSCOPE=True
export MODEL=Qwen/Qwen3-Omni-30B-A3B-Thinking
python3 -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 2 --swap-space 16 --disable-log-stats --disable-log-request --load-format dummy

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install -r vllm-ascend/benchmarks/requirements-bench.txt

vllm bench serve --model $MODEL --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
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
## 9 FAQ
