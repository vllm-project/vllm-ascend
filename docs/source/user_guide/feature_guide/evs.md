# Efficient Video Sampling (EVS)

## Overview

Efficient Video Sampling (EVS) reduces redundant video tokens before they are
processed by the language model. vLLM-Ascend reuses the upstream EVS pipeline
for Qwen3.5 video inputs, including token selection and M-RoPE position
reconstruction.

EVS can reduce language-model prefill work and KV cache usage. It runs after
visual encoding, so it does not skip the vision encoder for the input video.

## Requirements

- A Qwen3.5 model and processor that support the upstream multimodal-pruning
  interface. The example below uses `Qwen/Qwen3.5-2B`.
- A working vLLM-Ascend installation. See the
  [installation guide](../../getting_started/installation.md).

## Usage

### Online inference

Set `--video-pruning-rate` when starting the server:

```bash
vllm serve Qwen/Qwen3.5-2B \
    --limit-mm-per-prompt.video 1 \
    --mm-processor-kwargs '{"fps": 1}' \
    --video-pruning-rate 0.75
```

Send video inputs through the standard multimodal API. For example:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    },
                },
                {"type": "text", "text": "Describe this video in one sentence."},
            ],
        }
    ],
    max_tokens=64,
    temperature=0,
)
print(response.choices[0].message.content)
```

### Offline inference

Pass `video_pruning_rate` to `LLM`:

```python
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset

if __name__ == "__main__":
    video = VideoAsset("baby_reading", num_frames=8)
    llm = LLM(
        model="Qwen/Qwen3.5-2B",
        limit_mm_per_prompt={"video": 1},
        mm_processor_kwargs={"fps": 1},
        video_pruning_rate=0.75,
    )
    outputs = llm.generate(
        [
            {
                "prompt": (
                    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
                    "Describe this video in one sentence.<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
                "multi_modal_data": {"video": (video.np_ndarrays, video.metadata)},
            }
        ],
        sampling_params=SamplingParams(max_tokens=64, temperature=0),
    )
    print(outputs[0].outputs[0].text)
```

## Configuration

`video_pruning_rate` is the fraction of video tokens targeted for pruning.
Use a value greater than `0` and less than `1` to enable EVS. Omitting the
option or setting it to `0` disables pruning.

For example, `0.75` targets pruning 75% of video tokens. EVS retains at least
the first temporal grid's visual tokens. Text, timestamps, and video separator
tokens also remain, so the reduction in total prompt tokens can be smaller
than the configured rate.

Higher pruning rates can remove details needed by the task. Compare output
quality and performance with pruning disabled on the target video workload
before choosing a rate.

When Triton Ascend is available, unquantized Qwen3.5 models use deterministic
projection kernels on the EVS path to reduce batch-dependent numerical
differences. These kernels can add overhead, so measure end-to-end latency
as well as token counts. Quantized models and installations without Triton
keep their existing projection kernels; cross-batch output consistency has
not been validated for those configurations.
