import os

import numpy as np
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("QWEN35_EVS_MODEL", "/workspace/shared_assets/models/Qwen/Qwen3.5-2B")
PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
    "Describe this video in one sentence.<|im_end|>\n<|im_start|>assistant\n"
)
COMMON = dict(
    dtype="bfloat16", max_model_len=4096, max_num_batched_tokens=4096,
    limit_mm_per_prompt={"video": 1}, mm_processor_kwargs={"fps": 1},
    enforce_eager=True, gpu_memory_utilization=0.75,
)


def _video():
    frames = []
    for i in range(8):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 24
        frame[:, :, 1] = 255 - i * 24
        frame[i * 4 : i * 4 + 16] = 255
        frames.append(frame)
    video = np.stack(frames)
    return video, {"fps": 1.0, "duration": 8.0, "total_num_frames": 8,
                   "frames_indices": list(range(8)), "video_backend": "opencv",
                   "do_sample_frames": False}


def _run(rate):
    with VllmRunner(MODEL, video_pruning_rate=rate, **COMMON) as runner:
        result = runner.model.generate(
            runner.get_inputs([PROMPT], videos=[_video()]),
            sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
        )[0]
    return len(result.prompt_token_ids), result.outputs[0].text


@wait_until_npu_memory_free()
def test_qwen35_evs_baseline_is_stable():
    _, text = _run(0.0)
    assert text


@wait_until_npu_memory_free()
def test_qwen35_evs_pruning_is_stable():
    _, text = _run(0.75)
    assert text


@wait_until_npu_memory_free()
def test_qwen35_evs_reduces_prompt_tokens():
    baseline_tokens, baseline_text = _run(0.0)
    pruned_tokens, pruned_text = _run(0.75)
    assert baseline_text and pruned_text
    assert pruned_tokens < baseline_tokens
