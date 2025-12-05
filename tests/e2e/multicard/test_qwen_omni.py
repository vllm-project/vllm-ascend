import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


@pytest.fixture(scope="function")
def llm_engine():
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=20480,
    )
    return llm


def test_qwen_omni_multimodal_inputs(llm_engine):
    # Sampling Parameters
    sampling_params = SamplingParams(temperature=0.7,
                                     top_p=0.8,
                                     repetition_penalty=1.05,
                                     max_tokens=512)

    # multi-model inputs (OpenAI Chat Format)
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url":
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
            },
        }, {
            "type": "audio_url",
            "audio_url": {
                "url":
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
            }
        }, {
            "type": "video_url",
            "video_url": {
                "url":
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
            }
        }, {
            "type": "text",
            "text": "Analyze this audio, image, and video together."
        }]
    }]

    outputs = llm_engine.chat(messages=messages,
                              sampling_params=sampling_params)

    assert outputs is not None, "Output should not be None"
    assert len(outputs) > 0, "Should return at least one output"

    output_text = outputs[0].outputs[0].text
    print(
        f"\n[Output] Model generated text:\n{'-'*20}\n{output_text}\n{'-'*20}")

    # Check whether the audio, image, and video content are correctly understood.
    assert len(output_text.strip()) > 0, "Generated text should not be empty"
    assert "cough" in output_text.lower() and "mercedes" in output_text.lower(
    ) and "drawing" in output_text.lower()
