from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=16384,
    limit_mm_per_prompt={"image": 10},
)

sampling_params = SamplingParams(max_tokens=512)

image_messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role":
        "user",
        "content": [
            {
                "type": "image",
                "image":
                "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {
                "type": "text",
                "text": "Please provide a detailed description of this image"
            },
        ],
    },
]

messages = image_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
