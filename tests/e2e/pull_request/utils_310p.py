import os

from PIL import Image

from tests.e2e.conftest import VllmRunner


def get_test_image():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "..", "prompts", "qwen.png")
    return Image.open(image_path)


def get_test_prompts():
    return ["<|image_pad|>Describe this image in detail."]


def run_vl_model_test(
    model_name: str, tensor_parallel_size: int, max_tokens: int, dtype: str = "float16", enforce_eager: bool = True
):
    image = get_test_image()
    images = [image]
    prompts = get_test_prompts()

    with VllmRunner(
        model_name, tensor_parallel_size=tensor_parallel_size, enforce_eager=enforce_eager, dtype=dtype
    ) as vllm_model:
        vllm_model.generate_greedy(prompts, max_tokens, images=images)
