# Adapted from
# https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/tests/lora/test_llama.py
import os
from dataclasses import dataclass
from typing import Union

import pytest
import vllm
from vllm.lora.request import LoRARequest

from tests.conftest import VllmRunner


@dataclass
class ModelWithQuantization:
    model_path: str
    quantization: Union[str, None]


MODELS: list[ModelWithQuantization]
MODELS = [
    ModelWithQuantization(model_path="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
                          quantization=None),
    # ModelWithQuantization(
    #     model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
    #     quantization="AWQ"),  #AWQ quantization is currently not supported in ROCm. (Ref: https://github.com/vllm-project/vllm/blob/f6518b2b487724b3aa20c8b8224faba5622c4e44/tests/lora/test_quant_model.py#L23)
    # ModelWithQuantization(
    #     model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
    #     quantization="GPTQ"),
]

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["VLLM_USE_MODELSCOPE"] = "True"


def do_sample(llm: vllm.LLM,
              lora_path: str,
              lora_id: int,
              max_tokens: int = 256) -> list[str]:
    raw_prompts = [
        "Give me an orange-ish brown color",
        "Give me a neon pink color",
    ]

    def format_prompt_tuples(prompt):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    prompts = [format_prompt_tuples(p) for p in raw_prompts]

    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=max_tokens,
                                          stop=["<|im_end|>"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_lora(tinyllama_lora_files, model):

    if model.quantization is None:
        expected_no_lora_output = [
            "Here are some examples of orange-brown colors",
            "I'm sorry, I don't have"
        ]
        expected_lora_output = [
            "#ff8050",
            "#ff8080",
        ]
    elif model.quantization == "AWQ":
        expected_no_lora_output = [
            "I'm sorry, I don't understand",
            "I'm sorry, I don't understand",
        ]
        expected_lora_output = [
            "#f07700: A v",
            "#f00000: A v",
        ]
    elif model.quantization == "GPTQ":
        expected_no_lora_output = [
            "I'm sorry, I don't have",
            "I'm sorry, I don't have",
        ]
        expected_lora_output = [
            "#f08800: This is",
            "#f07788 \n#",
        ]

    def expect_match(output, expected_output):
        # HACK: GPTQ lora outputs are just incredibly unstable.
        # Assert that the outputs changed.
        if (model.quantization == "GPTQ"
                and expected_output is expected_lora_output):
            assert output != expected_no_lora_output
            for i, o in enumerate(output):
                assert o.startswith(
                    '#'), f"Expected example {i} to start with # but got {o}"
            return
        assert output == expected_output

    max_tokens = 10

    print("creating lora adapter")
    with VllmRunner(model_name=model.model_path,
                    quantization=model.quantization,
                    enable_lora=True,
                    max_loras=4,
                    max_model_len=400,
                    gpu_memory_utilization=0.7,
                    max_num_seqs=16) as vllm_model:
        print("no lora")
        output = do_sample(vllm_model.model,
                           tinyllama_lora_files,
                           lora_id=0,
                           max_tokens=max_tokens)
        expect_match(output, expected_no_lora_output)

        print("lora 1")
        output = do_sample(vllm_model.model,
                           tinyllama_lora_files,
                           lora_id=1,
                           max_tokens=max_tokens)
        expect_match(output, expected_lora_output)

        print("no lora")
        output = do_sample(vllm_model.model,
                           tinyllama_lora_files,
                           lora_id=0,
                           max_tokens=max_tokens)
        expect_match(output, expected_no_lora_output)

        print("lora 2")
        output = do_sample(vllm_model.model,
                           tinyllama_lora_files,
                           lora_id=2,
                           max_tokens=max_tokens)
        expect_match(output, expected_lora_output)

        print("removing lora")
