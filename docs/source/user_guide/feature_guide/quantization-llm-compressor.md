# llm-compressor Quantization Guide

Model quantization is a technique that reduces the size and computational requirements of a model by lowering the data precision of the weights and activation values in the model, thereby saving the memory and improving the inference speed.

## Supported llm-compressor Quantization Types

Support CompressedTensorsW8A8 static weight

weight: per-channel, int8, symmetric; activation: per-tensor, int8, symmetric.

Support CompressedTensorsW8A8Dynamic weight

weight: per-channel, int8, symmetric; activation: per-token, int8, symmetric, dynamic.

## Install llm-compressor

To quantize a model, you should install [llm-compressor](https://github.com/vllm-project/llm-compressor/blob/main/README.md). It is a unified library for creating compressed models for faster inference with vLLM.

Install llm-compressor

```bash
pip install llmcompressor
```

### Generate the W8A8 weights

```bash
cd examples/quantization/llm-compressor

python3 w8a8_int8_dynamic.py
```

for more details, see the [Official Sample](https://github.com/vllm-project/llm-compressor/tree/main/examples).

## Run the model

Now, you can run the quantized model with vLLM Ascend. Examples for online and offline inference are provided as follows:

### Offline inference

```python
import torch

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="{quantized_model_save_path}",
          max_model_len=2048,
          trust_remote_code=True)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Online inference

Start the quantized model using vLLM Ascend; no modifications to the startup command are required.
