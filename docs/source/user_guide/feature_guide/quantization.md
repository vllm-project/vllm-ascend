# Quantization Guide

Model quantization is a technique that reduces model size and computational overhead by lowering the numerical precision of weights and activations, thereby saving memory and improving inference speed.

`vLLM Ascend` supports multiple quantization methods. This guide provides instructions for using different quantization tools and running quantized models on vLLM Ascend.

> **Note**
>
> You can choose to convert the model yourself or use the quantized model we uploaded.
> See <https://www.modelscope.cn/models/vllm-ascend/Kimi-K2-Instruct-W8A8>.
> Before you quantize a model, ensure sufficient RAM is available.

## Quantization Tools

vLLM Ascend supports models quantized by three main tools: `ModelSlim`, `LLM-Compressor`, and `GPTQModel`.

### 1. ModelSlim (Recommended)

[ModelSlim](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/README.md) is an Ascend-friendly compression tool focused on acceleration, using compression techniques, and built for Ascend hardware. It includes a series of inference optimization technologies such as quantization and compression, aiming to accelerate large language dense models, MoE models, multimodal understanding models, multimodal generation models, etc.

#### Installation

To use ModelSlim for model quantization, install it from its [Git repository](https://gitcode.com/Ascend/msit):

```bash
# Install br_release_MindStudio_8.3.0_20261231 version
git clone https://gitcode.com/Ascend/msit.git -b br_release_MindStudio_8.3.0_20261231

cd msit/msmodelslim

bash install.sh
```

#### Model Quantization

The following example shows how to generate W8A8 quantized weights for the [Qwen3-MoE model](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/Qwen3-MOE/README.md).

**Quantization Script:**

```bash
cd example/Qwen3-MOE

# Support multi-card quantization
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# Set model and save paths
export MODEL_PATH="/path/to/your/model"
export SAVE_PATH="/path/to/your/quantized_model"

# Run quantization script
python3 quant_qwen_moe_w8a8.py --model_path $MODEL_PATH \
--save_path $SAVE_PATH \
--anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
--calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
--trust_remote_code True
```

After quantization completes, the output directory will contain the quantized model files.

For more examples, refer to the [official examples](https://gitcode.com/Ascend/msit/tree/master/msmodelslim/example).

### 2. LLM-Compressor

[LLM-Compressor](https://github.com/vllm-project/llm-compressor) is a unified compressed model library for faster vLLM inference.

#### Installation

```bash
pip install llmcompressor
```

#### Model Quantization

`LLM-Compressor` provides various quantization scheme examples.

##### Dense Quantization

An example to generate W8A8 dynamic quantized weights for dense model:

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic.py
```

##### MoE Quantization

An example to generate W8A8 dynamic quantized weights for MoE model:

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic_moe.py
```

For more content, refer to the [official examples](https://github.com/vllm-project/llm-compressor/tree/main/examples).

Currently supported quantization types by LLM-Compressor: `W8A8` and `W8A8_DYNAMIC`.

### 3. GPTQModel

[GPTQModel](https://github.com/ModelCloud/GPTQModel) is a post-training quantization (PTQ) library that implements the GPTQ algorithm for compressing large language models. GPTQ uses layer-wise quantization to achieve high compression ratios while maintaining model accuracy.

#### Installation

```bash
pip install gptqmodel
```

#### Model Quantization

GPTQModel supports quantizing models to 4-bit or 8-bit precision. The following example shows how to quantize a model using GPTQModel:

```python
from gptqmodel import GPTQModel, QuantizeConfig

# Define quantization configuration
quantize_config = QuantizeConfig(
    bits=8,  # 4 or 8 bits
    group_size=128,  # Quantization group size
    desc_act=False,  # Must be False for Ascend NPU
)

# Load and quantize model
model = GPTQModel.from_pretrained(
    "/path/to/your/model",
    quantize_config=quantize_config,
)

# Quantize the model with calibration data
model.quantize(calibration_dataset)

# Save quantized model
model.save_quantized("/path/to/save/quantized_model")
```

**Important Notes:**

- Ascend NPU currently supports **4-bit and 8-bit** quantization only (2-bit and 3-bit are not supported)
- The `desc_act` parameter must be set to `False` (activation reordering is not supported on Ascend NPU)
- Pre-quantized GPTQ models are available on ModelScope, such as [Qwen/Qwen3-0.6B-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-GPTQ-Int8)

For more details and examples, refer to the [GPTQModel documentation](https://github.com/ModelCloud/GPTQModel).

## Running Quantized Models

Once you have a quantized model, you can use vLLM Ascend for inference:

- For models quantized by **ModelSlim**, specify the `--quantization ascend` parameter
- For models quantized by **LLM-Compressor**, no additional parameter is needed
- For models quantized by **GPTQModel**, specify the `--quantization gptq` parameter

### Offline Inference

**Example 1: ModelSlim Quantized Model**

```python
import torch

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Set sampling parameters
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/path/to/your/quantized_model",
          max_model_len=4096,
          trust_remote_code=True,
          # Set appropriate TP and DP values
          tensor_parallel_size=2,
          data_parallel_size=1,
          # Set an unused port
          port=8000,
          # Set serving model name
          served_model_name="quantized_model",
          # Specify `quantization="ascend"` to enable quantization for models quantized by ModelSlim
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

**Example 2: GPTQ Quantized Model**

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
]

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Load GPTQ quantized model
llm = LLM(
    model="Qwen/Qwen3-0.6B-GPTQ-Int8",
    max_model_len=512,
    trust_remote_code=True,
    gpu_memory_utilization=0.7,
    # Specify `quantization="gptq"` to enable GPTQ quantization
    quantization="gptq",
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Online Inference

**Example 1: ModelSlim Quantized Model**

```bash
# Corresponding to offline inference
python -m vllm.entrypoints.api_server \
    --model /path/to/your/quantized_model \
    --max-model-len 4096 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --served-model-name quantized_model \
    --trust-remote-code \
    --quantization ascend
```

**Example 2: GPTQ Quantized Model**

```bash
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen3-0.6B-GPTQ-Int8 \
    --max-model-len 512 \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --quantization gptq
```

## References

- [ModelSlim Documentation](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/README.md)
- [LLM-Compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [GPTQModel GitHub](https://github.com/ModelCloud/GPTQModel)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/quantization/)
