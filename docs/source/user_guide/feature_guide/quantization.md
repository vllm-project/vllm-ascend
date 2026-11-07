# Quantization Guide

Model quantization is a technique that reduces model size and computational overhead by lowering the numerical precision of weights and activations, thereby saving memory and improving inference speed.

`vLLM Ascend` supports multiple quantization methods. This guide provides instructions for using different quantization tools and running quantized models on vLLM Ascend.

> **Note**
>
> You can choose to convert the model yourself or use the quantized model we uploaded.
> See <https://www.modelscope.cn/models/vllm-ascend/Kimi-K2-Instruct-W8A8>.
> Before you quantize a model, ensure sufficient RAM is available.

## Quantization Tools

vLLM Ascend supports models quantized by two main tools: `ModelSlim` and `LLM-Compressor`.

### 1. ModelSlim (Recommended)

[ModelSlim](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) is an Ascend-friendly compression tool focused on acceleration, using compression techniques, and built for Ascend hardware. It includes a series of inference optimization technologies such as quantization and compression, aiming to accelerate large language dense models, MoE models, multimodal understanding models, multimodal generation models, etc.

#### Installation

To use ModelSlim for model quantization, install it from its [Git repository](https://gitcode.com/Ascend/msmodelslim):

```bash
# Install 26.0.0 version, this is currently the latest stable branch
git clone https://gitcode.com/Ascend/msmodelslim.git -b 26.0.0

cd msmodelslim

bash install.sh
```

#### Model Quantization

The following example shows how to generate W8A8 quantized weights for the [Qwen3-MoE model](https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen3-MOE/README.md).

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

For more examples, refer to the [official examples](https://gitcode.com/Ascend/msmodelslim/tree/master/example).

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

The quantization types currently supported by LLM-Compressor can be viewed in the `vllm_ascend/quantization/compressed_tensors_config.py` file.

### 3. Native FP8 checkpoints (no quantization step needed)

Many models are now published directly in FP8, for example [Qwen/Qwen3.8-27B-FP8](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8) and [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). Their `config.json` carries `"quant_method": "fp8"` together with a `weight_block_size`, meaning every quantized weight is stored as `float8_e4m3fn` plus one `float32` scale per weight block. vLLM Ascend detects and serves these checkpoints as published, so no offline re-quantization is required:

```bash
vllm serve /path/to/Qwen3.8-27B-FP8 --trust-remote-code
vllm serve /path/to/GLM-5.3-Flash --tensor-parallel-size 8 --trust-remote-code
```

How the weights are executed depends on the hardware:

- **950PR&950DT Products**: the block scales are re-grouped into MXFP8 at load time, so weights stay at one byte per element and the native FP8 matmul is used.
- **Other Ascend generations**: the block scales are resolved into the model dtype at load time and served by the BF16 matmul. Numerically equivalent to the checkpoint, but plan for roughly twice the weight memory.

Layers the checkpoint left unquantized, such as `visual.merger.*` on multimodal models, are listed in `ignored_layers` or `modules_to_not_convert` and are served unquantized.

Only the block-wise flavour is supported. A native FP8 checkpoint without `weight_block_size` (per-tensor or per-channel scales) has no Ascend execution path yet; re-quantize it with ModelSlim or LLM-Compressor instead.

### 4. MXFP8/MXFP4 quantized models (Ascend 950 products only)

Microscaling (MX) quantization stores one `float8_e8m0` scale per group of
elements (group size 32 by default) instead of one scale per tensor or
channel, keeping quantized weights at one byte per element or below. vLLM
Ascend supports the following ModelSlim MX recipes:

| Recipe | Weight dtype | Activation dtype | Implementation |
|---|---|---|---|
| `W8A8_MXFP8` | `float8_e4m3fn` | `float8_e4m3fn` | `vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py` |
| `W4A4_MXFP4` | `float4_e2m1fn_x2` | `float4_e2m1fn_x2` | `vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py` |
| `W4A8_MXFP4` | `float4_e2m1fn_x2` | `float8_e4m3fn` | `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py` |

> **Note**
>
> MX quantization is an Ascend 950 feature: other Ascend generations have no
> MX execution path. On A2/A3, serve the W8A8 INT8 or BF16 releases of a model
> instead. Native block-wise FP8 checkpoints are also regrouped into MXFP8 at
> load time on 950 products (see section 3).

#### Generating MX quantized weights

MX weights are generated with [ModelSlim](https://gitcode.com/Ascend/msmodelslim),
the same tool as in section 1. For example, the
[MiniMax-M2 W8A8 MXFP8 example](https://gitcode.com/Ascend/msmodelslim/blob/master/example/MiniMax-M2/README.md)
shows how to quantize a MoE model into the W8A8 MXFP8 recipe.

#### Serving

Serve an MX checkpoint like any other ModelSlim-quantized model: the recipe is
read from the checkpoint quantization config, so `--quantization ascend` (plus
the usual parallelism flags) is all that is required.

Ready-made MX checkpoints include:

| Checkpoint | Recipe | Reference deployment |
|---|---|---|
| [GLM-5.3-Flash-w8a8-mxfp8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8) | W8A8 MXFP8, with dynamic FP8 KV cache | 1 node of 950DT products (96GB x 8) |
| [GLM-5.1-w4a4c8-mxfp4](https://www.modelscope.cn/models/Eco-Tech/GLM-5.1-w4a4c8-mxfp4) | W4A4 MXFP4 | 1 950DT product (96GB x 8) |

```bash
vllm serve /path/to/GLM-5.3-Flash-w8a8-mxfp8 \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

The command above also enables the dynamic FP8 KV cache (C8) shipped with this
checkpoint. Checkpoints that carry an MXFP8 KV recipe instead
(`K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL`) require `--kv-cache-dtype mxfp8`;
vLLM Ascend prints a warning when such a recipe is present but the flag is
not set.

For full deployment walkthroughs see the
[GLM-5.3-Flash](../../tutorials/models/GLM5.3-Flash.md) and
[GLM-5.1](../../tutorials/models/GLM5.md) tutorials.

## Running Quantized Models

Once you have a quantized model which is generated by **ModelSlim**, you can use vLLM Ascend for inference by specifying the `--quantization ascend` parameter to enable quantization features, while for models quantized by **LLM-Compressor**, it is not necessary to add this parameter.

### Offline Inference

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

### Online Inference

```bash
# Corresponding to offline inference
python -m vllm.entrypoints.api_server \
    --model /path/to/your/quantized_model \
    --max-model-len 4096 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --served-model-name quantized_model \
    --trust-remote-code 
```

## References

- [ModelSlim GitCode](https://gitcode.com/Ascend/msmodelslim)
- [LLM-Compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/features/quantization/)
