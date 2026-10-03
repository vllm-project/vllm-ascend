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

#### Low-Rank MoE Conversion

ModelSlim 1.0.0 DeepSeek-V3 checkpoints with per-channel W4A8 routed
experts can be converted to packed INT4 low-rank factors. The conversion is
resumable and does not retain the original dense expert matrices:

```bash
python benchmarks/moe_svd/convert.py \
    --input /path/to/modelslim_model \
    --output /path/to/low_rank_model \
    --rank 1024 \
    --workers 16 \
    --threads 4
```

The input and output directories must be different. The source expert weights
must use symmetric, zero-offset, per-channel W4A8 quantization. The rank must
be a multiple of 32, no larger than either expert dimension, and satisfy
`rank * (hidden_size + moe_intermediate_size) < hidden_size * moe_intermediate_size`
so that the packed expert weights contain fewer elements. Hidden and intermediate
dimensions must also be multiples of 32.

Conversion records SHA-256 fingerprints of source shards and checksums of output
tensors. Resuming validates shard contents, shapes, and dtypes before reusing
converted experts. Run one conversion process per output directory.

Serve the converted checkpoint with Ascend quantization and expert
parallelism when tensor parallelism is greater than one:

```bash
vllm serve /path/to/low_rank_model \
    --quantization ascend \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

Low-rank MoE inference uses BF16 activations, bias-free SiLU experts, and static
expert placement. SwiGLU clipping is preserved when configured. LoRA adapters
are rejected when loading a low-rank checkpoint.

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
