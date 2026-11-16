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

#### Qwen3.8 Flash Next PLE n-gram table INT8

Qwen3.8 Flash Next stores its large PLE n-gram embedding as checkpoint row shards named
`...ple.ple_embedding.ngram_embedding.shard_N.weight`. These are hashed n-gram vector rows,
not readable tokenizer vocabulary entries. The PLE table format is independent of the
W8A8 or MXFP8 recipe used for linear layers and MTP.

The [Qwen4Exp ModelSlim adapter](https://gitcode.com/code_mingming/msmodelslim/tree/master-qwen38-1022)
supports symmetric INT8 PLE tables. Add the following to the model recipe's `spec` section;
for MTP linear-layer selection, start from the
[MTP recipe](https://gitcode.com/code_mingming/msmodelslim/blob/master-qwen38-1022/lab_practice/qwen4_exp_flash_next/qwen4_w8a8_mtp.yaml)
and change its PLE setting from `bf16`:

```yaml
spec:
  model_adapter:
    ple_quant_type: int8_sym
    ple_quant_group_size: 32
```

The linked MTP recipe is a configuration reference; it does not establish that MTP
weights in an existing export were quantized. Check `quant_model_description.json`
in the exported model. If all `mtp.*` entries are marked `FLOAT`, its MTP weights
remain unquantized.

Run ModelSlim against the original BF16 checkpoint with the completed recipe:

```bash
python -m msmodelslim.cli quant \
  --model_path /path/to/Qwen3.8-Flash-Next \
  --save_path /path/to/new-quantized-model \
  --device npu --model_type Qwen3.8-Flash-Next \
  --config /path/to/qwen4_w8a8_mtp_ple.yaml \
  --trust_remote_code true
```

For each table row, ModelSlim groups 32 adjacent columns. For a group `x`, it computes
`scale = max(max(abs(x)) / 127, 1e-12)` and
`q = clamp(round(x / scale), -127, 127)`. The output holds `shard_N.weight` as signed
INT8 with the original `[rows, width]` shape, and `shard_N.scale` as FP32 with shape
`[rows, width / 32]`. Its `quant_model_description.json` records
`shard_N.quant_type: INT8_SYM`; `quant_model_weights.safetensors.index.json` maps both
tensors to their shard files. The smaller PLE projections, norms, convolution weights,
and integer hash-layout buffers retain their source dtypes.

At inference time, a selected row is reconstructed as
`weight[row, column].float() * scale[row, column // 32]` before the PLE projections.
This is lossy: use the original BF16 checkpoint when exact source values are needed.
Check every table has a matching scale with the expected shape, then compare sampled
reconstructed rows and model outputs against the BF16 PLE baseline. A small table
reconstruction error alone does not establish generation accuracy.

If the starting point is an existing ModelSlim W8A8/MXFP8 export whose PLE tables are
still BF16, convert only its PLE table shards into a **new** model directory. Preserve
other weight shards, add the scale keys to the index, set the `INT8_SYM` description
markers, and recompute the index's `metadata.total_size` from the actual tensor
payloads. Do not run the full-model quantizer on an already quantized checkpoint.

The current Ascend PLE INT8 runtime path keeps the table on device and dequantizes
only looked-up rows. Pinned-host PLE offload is not supported for this format and
fails explicitly during model initialization.

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
