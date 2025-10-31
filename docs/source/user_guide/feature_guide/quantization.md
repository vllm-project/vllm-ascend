# Quantization Guide

Model quantization is a technique that reduces the size and computational requirements of a model by lowering the data precision of the weights and activation values in the model, thereby saving the memory and improving the inference speed.

Since version 0.9.0rc2, the quantization feature is experimentally supported by vLLM Ascend. Users can enable the quantization feature by specifying `--quantization ascend`. Currently, only Qwen, DeepSeek series models are well tested. We will support more quantization algorithms and models in the future.

## Install ModelSlim

To quantize a model, you should install [ModelSlim](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md) which is the Ascend compression and acceleration tool. It is an affinity-based compression tool designed for acceleration, using compression as its core technology and built upon the Ascend platform.

Install ModelSlim:

```bash
# The branch(br_release_MindStudio_8.1.RC2_TR5_20260624) has been verified
git clone -b br_release_MindStudio_8.1.RC2_TR5_20260624 https://gitee.com/ascend/msit

cd msit/msmodelslim

bash install.sh
pip install accelerate
```

## Quantize model

:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded.
See https://www.modelscope.cn/models/vllm-ascend/Kimi-K2-Instruct-W8A8.
This conversion process requires a larger CPU memory, ensure that the RAM size is greater than 2 TB.
:::

### Adapt to changes
1. Ascend does not support the `flash_attn` library. To run the model, you need to follow the [guide](https://gitee.com/ascend/msit/blob/master/msmodelslim/example/DeepSeek/README.md#deepseek-v3r1) and comment out certain parts of the code in `modeling_deepseek.py` located in the weights folder.
2. The current version of transformers does not support loading weights in FP8 quantization format. you need to follow the [guide](https://gitee.com/ascend/msit/blob/master/msmodelslim/example/DeepSeek/README.md#deepseek-v3r1) and delete the quantization related fields from `config.json` in the weights folder.

### Generate the W8A8 weights

```bash
cd example/DeepSeek

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export MODEL_PATH="/root/.cache/Kimi-K2-Instruct"
export SAVE_PATH="/root/.cache/Kimi-K2-Instruct-W8A8"

python3 quant_deepseek_w8a8.py --model_path $MODEL_PATH --save_path $SAVE_PATH --batch_size 4
```

Here is the full converted model files except safetensors:

```bash
.
|-- config.json
|-- configuration.json
|-- configuration_deepseek.py
|-- generation_config.json
|-- modeling_deepseek.py
|-- quant_model_description.json
|-- quant_model_weight_w8a8_dynamic.safetensors.index.json
|-- tiktoken.model
|-- tokenization_kimi.py
`-- tokenizer_config.json
```

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
          trust_remote_code=True,
          # Enable quantization by specifying `quantization="ascend"`
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Online inference

Enable quantization by specifying `--quantization ascend`, for more details, see the [DeepSeek-V3-W8A8 Tutorial](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_node.html).

## FAQs

### 1. How to solve the KeyError "xxx.layers.0.self_attn.q_proj.weight"?

First, make sure you specify `ascend` as the quantization method. Second, check if your model is converted by the `br_release_MindStudio_8.1.RC2_TR5_20260624` ModelSlim version. Finally, if it still does not work, submit an issue. Maybe some new models need to be adapted.

### 2. How to solve the error "Could not locate the configuration_deepseek.py"?

Please convert DeepSeek series models using `br_release_MindStudio_8.1.RC2_TR5_20260624` ModelSlim, where the missing configuration_deepseek.py error has been fixed.

### 3. What should be considered when converting DeepSeek series models with ModelSlim?

When the MLA portion of the weights used the `W8A8_DYNAMIC` quantization with the torchair graph mode enabled, modify the configuration file in the CANN package to prevent incorrect inference results.

The operation steps are as follows:

1. Search in the CANN package directory, for example:
find /usr/local/Ascend/ -name fusion_config.json

2. Add `"AddRmsNormDynamicQuantFusionPass":"off",` and `"MultiAddRmsNormDynamicQuantFusionPass":"off",` to the fusion_config.json you find, the location is as follows:

```bash
{
    "Switch":{
        "GraphFusion":{
            "AddRmsNormDynamicQuantFusionPass":"off",
            "MultiAddRmsNormDynamicQuantFusionPass":"off",
```
